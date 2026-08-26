# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project overview

Undergraduate CS thesis on Text-to-SQL with **open-weights models** on Apple Silicon (MLX). Natural-language questions are translated into SQL against the [Spider](https://yale-lily.github.io/spider) benchmark, using few-shot retrieval, schema linking, and cross-consistency (semantic majority vote over execution results).

Call these **open-weights models**, not open-source LLMs: only the weights are available to run locally; many of the licenses are not open source.

## Environment setup

Uses `uv` (not conda). Python is pinned to 3.12.9 via `.python-version` (`requires-python >= 3.12.9`).

```bash
xcode-select --install         # first-time macOS toolchain
brew install uv wget
cp .env.example .env           # set ROOT_PATH (absolute path to this repo)
sh init.sh                     # uv sync, download Spider, populate DBs, build embedding index
```

All scripts read `ROOT_PATH` (required) and `TMP_DIR` (optional, defaults to `/tmp`) from `.env` via `python-dotenv`. Run scripts from the project root.

`init.sh` exits if `.env` is missing, then installs deps (`uv sync`), downloads Spider into `$TMP_DIR/spider_data`, copies SQLite DBs to `database/spider/`, and runs `src.pipeline.ingest` and `src.pipeline.embedding`.

## Common commands

`--limit N` takes a **deterministic random sample of up to N questions per difficulty** (seed `42` in `src/util/sampling.py`). Each difficulty has an independent RNG, so adding or dropping difficulties does not change the other samples. Omit `--limit` to use every matching row. `--difficulty` accepts one or more of `easy medium hard extra`.

```bash
# End-to-end: first model for preSQL, all models for finSQL (cross-consistency if 2+)
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --models Llama-3.2-3B-Instruct-4bit Qwen3.5-9B-MLX-4bit gemma-3-12b-it-4bit-DWQ \
    --source test --difficulty hard --limit 100 \
    --batch-size 10

# Explicit per-step models
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --presql-model Llama-3.2-3B-Instruct-4bit \
    --finsql-models Llama-3.2-3B-Instruct-4bit Qwen3.5-9B-MLX-4bit gemma-3-12b-it-4bit-DWQ \
    --source test --difficulty hard --limit 100

# Baseline: skip preSQL / schema linking; full-schema prompt goes straight to finSQL
uv run python -m src.ml.run \
    --config OpenText2SQL.json --skip-presql \
    --models Llama-3.2-3B-Instruct-4bit \
    --source test --difficulty hard --limit 100

# Standalone steps
uv run python -m src.ml.gen_presql --config OpenText2SQL.json --model Llama-3.2-3B-Instruct-4bit --source test --difficulty hard --limit 100
uv run python -m src.ml.gen_finsql --presql experiments/.../presql.jsonl --config OpenText2SQL.json --models Llama-3.2-3B-Instruct-4bit Qwen3.5-9B-MLX-4bit
uv run python -m src.ml.gen_metrics experiments/.../presql.jsonl experiments/.../finsql.jsonl --raw-metrics

# Fine-tuning (QLoRA) and adapter inference
uv run python -m src.util.finetune --model mlx-community/Llama-3.2-3B-Instruct-4bit
uv run python -m src.ml.predict --model mlx-community/Llama-3.2-3B-Instruct-4bit --source test --limit 100

# Tests
uv run pytest tests/

# Reset generated DBs and .venv (add --all to also delete the Spider download under TMP_DIR)
sh clean.sh
```

`--models` cannot be combined with `--presql-model` / `--finsql-models`. With `--skip-presql`, provide `--finsql-models` or `--models`.

## Querying the database

Inspect `database/OpenText2SQL.db` (created by `init.sh`). `gold_dataset` is a normal table; `embedding_dataset` is a `sqlite-vec` `vec0` virtual table and needs the extension loaded first.

### `gold_dataset`

```bash
sqlite3 database/OpenText2SQL.db "
SELECT source, difficulty, COUNT(*) AS n
FROM gold_dataset
GROUP BY source, difficulty
ORDER BY source, difficulty;
"

sqlite3 database/OpenText2SQL.db "
SELECT id, db_id, source, difficulty, question, query
FROM gold_dataset
WHERE source = 'test' AND difficulty = 'hard'
LIMIT 10;
"
```

### `embedding_dataset`

**DBeaver:** enable loading extensions on the SQLite connection, then run once per session (replace `ROOT_PATH` with the value from `.env`):

```sql
SELECT load_extension('ROOT_PATH/.venv/lib/python3.12/site-packages/sqlite_vec/vec0.dylib');
```

Then:

```sql
SELECT id, db_id, source, question, skeleton_question
FROM embedding_dataset
LIMIT 10;
```

**CLI / agents:** load via `sqlite_vec` (same as `src/util/nlp.py`), not bare `sqlite3`:

```bash
uv run python - <<'PY'
import os, sqlite3
import sqlite_vec
from dotenv import load_dotenv

load_dotenv(".env")
db = os.path.join(os.environ["ROOT_PATH"], "database/OpenText2SQL.db")
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
for row in conn.execute(
    "SELECT id, db_id, source, question, skeleton_question FROM embedding_dataset LIMIT 10"
):
    print(row)
PY
```

`MATCH` queries need a 300-dim float32 blob; see `get_few_shot` in `src/util/nlp.py`.

## Architecture

### Data pipeline (`src/pipeline/`)

Runs once via `init.sh`; produces `database/OpenText2SQL.db`. Per-database Spider files live in `database/spider/`.

1. **`ingest.py`** — one pass into four tables:
   - `bronze_dataset` — raw Q/SQL pairs
   - `spider_tables` — raw schema metadata
   - `silver_dataset` — cleaned, schema-enriched, difficulty-labelled rows
   - `gold_dataset` — curated rows consumed by ML scripts (`is_valid = 1`)
2. **`embedding.py`** — few-shot vector index (`sqlite-vec`) as `embedding_dataset` in the same DB. Retrieval lives in `src/util/nlp.py`.

### ML pipeline (`src/ml/`)

```
preSQL → finSQL → metrics
```

Output: `experiments/<YYYY-MM-DD_HH-MM-SS>/`.

- **`run.py`** — all three steps. Extra flags: `--top-k-few-shot` (default 3), `--batch-size`, `--max-tokens` (default 512), `--etype` (Spider eval bucket, default `all`). Always writes `raw_metrics.txt`.
- **`gen_presql.py`** — preliminary SQL for schema linking → `presql.jsonl`. Supports `--adapter-path` for `:fine-tuned` model specs.
- **`gen_finsql.py`** — loads `presql.jsonl`, schema-links, re-renders pruned prompts, runs cross-consistency → `finsql.jsonl` next to the input. Single model disables voting.
- **`gen_metrics.py`** — Spider execution accuracy + exact match (and clause F1s) on one or two JSONL files → `metrics.md`. `--raw-metrics` also writes `raw_metrics.txt`. One file: field auto-detected from the stem, or `--sql presql|finsql`.
- **`predict.py`** — inference with a LoRA adapter using `Finetune.json` (full DDL, no schema linking). Writes JSONL then calls `gen_metrics`. `--no-adapter` is the untuned baseline.

### Fine-tuning (`src/util/finetune.py`)

QLoRA on `gold_dataset` via `mlx_lm`. Adapters land in `config/adapter/<model-name>/`. Also writes a `finetune_dataset` table. Model specs may use `key:fine-tuned` in `infer()` when `adapter_path` is passed; cross-consistency does **not** support `:fine-tuned`.

### Utility (`src/util/`)

- **`llm.py`** — `resolve_model`, `parse_model_spec`, `infer`, `prompt_generation`, `render_prompt`, `schema_linking`, `cross_consistency`. Schema linking prunes `simplified_ddl` / `foreign_keys` / `cell_values` to tables in preSQL and sets `section_visibility`. Cross-consistency votes by execution equivalence on `database/spider/<db_id>`.
- **`nlp.py`** — `get_question_skeleton`, `get_few_shot`, `extract_referenced_tables_from_sql`.
- **`sampling.py`** — `sample_per_difficulty` (`SAMPLING_SEED = 42`). Used by prompt generation, `gen_finsql --limit`, and `predict`.

### LLM configs (`config/llm/`)

`models.json` maps short keys to `{ "path": "<hf-repo>", "tokenizer_config": {...} }`. Keys or full HuggingFace paths both work. Models download on first use. Extra entry fields besides `path` are passed to `mlx_lm.load`.

### Prompt configs (`config/prompt/`)

JSON sections with `text` (Jinja2) and `visible`. `render_prompt()` assembles them and accepts schema-linking visibility overrides.

| File | Role |
|---|---|
| `OpenText2SQL.json` | Main pipeline: schema, cell values, FKs, few-shot |
| `OpenText2SQL_no_fewshot.json` | Same without few-shot |
| `Baseline.json` | Schema + question only |
| `Finetune.json` | Full DDL + question (fine-tune / `predict.py`) |
| `OpenAI.json` | Alternate prompt layout |

### Output layout

```
experiments/<YYYY-MM-DD_HH-MM-SS>/
  presql.jsonl
  finsql.jsonl
  metrics.md
  raw_metrics.txt          # always from run.py; standalone gen_metrics needs --raw-metrics
```

### Tests (`tests/`)

`conftest.py` session fixtures (`root_path`, `db`, `gold_db`, `index_db`) skip when `ROOT_PATH` is unset or `OpenText2SQL.db` is missing. `test_sampling.py` does not need the DB.

```bash
uv run pytest tests/
```
