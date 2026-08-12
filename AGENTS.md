# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project overview

Undergraduate CS thesis exploring Text-to-SQL using open-source LLMs on Apple Silicon (MLX). The pipeline converts natural language questions into SQL queries against the [Spider](https://yale-lily.github.io/spider) benchmark dataset.

## Environment setup

Uses `uv` (not conda). Python version is pinned to 3.12.9 via `.python-version`.

```bash
uv sync                        # install dependencies
cp .env.example .env           # then set ROOT_PATH and TMP_DIR
sh init.sh                     # download Spider dataset, populate databases, build embedding index
```

All scripts read `ROOT_PATH` and `TMP_DIR` from `.env` via `python-dotenv`. Scripts must be run from the project root.

## Common commands

```bash
# End-to-end pipeline
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --models Qwen3-14B-4bit \
    --source test --difficulty hard --limit 100

# Standalone steps
uv run python -m src.ml.gen_presql --config OpenText2SQL.json --model Qwen3-14B-4bit --source dev
uv run python -m src.ml.gen_finsql --presql experiment/.../presql.jsonl --config OpenText2SQL.json --models Qwen3-14B-4bit
uv run python -m src.ml.gen_metrics experiment/.../presql.jsonl experiment/.../finsql.jsonl

# Tests
uv run pytest tests/

# Reset all generated data and databases
sh clean.sh
```

## Architecture

### Data pipeline (`src/pipeline/`)

Runs once via `init.sh`; produces a single SQLite database at `database/OpenText2SQL.db`.

1. **`ingest.py`** — ingests Spider JSON files into four tables in one pass:
   - `bronze_dataset` — raw Q/SQL pairs
   - `spider_tables` — raw schema metadata
   - `silver_dataset` — cleaned, schema-enriched, difficulty-labelled rows
   - `gold_dataset` — final curated dataset consumed by ML scripts

2. **`embedding.py`** — builds the few-shot vector index using `sqlite-vec`, adding an `embedding_dataset` table to `OpenText2SQL.db`. Retrieval utilities (`get_question_skeleton`, `get_few_shot`) live in `src/util/nlp.py`.

### ML pipeline (`src/ml/`)

#### `run.py` — end-to-end entry point

Executes all three steps and writes output to `experiment/<YYYY-MM-DD_HH-MM-SS>/`.

```
preSQL → finSQL → metrics
```

Key flags:
- `--models` — first model used for preSQL, all models used for finSQL cross-consistency
- `--presql-model` / `--finsql-models` — explicit per-step control
- `--skip-presql` — baseline mode: feeds full-schema prompts directly into finSQL
- `--source`, `--difficulty`, `--limit` — dataset filters
- `--batch-size`, `--max-tokens` — inference params

#### Standalone scripts

- **`gen_presql.py`** — generates preliminary SQL for schema linking; writes `presql.jsonl` to a timestamped experiment dir.
- **`gen_finsql.py`** — loads an existing `presql.jsonl`, applies schema linking, renders pruned prompts, runs cross-consistency, writes `finsql.jsonl` alongside it.
- **`gen_metrics.py`** — runs Spider evaluation on one or two JSONL files; writes `metrics.md` (and optionally `raw_metrics.txt`).

#### `models.json`

Registry mapping short keys (e.g. `Qwen3-14B-4bit`) to full HuggingFace paths. Models are downloaded automatically on first use. Pass short keys or full paths interchangeably — `resolve_model()` in `src/util/llm.py` handles both.

### Utility (`src/util/`)

- **`llm.py`** — public API: `resolve_model`, `infer`, `prompt_generation`, `render_prompt`, `schema_linking`, `cross_consistency`.
- **`nlp.py`** — NLP helpers: `get_question_skeleton` (masks schema tokens), `get_few_shot` (sqlite-vec nearest-neighbour retrieval).

### Prompt configs (`config/prompt/`)

JSON files (e.g. `OpenText2SQL.json`, `Baseline.json`) that define named sections with `text` and `visible` fields. `render_prompt()` assembles the final prompt from a config and a dataset record, and accepts per-section visibility overrides computed by schema linking.

### Output layout

All experiment output lands under:
```
experiment/<YYYY-MM-DD_HH-MM-SS>/
  presql.jsonl
  finsql.jsonl
  metrics.md
  raw_metrics.txt
```

### Tests (`tests/`)

Pytest suite. `conftest.py` provides session-scoped fixtures (`root_path`, `db`, `gold_db`, `index_db`) that skip automatically when `ROOT_PATH` is unset or `OpenText2SQL.db` is missing.

```bash
uv run pytest tests/
```
