# Natural Language to SQL Translation using Open-Weights Models

This project is part of my undergraduate thesis for a Bachelor's degree in Computer Science.

## 📌 Overview

The goal is to explore how large language models (LLMs) can be used to automatically translate natural language questions into SQL queries, making database interaction more accessible to non-technical users.

We focus on using **open-weights models** that can run on modest hardware, providing a cost-effective alternative to proprietary solutions. Only the weights are available to run locally; many of these models are not open source.

## 🧠 Core Ideas

- Evaluate and compare open-weights models for the Text-to-SQL task
- Explore effective prompt engineering techniques
- Address natural language ambiguities and complex database schemas
- Utilize methods like few-shot learning, schema linking, and self-consistency
- Benchmark with standard datasets such as [**Spider**](https://yale-lily.github.io/spider)

## 🔧 Tools & Techniques

- Prompt tuning and context injection
- Query evaluation based on execution accuracy and exact match
- Experiments with lightweight, locally deployable models

## 🛠️ Installation

1. Install development tools:
```bash
xcode-select --install
```

2. Install dependencies:
```bash
brew install uv wget
```

3. Clone this repo and navigate to it:
```bash
git clone https://github.com/dahue/unrc-cs-thesis.git && cd unrc-cs-thesis
```

4. Configure environment variables:
```bash
cp .env.example .env
# then edit .env and set ROOT_PATH to the absolute path of this repo
```

5. Run the initialization script (installs Python deps, downloads Spider, builds the DB):
```bash
sh init.sh
```

## 📚 Usage

### OpenText2SQL pipeline

The main entry point is `run.py`, which executes three steps end-to-end:

1. **preSQL** — generates a preliminary SQL query used exclusively for schema linking (identifying relevant tables/columns).
2. **finSQL** — builds a pruned prompt from the linked schema and runs inference, optionally with cross-consistency across multiple models.
3. **metrics** — evaluates both preSQL and finSQL against Spider gold SQL and writes `metrics.md` + `raw_metrics.txt`.

All output lands in a timestamped experiment directory: `experiments/<YYYY-MM-DD_HH-MM-SS>/`.

`--limit N` selects a deterministic random sample of up to `N` questions from each
matching difficulty, using the fixed experiment seed `42`. Omit `--limit` to include
all matching questions.

```bash
# Single model (no cross-consistency): first model used for preSQL, same model for finSQL
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --models Llama-3.2-3B-Instruct-4bit \
    --source test --difficulty hard --limit 100 \
    --batch-size 10

# Multiple models: first model for preSQL, all models for finSQL cross-consistency
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --models Llama-3.2-3B-Instruct-4bit Qwen3.5-9B-MLX-4bit gemma-3-12b-it-4bit-DWQ \
    --source test --difficulty hard --limit 100 \
    --batch-size 10

# Separate control over preSQL and finSQL models
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --presql-model Llama-3.2-3B-Instruct-4bit \
    --finsql-models Llama-3.2-3B-Instruct-4bit Qwen3.5-9B-MLX-4bit gemma-3-12b-it-4bit-DWQ \
    --source test --difficulty hard --limit 100 \
    --batch-size 10

```

Model short keys (defined in `config/llm/models.json`) map to their full HuggingFace paths and are downloaded automatically on first use.

### Standalone scripts

The three pipeline steps can also be run independently:

```bash
# Step 1: generate preSQL predictions
uv run python -m src.ml.gen_presql \
    --config OpenText2SQL.json \
    --model Llama-3.2-3B-Instruct-4bit \
    --source test --difficulty hard --limit 100 \
    --batch-size 10

# Step 2: generate finSQL from an existing presql.jsonl
uv run python -m src.ml.gen_finsql \
    --config OpenText2SQL.json \
    --presql experiments/2026-05-26_02-15-18/presql.jsonl \
    --models Llama-3.2-3B-Instruct-4bit Qwen3.5-9B-MLX-4bit gemma-3-12b-it-4bit-DWQ \

# Step 3: evaluate and export metrics
uv run python -m src.ml.gen_metrics \
    experiments/2026-05-26_02-15-18/presql.jsonl \
    experiments/2026-05-26_02-15-18/finsql.jsonl \
    --raw-metrics
```

## 🗄️ Querying the database

`init.sh` writes `database/OpenText2SQL.db`. Open that file in DBeaver (or any SQLite client).

### `gold_dataset`

Curated Spider rows used by the ML pipeline (`is_valid = 1`).

```sql
SELECT source, difficulty, COUNT(*) AS n
FROM gold_dataset
GROUP BY source, difficulty
ORDER BY source, difficulty;

SELECT id, db_id, source, difficulty, question, query
FROM gold_dataset
WHERE source = 'test' AND difficulty = 'hard'
LIMIT 10;
```

### `embedding_dataset`

Few-shot vector index (`sqlite-vec` `vec0` virtual table, training rows only). DBeaver cannot read it until the extension is loaded.

Enable loading extensions on the SQLite connection, then run this once per session (replace `ROOT_PATH` with the value from `.env`):

```sql
SELECT load_extension('ROOT_PATH/.venv/lib/python3.12/site-packages/sqlite_vec/vec0.dylib');
```

After that:

```sql
SELECT id, db_id, source, question, skeleton_question
FROM embedding_dataset
LIMIT 10;
```


## 👨‍💻 Author

Student: **Adrian Tissera**  
Thesis Director: **Dr. Pablo Ponzio**

## 📌 Resources

- [**Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task**](https://github.com/taoyds/spider)
- [**Text-To-SQL on spider**](https://paperswithcode.com/sota/text-to-sql-on-spider)
- [**MLX-LM: Large Language Models for MLX**](https://github.com/ml-explore/mlx-lm)

- [**Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation**](https://arxiv.org/pdf/2308.15363)
- [**PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency**](https://arxiv.org/pdf/2403.09732)
- [**C3: Zero-shot Text-to-SQL with ChatGPT**](https://arxiv.org/pdf/2307.07306)
- [**DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models**](https://arxiv.org/pdf/2402.01117)
- [**High Precision Natural Language Interfaces to Databases: a Graph Theoretic Approach**](https://aiweb.cs.washington.edu/research/projects/ai2/nli/aaai_submission.pdf)
- [**Towards a Theory of Natural Language Interfaces to Databases**](https://turing.cs.washington.edu/papers/nli-iui03.pdf)
- [**RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL**](https://arxiv.org/pdf/2302.05965v3)
- [**The Illusion of Thinking**](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)
