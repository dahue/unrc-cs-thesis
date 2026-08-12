# Natural Language to SQL Translation using Open-Source LLMs

This project is part of my undergraduate thesis for a Bachelor's degree in Computer Science.

## 📌 Overview

The goal is to explore how large language models (LLMs) can be used to automatically translate natural language questions into SQL queries, making database interaction more accessible to non-technical users.

We focus on using **open-source LLMs** that can run on modest hardware, providing a cost-effective alternative to proprietary solutions.

## 🧠 Core Ideas

- Evaluate and compare open-source LLMs for the Text-to-SQL task
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

4. Create a virtual environment and install dependencies:
```bash
uv sync
```

5. Configure environment variables:
```bash
cp .env.example .env
# then edit .env and set ROOT_PATH to the absolute path of this repo
```

6. Run the initialization script:
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

```bash
# Single model (no cross-consistency): first model used for preSQL, same model for finSQL
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --models Qwen3-14B-4bit \
    --source test --difficulty hard --limit 100

# Multiple models: first model for preSQL, all models for finSQL cross-consistency
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit gemma-3-12b-it-4bit-DWQ \
    --source test --difficulty hard --limit 100 \
    --batch-size 2

# Separate control over preSQL and finSQL models
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --presql-model Qwen3.5-9B-MLX-4bit \
    --finsql-models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit \
    --source test --difficulty hard --limit 100

# Baseline: skip preSQL and feed the full-schema prompt directly into finSQL
uv run python -m src.ml.run \
    --config OpenText2SQL.json \
    --skip-presql \
    --finsql-models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit \
    --source test --difficulty hard --limit 100
```

Model short keys (defined in `src/ml/models.json`) map to their full HuggingFace paths and are downloaded automatically on first use.

### Standalone scripts

The three pipeline steps can also be run independently:

```bash
# Step 1: generate preSQL predictions
uv run python -m src.ml.gen_presql \
    --config OpenText2SQL.json \
    --model Qwen3-14B-4bit \
    --source test --difficulty hard --limit 100

# Step 2: generate finSQL from an existing presql.jsonl
uv run python -m src.ml.gen_finsql \
    --presql experiments/2026-05-26_02-15-18/presql.jsonl \
    --config OpenText2SQL.json \
    --models Qwen3.5-9B-MLX-4bit Qwen3-14B-4bit

# Step 3: evaluate and export metrics
uv run python -m src.ml.gen_metrics \
    experiments/2026-05-26_02-15-18/presql.jsonl \
    experiments/2026-05-26_02-15-18/finsql.jsonl \
    --raw-metrics
```

## 📅 Timeline

Development started in **November 2024**.

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
