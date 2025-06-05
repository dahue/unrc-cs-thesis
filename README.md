# Natural Language to SQL Translation using Open-Source LLMs

This project is part of my undergraduate thesis for a Bachelor's degree in Computer Science.

## 📌 Overview

The goal is to explore how large language models (LLMs) can be used to automatically translate natural language questions into SQL queries, making database interaction more accessible to non-technical users.

We focus on using **open-source LLMs** that can run on modest hardware, providing a cost-effective alternative to proprietary solutions.

## 🧠 Core Ideas

- Evaluate and compare open-source LLMs for the Text-to-SQL task
- Explore effective prompt engineering techniques
- Address natural language ambiguities and complex database schemas
- Utilize methods like few-shot learning, self-consistency, and intermediate representations
- Benchmark with standard datasets such as [**Spider**](https://yale-lily.github.io/spider)

## 🔧 Tools & Techniques

- Prompt tuning and context injection
- Query evaluation based on accuracy and recall
- Experiments with lightweight, locally deployable models

## 🛠️ Installation

1. Install development tools:
```bash
xcode-select --install
```

2. Install miniforge:
```bash
brew install --cask miniforge
conda init zsh
source ~/.zshrc
```

3. Create a new conda environment:
```bash
conda create -n unrc-cs-thesis python=3.12.9
conda activate unrc-cs-thesis
```

4. Clone this repo and navigate to it:
```bash
git clone https://github.com/dahue/unrc-cs-thesis.git && cd unrc-cs-thesis
```

5. Set the project root path in a .env file:
```bash
echo "ROOT_PATH=$(pwd)" > .env
```
or manually:
```bash
echo "ROOT_PATH=/absolute/path/to/unrc-cs-thesis" > .env
```

6. Run the initialization script:
```bash
sh init.sh
```

## 📚 Usage

### Training set creation

```bash
python scripts/ML/create_training_set.py --model-type nl2SQL --template-name template_11.j2
```

### LLM Fine-tuning

```bash
python scripts/ML/finetune.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --model-type nl2SQL
```

### Prediction

```bash
python scripts/ML/predict.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --adapter data/adapters/nl2SQL/Llama-3.2-1B-Instruct-4b/ --input-file data/training/nl2SQL/template_11/t_valid.jsonl
```

### Benchmarking

```bash
python scripts/ML/benchmark.py --gold-file data/training/nl2SQL/template_11/t_valid.sql --predict-file data/predictions/pred.sql
```

## 📅 Timeline

Development started in **November 2024** and is expected to conclude by **June 2025**.

## 👨‍💻 Author

Student: **Adrian Tissera**  
Thesis Director: **Dr. Pablo Ponzio**

## 📌 Resources

- [**Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task**](https://github.com/taoyds/spider)
- [**Text-To-SQL on spider**](https://paperswithcode.com/sota/text-to-sql-on-spider)

- [**Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation**](https://arxiv.org/pdf/2308.15363)
- [**PET-SQL: A Prompt-Enhanced Two-Round Refinement of Text-to-SQL with Cross-consistency**](https://arxiv.org/pdf/2403.09732)
- [**C3: Zero-shot Text-to-SQL with ChatGPT**](https://arxiv.org/pdf/2307.07306)
- [**DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models**](https://arxiv.org/pdf/2402.01117)
- [**High Precision Natural Language Interfaces to Databases: a Graph Theoretic Approach**](https://aiweb.cs.washington.edu/research/projects/ai2/nli/aaai_submission.pdf)
- [**Towards a Theory of Natural Language Interfaces to Databases**](https://turing.cs.washington.edu/papers/nli-iui03.pdf)
- [**RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL**](https://arxiv.org/pdf/2302.05965v3)
