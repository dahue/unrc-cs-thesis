"""
llm.py — LLM inference utilities for MLX models.

Public API:
  resolve_model(key)                                                       -> str
  infer(model, prompts, batch_size=1, max_tokens=512, adapter_path=None) -> List[str]
  prompt_generation(config, db_path, ...)                                 -> List[Dict]
  render_prompt(config, rec, section_visibility=None)                     -> str
  cross_consistency(models, records, batch_size=1, max_tokens=512)       -> List[Dict]

Model keys are short names defined in scripts/ML/models.json (e.g. "Qwen3-14B-4bit").
Prompt configs are JSON dicts with sections keyed by name, each having "text" and "visible" fields.
"""

import re
import os
import sys
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, meta as jinja_meta
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

_MODELS_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "ML", "models.json")
)


def _resolve_model_entry(key: str) -> dict:
    """Return the full models.json entry for a key, normalised to a dict with at least 'path'."""
    with open(_MODELS_FILE, encoding="utf-8") as f:
        registry: dict = json.load(f)
    entry = registry.get(key, key)
    if isinstance(entry, str):
        return {"path": entry}
    return dict(entry)


def resolve_model(key: str) -> str:
    """
    Resolve a short model key to its full HuggingFace path using models.json.
    If key is not found in the registry, it is returned as-is (full path passthrough).
    """
    return _resolve_model_entry(key)["path"]


def parse_model_spec(model_spec: str) -> tuple[str, bool]:
    """Parse 'model_name' or 'model_name:fine-tuned' into (model_name, use_adapter)."""
    if not model_spec or not model_spec.strip():
        raise ValueError("Model specification cannot be empty")
    parts = model_spec.strip().split(":")
    if len(parts) == 1:
        return parts[0], False
    if len(parts) == 2:
        model_name, suffix = parts[0].strip(), parts[1].strip().lower()
        if not model_name:
            raise ValueError(f"Model name cannot be empty in: {model_spec}")
        if suffix == "fine-tuned":
            return model_name, True
        raise ValueError(f"Invalid suffix '{suffix}'. Expected 'fine-tuned' or no suffix.")
    raise ValueError(f"Invalid model spec: {model_spec}. Use 'model_name' or 'model_name:fine-tuned'")


def get_model_dir_name(name: str) -> str:
    """Strip common HuggingFace org prefix to get the local directory name."""
    if name.startswith("mlx-community/"):
        return name.removeprefix("mlx-community/")
    if "/" in name:
        return name.split("/")[-1]
    return name


def normalize_response(text: str) -> str:
    return " ".join(text.split())


def post_process_sql(sql_text: str) -> str:
    """Strip markdown, chain-of-thought tags, and common LLM verbosity from a SQL string."""
    if not sql_text:
        return ""

    if "</think>" in sql_text:
        sql_text = sql_text.split("</think>", 1)[1].strip()

    match = re.search(r"```(?:sql|sqlite)\b\s*(.*?)\s*```", sql_text, re.IGNORECASE | re.DOTALL)
    if match:
        sql_text = match.group(1).strip()
    else:
        match = re.search(r"```\s*(.*?)\s*```", sql_text, re.DOTALL)
        if match:
            sql_text = match.group(1).strip()
        else:
            sql_text = re.sub(r"^```sql\s*", "", sql_text, flags=re.IGNORECASE)
            sql_text = re.sub(r"^```\s*", "", sql_text)
            sql_text = re.sub(r"\s*```\s*$", "", sql_text)

    sql_text = sql_text.replace("`", "")
    sql_text = " ".join(sql_text.split())

    for prefix in (
        "here's the sql query:",
        "here is the sql query:",
        "the sql query is:",
        "sql query:",
        "query:",
        "sql:",
    ):
        if sql_text.lower().startswith(prefix):
            sql_text = sql_text[len(prefix):].strip()
            break

    sql_text = sql_text.rstrip(".").replace(";", "")
    return sql_text.lower()


def _load_model(model_name: str, adapter_path: Optional[str] = None, **load_kwargs):
    print(f"Loading model: {model_name}", file=sys.stderr)
    if adapter_path:
        print(f"  adapter: {adapter_path}", file=sys.stderr)
        load_kwargs["adapter_path"] = adapter_path
    model, tokenizer = load(model_name, **load_kwargs)
    print("Model loaded.", file=sys.stderr)
    return model, tokenizer


def infer(
    model: str,
    prompts: List[str],
    batch_size: int = 1,
    max_tokens: int = 512,
    adapter_path: Optional[str] = None,
) -> List[str]:
    """
    Run batch inference with a single model and return cleaned SQL strings.

    Args:
        model:        Model spec: short key from models.json or full HuggingFace path,
                      optionally suffixed with ':fine-tuned'.
        prompts:      Natural-language prompts to send to the model.
        batch_size:   Prompts per batch (default 1 = sequential).
        max_tokens:   Max tokens to generate per prompt.
        adapter_path: Optional explicit path to a LoRA adapter directory.

    Returns:
        List of post-processed SQL strings, one per prompt.
    """
    if not model:
        raise ValueError("A model spec is required")
    if not prompts:
        return []

    # Resolve short key, preserving any ':fine-tuned' suffix
    raw_spec = model
    key, _, suffix = raw_spec.partition(":")
    entry = _resolve_model_entry(key)
    resolved = entry["path"]
    load_kwargs = {k: v for k, v in entry.items() if k != "path"}
    resolved_spec = f"{resolved}:{suffix}" if suffix else resolved

    model_name, use_adapter = parse_model_spec(resolved_spec)

    if use_adapter and not adapter_path:
        raise ValueError(
            "Model spec includes ':fine-tuned' but no adapter_path was provided. "
            "Pass adapter_path= explicitly."
        )

    model, tokenizer = _load_model(model_name, adapter_path if use_adapter else None, **load_kwargs)

    results: List[str] = []
    total = len(prompts)

    for start in range(0, total, batch_size):
        chunk = prompts[start : start + batch_size]
        end = start + len(chunk)
        print(f"Inferring prompts {start + 1}–{end}/{total}...", file=sys.stderr)

        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in chunk
        ]

        batch_result = batch_generate(
            model, tokenizer, formatted, verbose=False, max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        )

        for text in batch_result.texts:
            results.append(post_process_sql(normalize_response(text)))

    return results


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def _config_to_template(config: dict) -> str:
    """Assemble a Jinja2 template string from the visible sections of a prompt config."""
    parts = [section["text"] for section in config.values()
             if section.get("visible", True) and section.get("text")]
    return "\n".join(parts)


def _parse_json_list(value: str) -> List[str]:
    """Parse a JSON array string into a list; return [] on failure."""
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _render_simplified_ddl(raw: str) -> str:
    return "\n".join(_parse_json_list(raw))


def _render_foreign_keys(raw: str) -> str:
    return "\n".join(_parse_json_list(raw))


def _render_cell_values(db_id: str, spider_db_dir: str, max_samples: int = 3) -> str:
    db_path = os.path.join(spider_db_dir, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return ""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        formatted = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            cursor.execute(f"SELECT * FROM [{table}] LIMIT {max_samples}")
            rows = cursor.fetchall()
            col_samples = list(zip(*rows)) if rows else [[] for _ in columns]
            col_strs = [
                f"{col}[{', '.join(str(v) for v in vals[:max_samples])}]"
                for col, vals in zip(columns, col_samples)
            ]
            formatted.append(f"{table}(" + ", ".join(col_strs) + ")")
        return "\n".join(formatted)
    finally:
        conn.close()


def _format_few_shot(examples: List[Dict]) -> str:
    lines = []
    for ex in examples:
        lines.append(ex["question"])
        lines.append(ex["sql"])
    return "\n".join(lines)


def _get_line_prefix(template_text: str, var_name: str) -> str:
    """
    Return the comment-style prefix before {{var_name}} on its line (e.g. '# '), or ''.
    Only recognised when the prefix is purely whitespace/comment characters so that
    variables embedded mid-sentence (like in 00_Baseline.j2) don't get the whole
    preceding sentence repeated on continuation lines.
    """
    match = re.search(
        r'^(.*?)\{\{\s*' + re.escape(var_name) + r'\s*[\|}\s]',
        template_text,
        re.MULTILINE,
    )
    if not match:
        return ""
    prefix = match.group(1)
    # Accept only whitespace / comment-marker characters as a repeatable prefix
    return prefix if re.fullmatch(r'[\s#\-\*]*', prefix) else ""


def _apply_prefix(value: str, prefix: str) -> str:
    """Prepend prefix to every line after the first (the first line gets it from the template)."""
    if not prefix:
        return value
    return value.replace("\n", "\n" + prefix)


def _template_vars(template_text: str) -> set:
    env = Environment()
    ast = env.parse(template_text)
    return jinja_meta.find_undeclared_variables(ast)


def prompt_generation(
    config: Dict[str, Any],
    db_path: str,
    source: Optional[str] = None,
    difficulty: Optional[Union[str, List[str]]] = None,
    limit: Optional[int] = None,
    top_k_few_shot: int = 3,
) -> List[Dict[str, Any]]:
    """
    Render prompts from a prompt config dict for rows in gold_dataset.

    The config is a dict whose values each have "text" (Jinja2 template string)
    and "visible" (bool). Only visible sections with non-empty text are included.

    Only fetches the data each template variable actually needs:
      - question, simplified_ddl, foreign_keys — from gold_dataset
      - cell_values  — sampled from the Spider SQLite database for that db_id
      - few_shot     — retrieved via vector similarity from embedding_dataset

    Args:
        config:          Parsed prompt config dict (from a data/prompt/*.json file).
        db_path:         Path to OpenText2SQL.db.
        source:          Filter by 'train', 'dev', or 'test'. None = all.
        difficulty:      Filter by one or more of 'easy', 'medium', 'hard', 'extra'. None = all.
        limit:           Cap the number of rows returned.
        top_k_few_shot:  Number of few-shot examples to retrieve (only when config uses {{few_shot}}).

    Returns:
        List of dicts, one per row:
          { prompt, question, db_id, source, difficulty, query, simplified_ddl,
            foreign_keys, cell_values, few_shot }
    """
    from dotenv import load_dotenv
    load_dotenv()
    root_path = os.environ.get("ROOT_PATH")
    if not root_path:
        raise ValueError("ROOT_PATH not set in .env")

    spider_db_dir = os.path.join(root_path, "database", "spider")
    template = _config_to_template(config)
    needs = _template_vars(template)

    # Build SQL query with optional filters
    clauses, params = [], []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if difficulty:
        difficulties = [difficulty] if isinstance(difficulty, str) else list(difficulty)
        clauses.append(f"difficulty IN ({','.join('?' * len(difficulties))})")
        params.extend(difficulties)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    sql = f"SELECT id, db_id, source, difficulty, question, query, simplified_ddl, foreign_keys FROM gold_dataset {where} {limit_clause}"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    env = Environment()
    tmpl = env.from_string(template)

    # Detect the prefix character(s) before each variable in the template (e.g. "# " or "").
    # This lets multi-line values line up correctly regardless of template style.
    prefixes = {var: _get_line_prefix(template, var) for var in needs}

    records = []
    for id_, db_id, src, diff, question, query, simplified_ddl_raw, foreign_keys_raw in rows:
        render_params: Dict[str, Any] = {}

        if "question" in needs:
            render_params["question"] = question
        if "simplified_ddl" in needs:
            render_params["simplified_ddl"] = _apply_prefix(
                _render_simplified_ddl(simplified_ddl_raw), prefixes["simplified_ddl"]
            )
        if "foreign_keys" in needs:
            render_params["foreign_keys"] = _apply_prefix(
                _render_foreign_keys(foreign_keys_raw), prefixes["foreign_keys"]
            )
        cell_values_raw = _render_cell_values(db_id, spider_db_dir)
        if "cell_values" in needs:
            render_params["cell_values"] = _apply_prefix(cell_values_raw, prefixes["cell_values"])
        few_shot_examples: List[Dict] = []
        if "few_shot" in needs:
            from scripts.util.nlp import get_few_shot
            few_shot_examples = get_few_shot(question, simplified_ddl_raw, db_path, db_path, top_k=top_k_few_shot)
            render_params["few_shot"] = _apply_prefix(
                _format_few_shot(few_shot_examples), prefixes["few_shot"]
            )

        records.append({
            "prompt": tmpl.render(render_params),
            "question": question,
            "db_id": db_id,
            "source": src,
            "difficulty": diff,
            "query": query,
            "simplified_ddl": simplified_ddl_raw,
            "foreign_keys": foreign_keys_raw,
            "cell_values": cell_values_raw,
            "few_shot": few_shot_examples,
        })

    return records


def render_prompt(
    config: Dict[str, Any],
    rec: Dict[str, Any],
    section_visibility: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Render a single prompt from a config dict using data already present in a record.

    No database queries — reads simplified_ddl, foreign_keys, cell_values, few_shot,
    and question directly from rec.

    Args:
        config:             Parsed prompt config dict.
        rec:                Record dict (e.g. from prompt_generation or schema_linking).
        section_visibility: Per-section visibility overrides (e.g. from schema_linking).
                            Keys match config section names; False disables that section.
                            Takes precedence over the section's own 'visible' flag.

    Returns:
        Rendered prompt string.
    """
    # Build effective config honoring base-config visibility (ablation settings).
    # section_visibility (from schema linking) is handled via placeholder injection
    # below — disabled sections stay in the template so the prompt's structural
    # shape is preserved between preSQL and finSQL calls.
    effective_config = {
        key: {**section}
        for key, section in config.items()
        if section.get("visible", True)
    }

    # Collect template variables that belong to schema-linking-disabled sections.
    # Those vars will receive a "None." placeholder instead of being dropped.
    placeholder_vars: set = set()
    if section_visibility:
        for key, visible in section_visibility.items():
            if not visible and key in effective_config:
                placeholder_vars.update(_template_vars(effective_config[key].get("text", "")))

    template = _config_to_template(effective_config)
    if not template.strip():
        return ""

    needs = _template_vars(template)
    prefixes = {var: _get_line_prefix(template, var) for var in needs}

    render_params: Dict[str, Any] = {}
    if "question" in needs:
        render_params["question"] = rec.get("question") or ""
    if "simplified_ddl" in needs:
        if "simplified_ddl" in placeholder_vars:
            render_params["simplified_ddl"] = "None."
        else:
            render_params["simplified_ddl"] = _apply_prefix(
                _render_simplified_ddl(rec.get("simplified_ddl") or "[]"),
                prefixes.get("simplified_ddl", ""),
            )
    if "foreign_keys" in needs:
        if "foreign_keys" in placeholder_vars:
            render_params["foreign_keys"] = "None."
        else:
            render_params["foreign_keys"] = _apply_prefix(
                _render_foreign_keys(rec.get("foreign_keys") or "[]"),
                prefixes.get("foreign_keys", ""),
            )
    if "cell_values" in needs:
        if "cell_values" in placeholder_vars:
            render_params["cell_values"] = "None."
        else:
            render_params["cell_values"] = _apply_prefix(
                rec.get("cell_values") or "",
                prefixes.get("cell_values", ""),
            )
    if "few_shot" in needs:
        render_params["few_shot"] = _apply_prefix(
            _format_few_shot(rec.get("few_shot") or []),
            prefixes.get("few_shot", ""),
        )

    return Environment().from_string(template).render(render_params)


# ---------------------------------------------------------------------------
# Schema linking
# ---------------------------------------------------------------------------

_FK_TABLE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.UNICODE)


def _linked_simplified_ddl(ddl_list: List[str], referenced: set) -> List[str]:
    kept = [item for item in ddl_list
            if item.split("(", 1)[0].strip().lower() in referenced]
    return kept or ddl_list  # fallback: keep all if nothing matched


def _linked_foreign_keys(fk_list: List[str], referenced: set) -> List[str]:
    kept = []
    for item in fk_list:
        tables = {m.lower() for m in _FK_TABLE_RE.findall(item)}
        if tables and tables.issubset(referenced):
            kept.append(item)
    return kept


def _linked_cell_values(cell_values: str, referenced: set) -> str:
    lines = [ln for ln in cell_values.splitlines() if ln.strip()]
    kept = [ln for ln in lines
            if ln.split("(", 1)[0].strip().lower() in referenced]
    return "\n".join(kept) if kept else cell_values  # fallback: keep all if nothing matched


def schema_linking(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prune simplified_ddl, foreign_keys and cell_values in each record to only the
    tables/columns referenced in the presql field.

    Args:
        records: List of dicts from presql.jsonl. Each dict must contain at least:
                 presql, simplified_ddl (JSON string), foreign_keys (JSON string),
                 cell_values (rendered text).

    Returns:
        List of dicts (same length and same keys as input, minus 'prompt') where
        simplified_ddl, foreign_keys and cell_values contain only the entries
        relevant to the presql output.
        simplified_ddl and foreign_keys are returned as JSON strings to preserve
        the original format; cell_values is returned as rendered text.

        Each record also gains a 'section_visibility' key — a dict mapping config
        section names to booleans. False means the section is empty after pruning
        and should be disabled in the finSQL prompt config:
          {
            "schema":            bool,   # True unless simplified_ddl pruned to empty
            "foreign_keys":      bool,   # True unless no FKs reference the linked tables
            "reference_values":  bool,   # True unless cell_values pruned to empty
          }
    """
    from scripts.util.nlp import extract_referenced_tables_from_sql

    results = []
    for rec in records:
        presql = rec.get("presql") or ""
        tables, _ = extract_referenced_tables_from_sql(presql)
        referenced = {t.lower() for t in tables}

        ddl_list = _parse_json_list(rec.get("simplified_ddl") or "[]")
        fk_list  = _parse_json_list(rec.get("foreign_keys")  or "[]")
        cell_raw = rec.get("cell_values") or ""

        if referenced:
            ddl_list = _linked_simplified_ddl(ddl_list, referenced)
            fk_list  = _linked_foreign_keys(fk_list,  referenced)
            cell_raw = _linked_cell_values(cell_raw,  referenced)

        out = {k: v for k, v in rec.items() if k != "prompt"}
        out["simplified_ddl"] = json.dumps(ddl_list, ensure_ascii=False)
        out["foreign_keys"]   = json.dumps(fk_list,  ensure_ascii=False)
        out["cell_values"]    = cell_raw
        out["section_visibility"] = {
            "schema":           bool(ddl_list),
            "foreign_keys":     bool(fk_list),
            "reference_values": bool(cell_raw.strip()),
        }
        results.append(out)

    return results


# ---------------------------------------------------------------------------
# Cross-consistency
# ---------------------------------------------------------------------------

def _execute_sql(sql: str, db_path: str):
    """Execute SQL against a SQLite DB. Returns a sorted DataFrame, empty DataFrame, or None on error."""
    import sqlite3
    import pandas as pd

    if not sql or not sql.strip():
        return None
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        if not columns:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=columns)
        return df.sort_values(by=list(df.columns)).reset_index(drop=True)
    except Exception:
        return None


def _results_equal(r1, r2) -> bool:
    """True if two SQL execution results are semantically equivalent."""
    import pandas as pd

    if r1 is None and r2 is None:
        return True
    if r1 is None or r2 is None:
        return False
    if isinstance(r1, pd.DataFrame) and isinstance(r2, pd.DataFrame):
        if r1.empty and r2.empty:
            return True
        if r1.empty or r2.empty:
            return False
        try:
            return r1.equals(r2)
        except Exception:
            return False
    return False


def cross_consistency(
    models: List[str],
    records: List[Dict[str, Any]],
    batch_size: int = 1,
    max_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    Run inference with multiple models sequentially and select the SQL whose execution
    result appears most often (cross-consistency via semantic majority vote).

    For each model, infer() is called over all records. After all models finish, every
    candidate SQL is executed against its Spider database and grouped by result
    equivalence. The SQL from the largest equivalence group wins; ties are broken
    randomly.

    Args:
        models:     List of model specs — short keys from models.json or full HuggingFace
                    paths. Fine-tuned models (':fine-tuned' suffix) are not supported.
        records:    List of dicts from prompt_generation(). Each must have 'prompt' and
                    'db_id' keys.
        batch_size: Prompts per inference batch passed to infer() (default 1).
        max_tokens: Max tokens to generate per prompt.

    Returns:
        List of dicts (same length and keys as records) with added keys:
          sql               — winning SQL string (representative from majority group)
          all_sql           — list of all candidate SQLs, one per model, in model order
          consistency_score — fraction of models whose SQL produced the winning result
          models            — list of resolved model names used
    """
    import random
    from dotenv import load_dotenv
    load_dotenv()
    root_path = os.environ.get("ROOT_PATH")
    if not root_path:
        raise ValueError("ROOT_PATH not set in .env")

    spider_db_dir = os.path.join(root_path, "database", "spider")
    prompts = [r["prompt"] for r in records]

    # ── Step 1: run each model sequentially, keep all outputs in memory ───────
    # all_model_sql[model_idx][record_idx] = sql_string
    all_model_sql: List[List[str]] = []
    resolved_names: List[str] = []

    for model_spec in models:
        key = model_spec.partition(":")[0]
        resolved_names.append(resolve_model(key))
        sql_list = infer(model=model_spec, prompts=prompts, batch_size=batch_size, max_tokens=max_tokens)
        all_model_sql.append(sql_list)

    # ── Step 2: reorganize by record ─────────────────────────────────────────
    # candidates[record_idx] = [sql_model_0, sql_model_1, ...]
    candidates_per_record = [
        [all_model_sql[m][r] for m in range(len(models))]
        for r in range(len(records))
    ]

    # ── Step 3: semantic majority vote per record (skipped for single model) ──
    single_model = len(models) == 1
    results = []
    for rec, candidates in zip(records, candidates_per_record):
        out = dict(rec)
        out["all_sql"] = candidates
        out["models"] = resolved_names

        if single_model:
            out["sql"] = candidates[0]
            out["consistency_score"] = None
        else:
            db_id = rec.get("db_id") or ""
            db_path = os.path.join(spider_db_dir, db_id, f"{db_id}.sqlite")

            exec_results = [_execute_sql(sql, db_path) for sql in candidates]

            groups: List[List[int]] = []
            for i, res in enumerate(exec_results):
                placed = False
                for group in groups:
                    if _results_equal(res, exec_results[group[0]]):
                        group.append(i)
                        placed = True
                        break
                if not placed:
                    groups.append([i])

            max_size = max(len(g) for g in groups)
            winner_group = random.choice([g for g in groups if len(g) == max_size])
            out["sql"] = candidates[winner_group[0]]
            out["consistency_score"] = max_size / len(candidates)

        results.append(out)

    return results
