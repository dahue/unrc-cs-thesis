"""
nlp.py — NLP, few-shot retrieval, and schema refinement utilities.

Public API:
  get_question_skeleton(question, schema)                               -> str
  get_few_shot(question, schema, index_db_path, gold_db_path, top_k=3) -> list
  extract_referenced_tables_from_sql(sql)                              -> (set, error)
"""

import re
import sys
import json
import sqlite3
import subprocess
from typing import Optional, Set, Tuple

import nltk
import spacy
import sqlite_vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK data is expected to already be present (downloaded by src/pipeline/embedding.py).

_nlp = None
_TABLE = "embedding_dataset"


def _get_nlp():
    global _nlp
    if _nlp is None:
        spacy.prefer_gpu()
        name = "en_core_web_md"
        try:
            _nlp = spacy.load(name)
        except OSError:
            print(f"Downloading {name}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", name])
            _nlp = spacy.load(name)
    return _nlp


def _open_vec_conn(path, read_only=False):
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


# ---------------------------------------------------------------------------
# Skeleton computation
# ---------------------------------------------------------------------------

def get_question_skeleton(question, schema):
    """Replace domain-specific tokens (table/column names, numbers, literals) with <mask>."""
    lemmatizer = WordNetLemmatizer()

    try:
        schema_data = json.loads(schema)
    except (json.JSONDecodeError, TypeError):
        return question

    domain_tokens = set()
    for table_info in schema_data:
        table_name = table_info.split('(')[0].strip()
        domain_tokens.add(table_name.lower())
        columns_part = table_info.split('(')[1].split(')')[0]
        for col in columns_part.split(','):
            domain_tokens.add(col.strip().split()[0].lower())

    quoted_strings = []
    quote_pattern = r"'([^']*)'|\"([^\"]*)\""

    def replace_quoted(match):
        quoted_strings.append(match.group(0))
        return f"__QUOTED_STRING_{len(quoted_strings) - 1}__"

    question_with_placeholders = re.sub(quote_pattern, replace_quoted, question)
    tokens = word_tokenize(question_with_placeholders.lower())

    skeleton_tokens = []
    for token in tokens:
        if token.startswith("__quoted_string_") and token.endswith("__"):
            skeleton_tokens.append('<mask>')
        elif token.isdigit() or token.replace('.', '').replace(',', '').isdigit():
            skeleton_tokens.append('<mask>')
        else:
            lemmatized = lemmatizer.lemmatize(token)
            if lemmatized in domain_tokens or token in domain_tokens:
                skeleton_tokens.append('<mask>')
            else:
                skeleton_tokens.append(token)

    return ' '.join(skeleton_tokens)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def get_few_shot(question, schema, index_db_path, gold_db_path, top_k=3):
    """
    Return the top_k most similar training examples as a list of dicts:
      [{"question": str, "sql": str, "distance": float}, ...]
    """
    nlp = _get_nlp()
    skeleton = get_question_skeleton(question, schema)
    query_vector = nlp(skeleton).vector.astype('float32').tobytes()

    conn = _open_vec_conn(index_db_path, read_only=True)
    try:
        hits = conn.execute(
            f"SELECT id, source, distance FROM {_TABLE} WHERE vector MATCH ? ORDER BY distance LIMIT ?",
            (query_vector, top_k),
        ).fetchall()
    finally:
        conn.close()

    if not hits:
        return []

    gold_conn = sqlite3.connect(f"file:{gold_db_path}?mode=ro", uri=True)
    try:
        results = []
        for id_, source, distance in hits:
            row = gold_conn.execute(
                "SELECT question, query FROM gold_dataset WHERE id=? AND source=?",
                (id_, source),
            ).fetchone()
            if row:
                results.append({"question": row[0], "sql": row[1], "distance": distance})
        return results
    finally:
        gold_conn.close()


# ---------------------------------------------------------------------------
# Schema refinement
# ---------------------------------------------------------------------------

def _normalize_identifier(name: str) -> str:
    return (name or "").strip().strip('"').strip("'").lower()


def extract_referenced_tables_from_sql(sql: str) -> Tuple[Set[str], Optional[str]]:
    """Extract referenced base tables from a SQL query using sqlglot."""
    sql = (sql or "").strip()
    if not sql:
        return set(), "empty_sql"

    try:
        from sqlglot import parse_one, exp

        tree = parse_one(sql, read="sqlite")

        cte_names: Set[str] = set()
        for cte in tree.find_all(exp.CTE):
            alias = cte.alias
            if alias:
                cte_names.add(_normalize_identifier(alias))

        tables: Set[str] = set()
        for t in tree.find_all(exp.Table):
            name = _normalize_identifier(t.name)
            if not name or name in cte_names:
                continue
            tables.add(name)

        return tables, None
    except Exception as e:
        return set(), str(e)
