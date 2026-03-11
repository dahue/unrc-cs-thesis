import re
from dataclasses import dataclass
from typing import Optional, Set, Tuple


@dataclass(frozen=True)
class SchemaRefinementResult:
    referenced_tables: Set[str]
    refined_simplified_ddl: str
    refined_foreign_keys: str
    parse_error: Optional[str] = None
    used_fallback: bool = False


def _normalize_identifier(name: str) -> str:
    return (name or "").strip().strip('"').strip("'").lower()


def extract_referenced_tables_from_sql(sql: str) -> Tuple[Set[str], Optional[str]]:
    """
    Extract referenced base tables from a SQL query using sqlglot.

    Returns:
        (tables, error) where tables are lowercased.
    """
    sql = (sql or "").strip()
    if not sql:
        return set(), "empty_sql"

    try:
        from sqlglot import parse_one, exp

        tree = parse_one(sql, read="sqlite")

        # Collect CTE names so we can avoid counting them as base tables
        cte_names: Set[str] = set()
        for cte in tree.find_all(exp.CTE):
            alias = getattr(cte, "alias", None)
            if alias and getattr(alias, "this", None):
                cte_names.add(_normalize_identifier(alias.this.name))

        tables: Set[str] = set()
        for t in tree.find_all(exp.Table):
            name = _normalize_identifier(t.name)
            if not name:
                continue
            if name in cte_names:
                continue
            tables.add(name)

        return tables, None
    except Exception as e:
        return set(), str(e)


def refine_simplified_ddl(simplified_ddl: str, referenced_tables: Set[str]) -> str:
    """
    Simplified DDL is stored as lines like:
        table_name(col1, col2, ...)
    We keep only the lines whose table_name appears in referenced_tables.
    """
    simplified_ddl = simplified_ddl or ""
    referenced_tables = {_normalize_identifier(t) for t in (referenced_tables or set()) if t}

    lines = [ln for ln in simplified_ddl.splitlines() if ln.strip()]
    if not lines or not referenced_tables:
        return simplified_ddl.strip()

    kept = []
    for ln in lines:
        table = ln.split("(", 1)[0].strip()
        if _normalize_identifier(table) in referenced_tables:
            kept.append(ln)

    return ("\n".join(kept) if kept else simplified_ddl).strip()


_FK_TABLE_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", re.UNICODE)


def refine_foreign_keys(foreign_keys: str, referenced_tables: Set[str]) -> str:
    """
    Foreign keys appear as free-form lines. We keep FK lines that only mention tables
    within referenced_tables (based on occurrences like `table.column`).
    """
    foreign_keys = foreign_keys or ""
    referenced_tables = {_normalize_identifier(t) for t in (referenced_tables or set()) if t}

    lines = [ln for ln in foreign_keys.splitlines() if ln.strip()]
    if not lines or not referenced_tables:
        return foreign_keys.strip()

    kept = []
    for ln in lines:
        mentioned = {_normalize_identifier(m) for m in _FK_TABLE_PATTERN.findall(ln)}
        if mentioned and mentioned.issubset(referenced_tables):
            kept.append(ln)

    return ("\n".join(kept) if kept else "").strip()


def refine_schema_from_sql(pre_sql: str, simplified_ddl: str, foreign_keys: str) -> SchemaRefinementResult:
    """
    High-level convenience wrapper:
      - parse pre_sql
      - extract referenced tables
      - filter schema + foreign keys
      - if parsing fails or yields no tables, fall back to full schema
    """
    tables, err = extract_referenced_tables_from_sql(pre_sql)
    tables_norm = {_normalize_identifier(t) for t in tables if t}

    if err or not tables_norm:
        return SchemaRefinementResult(
            referenced_tables=tables_norm,
            refined_simplified_ddl=(simplified_ddl or "").strip(),
            refined_foreign_keys=(foreign_keys or "").strip(),
            parse_error=err,
            used_fallback=True,
        )

    refined_ddl = refine_simplified_ddl(simplified_ddl, tables_norm)
    refined_fk = refine_foreign_keys(foreign_keys, tables_norm)

    return SchemaRefinementResult(
        referenced_tables=tables_norm,
        refined_simplified_ddl=refined_ddl,
        refined_foreign_keys=refined_fk,
        parse_error=None,
        used_fallback=False,
    )

