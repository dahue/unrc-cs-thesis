DROP TABLE IF EXISTS bronze_dataset;
DROP TABLE IF EXISTS spider_tables;
DROP TABLE IF EXISTS silver_dataset;
DROP TABLE IF EXISTS gold_dataset;
DROP TABLE IF EXISTS finetune_dataset;

CREATE TABLE bronze_dataset (
    id INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    question_toks TEXT,
    query TEXT NOT NULL,
    query_toks TEXT,
    query_toks_no_value TEXT,
    sql_json TEXT,
    PRIMARY KEY (id, source)
);

CREATE TABLE spider_tables (
    db_id TEXT,
    source TEXT,
    table_names TEXT,
    table_names_original TEXT,
    column_names TEXT,
    column_names_original TEXT,
    column_types TEXT,
    primary_keys TEXT,
    foreign_keys TEXT
);

CREATE TABLE silver_dataset (
    id INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    query TEXT NOT NULL,
    query_toks_no_value TEXT,
    sql_json TEXT,
    is_valid BOOLEAN NOT NULL DEFAULT 1,
    simplified_ddl TEXT,
    full_ddl TEXT,
    foreign_keys TEXT,
    difficulty TEXT,
    PRIMARY KEY (id, source)
);

CREATE TABLE gold_dataset (
    id INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    query TEXT NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT 1,
    simplified_ddl TEXT,
    full_ddl TEXT,
    foreign_keys TEXT,
    difficulty TEXT,
    PRIMARY KEY (id, source)
);

CREATE TABLE finetune_dataset (
    id INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    source TEXT NOT NULL,
    split TEXT NOT NULL,
    difficulty TEXT,
    optimization_rule TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    PRIMARY KEY (id, source),
    FOREIGN KEY (id, source) REFERENCES gold_dataset (id, source)
);
