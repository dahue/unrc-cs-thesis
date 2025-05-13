DROP TABLE IF EXISTS spider_dataset;
DROP TABLE IF EXISTS spider_tables;
DROP TABLE IF EXISTS spider_natsql;

CREATE TABLE spider_dataset (
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

CREATE TABLE spider_natsql (
    id INTEGER NOT NULL,
    source TEXT NOT NULL,
    natsql TEXT NOT NULL,
    PRIMARY KEY (id, source)
);