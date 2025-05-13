DROP TABLE IF EXISTS silver_dataset;

CREATE TABLE silver_dataset (
    -- index INTEGER PRIMARY KEY AUTOINCREMENT,
    id INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    query TEXT NOT NULL,
    query_toks_no_value TEXT,
    sql_json TEXT,
    is_valid BOOLEAN DEFAULT 1,
    notes TEXT,
    simplified_ddl TEXT,
    full_ddl TEXT,
    foreign_keys TEXT,
    difficulty TEXT,
    natsql TEXT,
    PRIMARY KEY (id, source)
);