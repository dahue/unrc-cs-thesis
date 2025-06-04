DROP TABLE IF EXISTS gold_dataset;

CREATE TABLE gold_dataset (
    id INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    query TEXT NOT NULL,
    is_valid BOOLEAN DEFAULT 1,
    notes TEXT,
    simplified_ddl TEXT,
    full_ddl TEXT,
    foreign_keys TEXT,
    difficulty TEXT,
    natsql TEXT,
    PRIMARY KEY (id, source)
);