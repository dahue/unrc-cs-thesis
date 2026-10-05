DROP TABLE IF EXISTS embedding_dataset;
CREATE VIRTUAL TABLE embedding_dataset USING vec0(
    vector float[300] distance_metric=cosine,
    +id                INTEGER,
    +db_id             TEXT,
    +source            TEXT,
    +question          TEXT,
    +skeleton_question TEXT
);
