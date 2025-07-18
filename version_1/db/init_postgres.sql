--Re-run the SQL script generation after kernel -- DreamSeeker: Initialize PostgreSQL Table from CSV
-- --------------------------------------------------
-- This script creates a table named `bucket_list_activities`
-- and imports data from your CSV file located at data/raw/bucket_list_extended.csv.
-- Adjust the absolute path to match your local environment before executing.

-- DreamSeeker: Initialize PostgreSQL Table from CSV

BEGIN;

CREATE TABLE IF NOT EXISTS bucket_list_activities (
    user_id INTEGER,
    activity_id INTEGER,
    activity_name VARCHAR(100),
    category VARCHAR(50),
    activity_type VARCHAR(50),
    user_mood VARCHAR(50),
    user_interest_score FLOAT,
    label INTEGER
);

COPY bucket_list_activities
FROM '/Users/glizkmoe.fit/Documents/Pycharm/DreamSeeker/data/raw/bucket_list_extended.csv'
DELIMITER ','
CSV HEADER;

COMMIT;

