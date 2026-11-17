/*
  Migration 001 -> 002

  What changes
  - raw_payloads is keyed on (feed_name, payload_sha256) instead of the hash
    alone, so two feeds that produce the same data object both get archived.
  - New index on fetch_log (feed_name, payload_sha256) for the poller's
    on_change lookup.

  Run it on a stopped poller:
      sqlite3 bluebikes.db < polling/migrations/002_raw_payloads_key.sql

  Safe to run on a v1 database only. On anything else the guard below fails
  the script before a single row is touched. The old key was stricter than
  the new one, so no row can collide on copy.
*/

-- Without this the CLI runs on past a failed statement. A dot command takes
-- no trailing comment, so this one sits on its own line.
.bail on

PRAGMA foreign_keys = OFF;   -- the table rebuild below would trip the fetch_log FK

BEGIN;

/* Stop unless this is a v1 database. The CHECK fails when user_version is
   anything else, and .bail on turns that into an exit before BEGIN commits. */
CREATE TEMP TABLE expected_user_version_1 (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO expected_user_version_1 (ok)
SELECT CASE WHEN (SELECT * FROM pragma_user_version) = 1 THEN 1 ELSE 0 END;

CREATE TABLE raw_payloads_new (
    feed_name          TEXT    NOT NULL,
    payload_sha256     TEXT    NOT NULL,
    first_fetch_id     INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    first_seen         INTEGER NOT NULL,
    payload            TEXT    NOT NULL,
    PRIMARY KEY (feed_name, payload_sha256)
);

INSERT INTO raw_payloads_new (feed_name, payload_sha256, first_fetch_id, first_seen, payload)
SELECT feed_name, payload_sha256, first_fetch_id, first_seen, payload FROM raw_payloads;

DROP TABLE raw_payloads;
ALTER TABLE raw_payloads_new RENAME TO raw_payloads;

CREATE INDEX IF NOT EXISTS idx_fetch_log_feed_payload
    ON fetch_log (feed_name, payload_sha256);

PRAGMA user_version = 2;

COMMIT;

PRAGMA foreign_key_check;    -- prints nothing when the rebuild kept every reference
PRAGMA foreign_keys = ON;
