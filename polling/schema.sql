/*
  Bluebikes polling database (SQLite)

  Conventions
  - All times are UTC epoch seconds (INTEGER). We never store local time.
  - station_id is always TEXT. The feed mixes UUIDs and long numbers.
  - Booleans are INTEGER 0 or 1.
  - Every table that mirrors a feed object has an extra_json or item_json column.
    It holds the full original object, so no value is lost if the feed adds or
    changes a field.
  - Run with: sqlite3 bluebikes.db < polling/schema.sql
  - foreign_keys is a per connection setting. The poller must run
    PRAGMA foreign_keys = ON; every time it connects.
*/

PRAGMA journal_mode = WAL;
PRAGMA user_version = 1;
PRAGMA foreign_keys = ON;


/* ------------------------------------------------------------------
   Fetch log: one row per feed per poll attempt.
   This is how we tell "nothing happened" from "the poller was down".
   It replaces the old poll_log idea and covers all 12 feeds.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS fetch_log (
    fetch_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_name          TEXT    NOT NULL,   -- key from feeds.json, e.g. station_status
    url                TEXT,
    fetched_at         INTEGER NOT NULL,   -- our clock
    feed_last_updated  INTEGER,            -- top level last_updated from the feed
    ttl                INTEGER,            -- top level ttl from the feed
    version            TEXT,               -- top level version from the feed
    ok                 INTEGER NOT NULL CHECK (ok IN (0, 1)),
    http_status        INTEGER,
    error              TEXT,
    record_count       INTEGER,            -- stations, bikes, alerts, and so on
    payload_sha256     TEXT                -- links to raw_payloads when stored
                                           -- Hash only the data object. last_updated
                                           -- changes on every fetch, even when the
                                           -- content is the same.
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_feed_time
    ON fetch_log (feed_name, fetched_at);


/* ------------------------------------------------------------------
   Raw payloads: full JSON text, stored once per distinct content.
   Policy for the poller:
   - Store every changed payload for all feeds except station_status.
   - For station_status, the typed table below already holds every value,
     and the gzipped raw files on disk keep the original responses.
   Because we dedupe by hash, this table is also the change history for
   station_information (capacity, names, and so on).
   Compute the hash over the data object only, as canonical JSON (sorted keys).
   The top level last_updated and ttl change every minute on all 12 feeds,
   so hashing the whole response would make every fetch look new.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS raw_payloads (
    payload_sha256     TEXT    PRIMARY KEY,
    feed_name          TEXT    NOT NULL,
    first_fetch_id     INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    first_seen         INTEGER NOT NULL,
    payload            TEXT    NOT NULL
);


/* ------------------------------------------------------------------
   gbfs and gbfs_versions: which feeds and versions exist
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS gbfs_feeds (
    language           TEXT    NOT NULL,   -- usually en
    name               TEXT    NOT NULL,
    url                TEXT    NOT NULL,
    first_seen         INTEGER NOT NULL,
    last_seen          INTEGER NOT NULL,
    PRIMARY KEY (language, name)
);

CREATE TABLE IF NOT EXISTS gbfs_versions (
    version            TEXT    PRIMARY KEY,
    url                TEXT,
    first_seen         INTEGER NOT NULL,
    last_seen          INTEGER NOT NULL
);


/* ------------------------------------------------------------------
   stations: current state of station_information, one row per station.
   Refresh it now and then. first_seen and last_seen show when a station
   appeared and disappeared. Old versions live in raw_payloads.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS stations (
    station_id                       TEXT    PRIMARY KEY,
    name                             TEXT,
    short_name                       TEXT,
    lat                              REAL,
    lon                              REAL,
    capacity                         INTEGER,
    region_id                        TEXT,      -- nullable, some towns have none
    legacy_id                        TEXT,
    external_id                      TEXT,
    station_type                     TEXT,
    rental_uris_json                 TEXT,
    rental_methods_json              TEXT,
    electric_bike_surcharge_waiver   INTEGER,
    eightd_has_key_dispenser         INTEGER,
    has_kiosk                        INTEGER,
    eightd_station_services_json     TEXT,
    extra_json                       TEXT,      -- any field not listed above
    first_seen                       INTEGER NOT NULL,
    last_seen                        INTEGER NOT NULL
);


/* ------------------------------------------------------------------
   station_status: one row per station per poll. This is the main table.
   No foreign key to stations on purpose. A station can show up here
   before it shows up in station_information.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS station_status (
    station_id                  TEXT    NOT NULL,
    fetched_at                  INTEGER NOT NULL,  -- same value for the whole poll
    fetch_id                    INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    last_reported               INTEGER,           -- NULL when the feed sends 86400
    last_reported_raw           INTEGER,           -- exactly what the feed sent
    num_bikes_available         INTEGER,
    num_ebikes_available        INTEGER,
    num_bikes_disabled          INTEGER,
    num_docks_available         INTEGER,
    num_docks_disabled          INTEGER,
    is_installed                INTEGER,
    is_renting                  INTEGER,
    is_returning                INTEGER,
    legacy_id                   TEXT,
    eightd_has_available_keys   INTEGER,
    num_scooters_available      INTEGER,           -- only on some stations
    num_scooters_unavailable    INTEGER,           -- only on some stations
    extra_json                  TEXT,
    PRIMARY KEY (station_id, fetched_at)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_station_status_time
    ON station_status (fetched_at);

/* Empty and full, using the rule we agreed on.
   An offline or not renting station does not count as empty or full. */
CREATE VIEW IF NOT EXISTS v_station_status AS
SELECT
    s.*,
    CASE WHEN s.num_bikes_available = 0
              AND s.is_installed = 1 AND s.is_renting = 1
         THEN 1 ELSE 0 END AS is_empty,
    CASE WHEN s.num_docks_available = 0
              AND s.is_installed = 1 AND s.is_returning = 1
         THEN 1 ELSE 0 END AS is_full
FROM station_status AS s;


/* ------------------------------------------------------------------
   ebikes_at_stations: one row per e-bike listed at a station.
   free_bike_status: one row per bike outside a dock.
   We have not seen real data for these two feeds, so the whole object
   is kept in item_json. Bluebikes is docked, so free_bike_status may
   be empty most of the time.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS ebikes_at_stations (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    station_id         TEXT    NOT NULL,
    item_index         INTEGER NOT NULL,       -- position inside the station list
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, station_id, item_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS free_bike_status (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    item_index         INTEGER NOT NULL,
    bike_id            TEXT,
    lat                REAL,
    lon                REAL,
    is_reserved        INTEGER,
    is_disabled        INTEGER,
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, item_index)
) WITHOUT ROWID;


/* ------------------------------------------------------------------
   System feeds. These change rarely, so the poller inserts rows only
   when the payload hash is new. fetch_id says which fetch the rows came
   from. "Current" means the rows with the highest fetch_id per table.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS system_information (
    fetch_id           INTEGER PRIMARY KEY REFERENCES fetch_log (fetch_id),
    system_id          TEXT,
    language           TEXT,
    name               TEXT,
    short_name         TEXT,
    operator           TEXT,
    url                TEXT,
    purchase_url       TEXT,
    start_date         TEXT,
    phone_number       TEXT,
    email              TEXT,
    timezone           TEXT,
    license_url        TEXT,
    item_json          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS system_regions (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    region_id          TEXT    NOT NULL,
    name               TEXT,
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, region_id)
) WITHOUT ROWID;

/* In the real feed, price is a string like "3.00" and is_taxable is a JSON
   boolean. SQLite turns the string into a number for the REAL column, and the
   poller must write is_taxable as 0 or 1. */
CREATE TABLE IF NOT EXISTS system_pricing_plans (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    plan_id            TEXT    NOT NULL,
    url                TEXT,
    name               TEXT,
    currency           TEXT,
    price              REAL,
    is_taxable         INTEGER,
    description        TEXT,
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, plan_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS system_alerts (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    item_index         INTEGER NOT NULL,
    alert_id           TEXT,
    type               TEXT,
    times_json         TEXT,
    station_ids_json   TEXT,
    region_ids_json    TEXT,
    url                TEXT,
    summary            TEXT,
    description        TEXT,
    last_updated       INTEGER,
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, item_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS system_hours (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    item_index         INTEGER NOT NULL,
    user_types_json    TEXT,
    days_json          TEXT,
    start_time         TEXT,
    end_time           TEXT,
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, item_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS system_calendar (
    fetch_id           INTEGER NOT NULL REFERENCES fetch_log (fetch_id),
    item_index         INTEGER NOT NULL,
    start_month        INTEGER,
    start_day          INTEGER,
    start_year         INTEGER,
    end_month          INTEGER,
    end_day            INTEGER,
    end_year           INTEGER,
    item_json          TEXT    NOT NULL,
    PRIMARY KEY (fetch_id, item_index)
) WITHOUT ROWID;


/* ------------------------------------------------------------------
   Neighborhood tags. Kept apart from stations, so we can change the
   boundaries without touching raw data. A NULL neighborhood means the
   station is outside every Boston polygon.
   ------------------------------------------------------------------ */
CREATE TABLE IF NOT EXISTS station_neighborhoods (
    station_id         TEXT    PRIMARY KEY,
    neighborhood       TEXT,
    boundary_file      TEXT,                   -- e.g. boston_neighborhoods.geojson
    tagged_at          INTEGER NOT NULL
);
