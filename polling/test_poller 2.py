"""Tests for poller.py. Run from the repo root: python3 -m unittest polling.test_poller -v"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

try:
    from polling import poller
except ImportError:          # run from inside polling/
    import poller

SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def feed(data, last_updated=1000):
    return json.dumps({"last_updated": last_updated, "ttl": 60, "version": "1.1", "data": data}).encode()


class PollerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        sqlite3.connect(self.db).executescript(SCHEMA.read_text())
        self.conn = poller.connect(self.db)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def count(self, sql, *args):
        return self.conn.execute(sql, args).fetchone()[0]

    def test_foreign_keys_on(self):
        self.assertEqual(self.count("PRAGMA foreign_keys"), 1)

    def test_station_status(self):
        data = {"stations": [
            {"station_id": "a", "last_reported": 86400, "num_bikes_available": 3,
             "num_docks_available": 5, "is_installed": True, "is_renting": 1,
             "is_returning": 1, "legacy_id": 12, "new_field": "x"},
            {"station_id": 99, "last_reported": 1700000000, "num_bikes_available": 0},
        ]}
        self.assertTrue(poller.store_feed(self.conn, "station_status", "u", feed(data), 2000))
        row = self.conn.execute("SELECT * FROM station_status WHERE station_id='a'").fetchone()
        self.assertIsNone(row[3])            # last_reported NULL for 86400
        self.assertEqual(row[4], 86400)      # raw kept
        self.assertEqual(row[10], 1)         # is_installed True -> 1
        self.assertEqual(json.loads(row[-1]), {"new_field": "x"})
        self.assertEqual(self.count("SELECT COUNT(*) FROM station_status"), 2)
        self.assertEqual(self.count("SELECT COUNT(*) FROM raw_payloads"), 0)   # never stored
        self.assertEqual(self.count("SELECT record_count FROM fetch_log"), 2)

    def test_station_information_upsert_keeps_first_seen(self):
        d = {"stations": [{"station_id": "a", "name": "A", "capacity": 10, "lat": 42.3,
                           "lon": -71.1, "rental_uris": {"ios": "x"}, "extra": 1}]}
        poller.store_feed(self.conn, "station_information", "u", feed(d), 1000)
        d["stations"][0]["capacity"] = 12
        poller.store_feed(self.conn, "station_information", "u", feed(d), 5000)
        row = self.conn.execute("SELECT capacity, first_seen, last_seen, extra_json FROM stations").fetchone()
        self.assertEqual(row, (12, 1000, 5000, '{"extra": 1}'))
        self.assertEqual(self.count("SELECT COUNT(*) FROM raw_payloads"), 2)   # change history

    def test_on_change_feed_skips_repeat(self):
        d = {"plans": [{"plan_id": "p", "price": "3.00", "is_taxable": False, "name": "Single"}]}
        poller.store_feed(self.conn, "system_pricing_plans", "u", feed(d, 1), 1000)
        poller.store_feed(self.conn, "system_pricing_plans", "u", feed(d, 2), 2000)  # new last_updated, same data
        self.assertEqual(self.count("SELECT COUNT(*) FROM system_pricing_plans"), 1)
        self.assertEqual(self.count("SELECT price FROM system_pricing_plans"), 3.0)
        self.assertEqual(self.count("SELECT is_taxable FROM system_pricing_plans"), 0)
        self.assertEqual(self.count("SELECT COUNT(*) FROM fetch_log WHERE ok=1"), 2)
        self.assertEqual(self.count("SELECT MIN(record_count) FROM fetch_log"), 1)

    def test_empty_feeds(self):
        for name, data in [("system_alerts", {"alerts": []}), ("free_bike_status", {"bikes": []}),
                           ("ebikes_at_stations", {"stations": []}), ("system_hours", {"rental_hours": []})]:
            self.assertTrue(poller.store_feed(self.conn, name, "u", feed(data), 1000), name)

    def test_same_payload_from_two_feeds_is_archived_once_per_feed(self):
        """Two feeds can produce the same data object, most easily when both are empty.

        raw_payloads is keyed on (feed_name, payload_sha256), so neither feed
        loses its payload to the other.
        """
        data = {"stations": []}        # the real empty shape of both feeds
        self.assertTrue(poller.store_feed(self.conn, "station_information", "u", feed(data), 1000))
        self.assertTrue(poller.store_feed(self.conn, "ebikes_at_stations", "u", feed(data), 1000))
        self.assertEqual(self.count("SELECT COUNT(*) FROM raw_payloads"), 2)
        self.assertEqual(
            {r[0] for r in self.conn.execute("SELECT feed_name FROM raw_payloads")},
            {"station_information", "ebikes_at_stations"},
        )

    def test_on_change_skip_does_not_cross_feeds(self):
        """A matching hash from a DIFFERENT feed must not skip parsing.

        Both parsers read their own key out of one shared document, so the two
        feeds hash alike. Keying on the hash alone made the second feed inherit
        the first feed's record_count and never parse its own rows at all.
        """
        data = {"regions": [{"region_id": "r1"}, {"region_id": "r2"}],
                "alerts": [{"alert_id": "a1", "type": "SYSTEM_CLOSURE"}]}
        self.assertTrue(poller.store_feed(self.conn, "system_regions", "u", feed(data), 1000))
        self.assertTrue(poller.store_feed(self.conn, "system_alerts", "u", feed(data), 2000))
        self.assertEqual(self.count("SELECT COUNT(*) FROM system_regions"), 2)
        self.assertEqual(self.count("SELECT COUNT(*) FROM system_alerts"), 1)
        counts = dict(self.conn.execute(
            "SELECT feed_name, record_count FROM fetch_log WHERE ok=1"))
        self.assertEqual(counts, {"system_regions": 2, "system_alerts": 1})

    def test_bad_json_logs_failure_and_keeps_earlier_data(self):
        poller.store_feed(self.conn, "station_status", "u", feed({"stations": [{"station_id": "a"}]}), 1000)
        self.assertFalse(poller.store_feed(self.conn, "station_status", "u", b"<html>oops", 1600))
        self.assertEqual(self.count("SELECT COUNT(*) FROM station_status"), 1)
        row = self.conn.execute("SELECT ok, error FROM fetch_log ORDER BY fetch_id DESC LIMIT 1").fetchone()
        self.assertEqual(row[0], 0)
        self.assertIn("JSONDecodeError", row[1])

    def test_parser_failure_rolls_back_everything(self):
        bad = {"stations": {"not": "a list"}}
        self.assertFalse(poller.store_feed(self.conn, "station_information", "u", feed(bad), 1000))
        self.assertEqual(self.count("SELECT COUNT(*) FROM raw_payloads"), 0)
        self.assertEqual(self.count("SELECT COUNT(*) FROM fetch_log WHERE ok=1"), 0)
        self.assertEqual(self.count("SELECT COUNT(*) FROM fetch_log WHERE ok=0"), 1)

    def test_run_poll_hourly_rule(self):
        feeds = {"station_status": "s", "system_regions": "r", "unknown_feed": "x"}
        bodies = {"station_status": feed({"stations": []}),
                  "system_regions": feed({"regions": [{"region_id": 3, "name": "Boston"}]})}
        fetch = lambda name, url: poller.FetchResult(bodies[name], 1000, 200, None)
        self.assertEqual(poller.run_poll(self.conn, feeds, fetch, now=1000), (2, 0))
        self.assertEqual(poller.run_poll(self.conn, feeds, fetch, now=1600), (1, 0))   # regions not due
        self.assertEqual(poller.run_poll(self.conn, feeds, fetch, now=1000 + 3600), (2, 0))

    def test_run_poll_counts_fetch_failure(self):
        fetch = lambda name, url: poller.FetchResult(None, 1000, 503, "HTTP 503: down")
        self.assertEqual(poller.run_poll(self.conn, {"station_status": "s"}, fetch, now=1000), (0, 1))
        self.assertEqual(self.count("SELECT http_status FROM fetch_log WHERE ok=0"), 503)


if __name__ == "__main__":
    unittest.main()
