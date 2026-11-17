"""Tests for health_check.py. Run from the repo root: python3 -m unittest polling.test_health_check -v"""
import contextlib
import io
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    from polling import health_check
except ImportError:          
    import health_check

SCHEMA = Path(__file__).resolve().parent / "schema.sql"


class FindTest(unittest.TestCase):
    """The pairwise checks, on plain lists. No database needed."""

    def test_gap(self):
        self.assertEqual(health_check.find_gaps([0, 600, 1200, 3600, 4200]), [(1200, 3600)])

    def test_jitter_is_not_a_gap(self):
        self.assertEqual(health_check.find_gaps([0, 620, 1250]), [])

    def test_empty_and_single(self):
        self.assertEqual(health_check.find_gaps([]), [])
        self.assertEqual(health_check.find_gaps([0]), [])
        self.assertEqual(health_check.find_duplicates([]), [])

    def test_duplicates_from_vm_setup(self):
        # The VM's first five polls: two manual runs, then the timer.
        self.assertEqual(health_check.find_duplicates([0, 73, 247, 847, 1447]),
                         [(0, 73), (73, 247)])


class DatabaseTest(unittest.TestCase):
    """Queries and main(), against a temporary database built from schema.sql."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        self.writer = sqlite3.connect(self.db)
        self.writer.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.now = int(time.time())

    def tearDown(self):
        self.writer.close()
        self.tmp.cleanup()

    def add_poll(self, minutes_ago, ok=1, error=None):
        self.writer.execute(
            "INSERT INTO fetch_log (feed_name, fetched_at, ok, error) VALUES ('station_status', ?, ?, ?)",
            (self.now - int(minutes_ago * 60), ok, error),
        )
        self.writer.commit()

    def query(self, fn, *args):
        conn = health_check.connect_readonly(self.db)
        try:
            return fn(conn, *args)
        finally:
            conn.close()

    def run_main(self):
        out = io.StringIO()
        with mock.patch("sys.argv", ["health_check.py", "--db", str(self.db)]), \
                contextlib.redirect_stdout(out):
            code = health_check.main()
        return code, out.getvalue()

    def test_last_good_poll_ignores_failures(self):
        self.add_poll(20)
        self.add_poll(10, ok=0, error="timeout")
        self.assertEqual(self.query(health_check.last_good_poll), self.now - 20 * 60)

    def test_last_good_poll_none_when_never_ok(self):
        self.add_poll(10, ok=0, error="timeout")
        self.assertIsNone(self.query(health_check.last_good_poll))

    def test_poll_times_window_and_order(self):
        self.add_poll(10)
        self.add_poll(25 * 60)   
        self.add_poll(20)
        since = self.now - health_check.WINDOW_SECONDS
        self.assertEqual(self.query(health_check.poll_times, since),
                         [self.now - 20 * 60, self.now - 10 * 60])

    def test_healthy_day(self):
        for minutes_ago in range(0, 24 * 60, 10):
            self.add_poll(minutes_ago)
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("144 good polls", out)
        self.assertNotIn("PROBLEM", out)

    def test_stale_poller(self):
        self.add_poll(45)
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("last good poll was 45 minutes ago", out)

    def test_gap_reported(self):
        for minutes_ago in (70, 60, 20, 10, 0):
            self.add_poll(minutes_ago)
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("gap of 40 minutes", out)

    def test_few_failures_are_ok(self):
        self.add_poll(5, ok=0, error="timeout")
        self.add_poll(0)
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("1 failures", out)

    def test_many_failures_reported(self):
        for minutes_ago in range(10, 80, 10):     
            self.add_poll(minutes_ago, ok=0, error="URLError: timed out")
        self.add_poll(0)
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("7 failed fetches", out)
        self.assertIn("URLError: timed out", out)

    def test_missing_database(self):
        with self.assertRaises(FileNotFoundError):
            health_check.connect_readonly(Path(self.tmp.name) / "nope.db")


if __name__ == "__main__":
    unittest.main()