"""Tests for fetch_feeds.py. Run from the repo root: python3 -m unittest polling.test_fetch_feeds -v"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from polling import fetch_feeds
except ImportError:          # run from inside polling/
    import fetch_feeds


def entry(name, url=None):
    return {"name": name, "url": url or f"https://example.test/{name}.json"}


REQUIRED = [entry("station_information"), entry("station_status")]


class ParseFeedsTest(unittest.TestCase):
    def test_v2_language_layout(self):
        feeds = fetch_feeds.parse_feeds({"data": {"en": {"feeds": REQUIRED}}})
        self.assertEqual(sorted(feeds), ["station_information", "station_status"])

    def test_v3_layout_without_language(self):
        """GBFS v3 drops the language key. v2 only would break on the day Lyft upgrades."""
        feeds = fetch_feeds.parse_feeds({"data": {"feeds": REQUIRED}})
        self.assertEqual(sorted(feeds), ["station_information", "station_status"])

    def test_falls_back_to_another_language(self):
        feeds = fetch_feeds.parse_feeds({"data": {"fr": {"feeds": REQUIRED}}})
        self.assertEqual(len(feeds), 2)

    def test_prefers_en_over_other_languages(self):
        doc = {"data": {"fr": {"feeds": [entry("station_information", "https://fr/si"),
                                         entry("station_status", "https://fr/ss")]},
                        "en": {"feeds": [entry("station_information", "https://en/si"),
                                         entry("station_status", "https://en/ss")]}}}
        self.assertEqual(fetch_feeds.parse_feeds(doc)["station_status"], "https://en/ss")

    def test_null_url_is_rejected(self):
        """A null url passes a key check and then fails inside urlopen on every poll."""
        doc = {"data": {"en": {"feeds": [{"name": "station_information", "url": None},
                                         entry("station_status")]}}}
        with self.assertRaises(fetch_feeds.FeedError):
            fetch_feeds.parse_feeds(doc)

    def test_empty_and_non_string_values_are_rejected(self):
        for bad in ({"name": "station_information", "url": "   "},
                    {"name": "station_information", "url": 42},
                    {"name": "", "url": "https://x"},
                    {"name": None, "url": "https://x"}):
            doc = {"data": {"en": {"feeds": [bad, entry("station_status")]}}}
            with self.assertRaises(fetch_feeds.FeedError, msg=repr(bad)):
                fetch_feeds.parse_feeds(doc)

    def test_duplicate_name_is_rejected(self):
        doc = {"data": {"en": {"feeds": REQUIRED + [entry("station_status", "https://other")]}}}
        with self.assertRaises(fetch_feeds.FeedError):
            fetch_feeds.parse_feeds(doc)

    def test_missing_required_feed(self):
        with self.assertRaises(fetch_feeds.FeedError):
            fetch_feeds.parse_feeds({"data": {"en": {"feeds": [entry("system_alerts")]}}})

    def test_empty_or_unknown_structure(self):
        for doc in ({"data": {"en": {"feeds": []}}}, {"data": {}}, {"data": None}, {}):
            with self.assertRaises(fetch_feeds.FeedError, msg=repr(doc)):
                fetch_feeds.parse_feeds(doc)


class DownloadTest(unittest.TestCase):
    def test_sends_an_explicit_user_agent(self):
        """The default Python-urllib agent is an easy 403 target on Lyft's CDN."""
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["ua"] = request.get_header("User-agent")
            return io.BytesIO(json.dumps({"data": {}}).encode())

        with mock.patch.object(fetch_feeds.urllib.request, "urlopen", fake_urlopen):
            fetch_feeds.download_gbfs("https://example.test/gbfs.json")
        self.assertEqual(captured["ua"], fetch_feeds.USER_AGENT)

    def test_network_and_json_errors_become_feed_errors(self):
        for side in (OSError("no route"), ):
            with mock.patch.object(fetch_feeds.urllib.request, "urlopen", side_effect=side):
                with self.assertRaises(fetch_feeds.FeedError):
                    fetch_feeds.download_gbfs("https://example.test/gbfs.json")
        with mock.patch.object(fetch_feeds.urllib.request, "urlopen",
                               lambda r, timeout=None: io.BytesIO(b"<html>")):
            with self.assertRaises(fetch_feeds.FeedError):
                fetch_feeds.download_gbfs("https://example.test/gbfs.json")


class FileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "feeds.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_existing_survives_undecodable_bytes(self):
        """read_text raises UnicodeDecodeError before json ever sees the bytes."""
        self.path.write_bytes(b"\xff\xfe not utf 8")
        self.assertIsNone(fetch_feeds.load_existing(self.path))

    def test_load_existing_survives_bad_json_and_missing_file(self):
        self.path.write_text("{not json")
        self.assertIsNone(fetch_feeds.load_existing(self.path))
        self.assertIsNone(fetch_feeds.load_existing(self.path.with_name("absent.json")))

    def test_save_then_load_round_trip(self):
        feeds = {"station_status": "https://x/ss", "station_information": "https://x/si"}
        fetch_feeds.save_feeds(self.path, feeds)
        self.assertEqual(fetch_feeds.load_existing(self.path), feeds)
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())   # temp file swapped in

    def test_write_failure_raises_feed_error_and_leaves_no_temp_file(self):
        blocked = Path(self.tmp.name) / "a_file" / "feeds.json"
        blocked.parent.write_text("I am a file, not a directory")
        with self.assertRaises(fetch_feeds.FeedError):
            fetch_feeds.save_feeds(blocked, {"station_status": "https://x"})
        self.assertFalse(blocked.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
