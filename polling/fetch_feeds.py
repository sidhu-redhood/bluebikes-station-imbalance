"""Fetch the Bluebikes GBFS discovery file and save the feed URLs.

Reads data -> en -> feeds from gbfs.json and writes a simple
name to URL map to polling/feeds/feeds.json.

The raw gbfs.json is never saved. The file is only rewritten when the
result changes, and never when something looks wrong.

Run from anywhere:
    python polling/fetch_feeds.py
"""

import http.client
import json
import sys
import urllib.request
from pathlib import Path

DISCOVERY_URL = "https://gbfs.bluebikes.com/gbfs/gbfs.json"
REQUIRED_FEEDS = ("station_information", "station_status")
TIMEOUT_SECONDS = 15

OUTPUT_PATH = Path(__file__).resolve().parent / "feeds" / "feeds.json"


class FeedError(Exception):
    """Raised when the discovery file cannot be fetched or understood."""


def download_gbfs(url: str) -> dict:
    """Download the discovery file and return it as a dict."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            body = response.read()
    except (http.client.HTTPException, OSError) as err:
        # OSError already covers URLError, HTTPError and TimeoutError.
        # HTTPException covers cut off or malformed responses (IncompleteRead).
        raise FeedError(f"Could not download {url}: {err}") from err

    try:
        return json.loads(body)
    except ValueError as err:
        # JSONDecodeError and UnicodeDecodeError are both ValueErrors.
        raise FeedError(f"Response from {url} is not valid JSON: {err}") from err


def parse_feeds(gbfs: dict) -> dict:
    """Turn data.en.feeds into a {name: url} map and check required feeds."""
    try:
        feed_list = gbfs["data"]["en"]["feeds"]
    except (KeyError, TypeError) as err:
        raise FeedError(
            "Unexpected structure: expected data -> en -> feeds"
        ) from err

    if not isinstance(feed_list, list) or not feed_list:
        raise FeedError("data -> en -> feeds is empty or not a list")

    feeds = {}
    for entry in feed_list:
        if not isinstance(entry, dict) or "name" not in entry or "url" not in entry:
            raise FeedError(f"Feed entry is missing name or url: {entry!r}")
        feeds[entry["name"]] = entry["url"]

    missing = [name for name in REQUIRED_FEEDS if name not in feeds]
    if missing:
        raise FeedError(f"Required feeds missing: {', '.join(missing)}")

    return feeds


def load_existing(path: Path):
    """Return the saved feeds map, or None if there is no usable file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def save_feeds(path: Path, feeds: dict) -> None:
    """Write feeds.json safely: write a temp file, then swap it in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(feeds, indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)


def main() -> int:
    try:
        gbfs = download_gbfs(DISCOVERY_URL)
        feeds = parse_feeds(gbfs)
    except FeedError as err:
        print(f"Error: {err}", file=sys.stderr)
        print("feeds.json was not changed.", file=sys.stderr)
        return 1

    if load_existing(OUTPUT_PATH) == feeds:
        print(f"No change. {len(feeds)} feeds already saved in {OUTPUT_PATH}")
        return 0

    save_feeds(OUTPUT_PATH, feeds)
    print(f"Saved {len(feeds)} feeds to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
