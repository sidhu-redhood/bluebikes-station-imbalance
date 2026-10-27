"""Fetch the Bluebikes GBFS discovery file and save the feed URLs.

Reads the feed list from gbfs.json and writes a simple name to URL map to
polling/feeds/feeds.json. Both GBFS layouts are accepted: v2 nests the list
under a language key (data -> en -> feeds), v3 puts it at data -> feeds.

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
# Lyft serves GBFS from a CDN. The default Python-urllib agent is an easy
# target for a 403, so say who we are, the same way poller.py does.
USER_AGENT = "bluebikes-imbalance-poller/1.0 (Northeastern CS5100 student project)"

OUTPUT_PATH = Path(__file__).resolve().parent / "feeds" / "feeds.json"


class FeedError(Exception):
    """Raised when the discovery file cannot be fetched or understood."""


def download_gbfs(url: str) -> dict:
    """Download the discovery file and return it as a dict."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
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


def select_feed_list(gbfs: dict) -> list:
    """Find the feed list, whichever GBFS layout the discovery file uses.

    v2 nests it under a language key (data -> en -> feeds). v3 drops the
    language and puts it at data -> feeds. We prefer en and fall back to the
    first language present, so a feed that stops publishing en still works.
    """
    data = gbfs.get("data") if isinstance(gbfs, dict) else None
    if not isinstance(data, dict):
        raise FeedError("Unexpected structure: no data object")

    if isinstance(data.get("feeds"), list):        # v3
        return data["feeds"]

    for language in ("en", *sorted(data)):          # v2, en first
        section = data.get(language)
        if isinstance(section, dict) and isinstance(section.get("feeds"), list):
            return section["feeds"]

    raise FeedError(
        "Unexpected structure: expected data -> feeds or data -> <language> -> feeds"
    )


def parse_feeds(gbfs: dict) -> dict:
    """Turn the discovery file into a {name: url} map and check required feeds."""
    feed_list = select_feed_list(gbfs)
    if not feed_list:
        raise FeedError("The feed list is empty")

    feeds = {}
    for entry in feed_list:
        if not isinstance(entry, dict):
            raise FeedError(f"Feed entry is not an object: {entry!r}")
        name, url = entry.get("name"), entry.get("url")
        # Check the values, not just the keys. A null or empty url passes a
        # key check and then fails on every poll inside urlopen.
        if not isinstance(name, str) or not name.strip():
            raise FeedError(f"Feed entry has no usable name: {entry!r}")
        if not isinstance(url, str) or not url.strip():
            raise FeedError(f"Feed {name!r} has no usable url: {entry!r}")
        if name in feeds:
            raise FeedError(f"Feed {name!r} is listed twice")
        feeds[name] = url

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
    except (ValueError, OSError):
        # ValueError covers JSONDecodeError and UnicodeDecodeError, which
        # read_text raises before json ever sees the bytes.
        return None


def save_feeds(path: Path, feeds: dict) -> None:
    """Write feeds.json safely: write a temp file, then swap it in."""
    temp_path = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(feeds, indent=2, sort_keys=True) + "\n")
        temp_path.replace(path)
    except OSError as err:
        # Do not leave a half written temp file behind for the next run.
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise FeedError(f"Could not write {path}: {err}") from err


def main() -> int:
    try:
        gbfs = download_gbfs(DISCOVERY_URL)
        feeds = parse_feeds(gbfs)

        if load_existing(OUTPUT_PATH) == feeds:
            print(f"No change. {len(feeds)} feeds already saved in {OUTPUT_PATH}")
            return 0

        save_feeds(OUTPUT_PATH, feeds)
    except FeedError as err:
        print(f"Error: {err}", file=sys.stderr)
        print("feeds.json was not changed.", file=sys.stderr)
        return 1

    print(f"Saved {len(feeds)} feeds to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
