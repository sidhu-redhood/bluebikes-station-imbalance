"""Tag each station with the Boston neighborhood it sits in.

Reads lat and lon from the stations table, finds the neighborhood polygon that
contains each point, and writes the result to station_neighborhoods.

Rules
- A NULL neighborhood means the station is outside every polygon. That is
  expected for Cambridge, Somerville, Brookline, and so on.
- A station that falls in more than one polygon is NOT tagged. It is listed as
  a conflict, and a human decides. Any older row for that station is left alone.
- A station just outside every polygon (within --near-meters) is tagged NULL
  and listed in a near miss report. The script never moves it into a polygon.
- A point exactly on a polygon edge counts as inside that polygon. In practice
  that only happens on a shared edge, where it shows up as a conflict.
- Running it twice changes nothing: a row is only rewritten when its
  neighborhood or boundary file changes.

Run it (needs bluebikes.db built from schema.sql and filled by the poller):
    python3 polling/tag_neighborhoods.py --db /srv/bluebikes/bluebikes.db
    python3 polling/tag_neighborhoods.py --dry-run     # report only, write nothing

Standard library only. GeoJSON order is [longitude, latitude].
"""

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from polling import poller
except ImportError:          # run as a script from inside polling/
    import poller

HERE = Path(__file__).resolve().parent
DEFAULT_GEOJSON = HERE / "reference" / "boston_neighborhoods.geojson"
DEFAULT_NEAR_METERS = 100.0
METERS_PER_DEG_LAT = 111_200.0       # close enough at Boston's latitude
BBOX_MARGIN_PER_METER = 1 / 80_000   # degrees per meter, a bit generous on purpose

OUTSIDE, BOUNDARY, INSIDE = 0, 1, 2


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

@dataclass
class Polygon:
    outer: list                 # [(lon, lat), ...] closed ring
    holes: list                 # list of closed rings
    bbox: tuple                 # (min_lon, min_lat, max_lon, max_lat) of the outer ring


@dataclass
class Neighborhood:
    name: str
    polygons: list

    def contains(self, lon: float, lat: float) -> bool:
        return any(polygon_contains(p, lon, lat) for p in self.polygons)


def _bbox(ring) -> tuple:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def locate(x: float, y: float, ring) -> int:
    """Where a point sits against one closed ring: OUTSIDE, BOUNDARY or INSIDE.

    Ray casting. Cast a ray to the right and count edge crossings. An odd count
    means inside. Points on an edge or vertex are reported as BOUNDARY first.
    """
    inside = False
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if (cross == 0 and min(x1, x2) <= x <= max(x1, x2)
                and min(y1, y2) <= y <= max(y1, y2)):
            return BOUNDARY
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return INSIDE if inside else OUTSIDE


def polygon_contains(polygon: Polygon, x: float, y: float) -> bool:
    """Inside the outer ring and not strictly inside a hole. Boundaries count as inside."""
    min_x, min_y, max_x, max_y = polygon.bbox
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        return False
    if locate(x, y, polygon.outer) == OUTSIDE:
        return False
    return all(locate(x, y, hole) != INSIDE for hole in polygon.holes)


def distance_to_ring_m(lon: float, lat: float, ring) -> float:
    """Shortest distance in meters from a point to a ring's edges (flat map approximation)."""
    kx = METERS_PER_DEG_LAT * math.cos(math.radians(lat))
    ky = METERS_PER_DEG_LAT
    best = math.inf
    for i in range(len(ring) - 1):
        ax, ay = (ring[i][0] - lon) * kx, (ring[i][1] - lat) * ky
        bx, by = (ring[i + 1][0] - lon) * kx, (ring[i + 1][1] - lat) * ky
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            d = math.hypot(ax, ay)
        else:
            t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
            d = math.hypot(ax + t * dx, ay + t * dy)
        best = min(best, d)
    return best


def nearest_boundary(neighborhoods, lon: float, lat: float, limit_m: float):
    """(name, meters) of the closest polygon edge within limit_m, or None."""
    margin = limit_m * BBOX_MARGIN_PER_METER
    best = None
    for hood in neighborhoods:
        for poly in hood.polygons:
            min_x, min_y, max_x, max_y = poly.bbox
            if not (min_x - margin <= lon <= max_x + margin
                    and min_y - margin <= lat <= max_y + margin):
                continue
            for ring in [poly.outer] + poly.holes:
                d = distance_to_ring_m(lon, lat, ring)
                if d <= limit_m and (best is None or d < best[1]):
                    best = (hood.name, d)
    return best


# ----------------------------------------------------------------------
# Loading the boundary file
# ----------------------------------------------------------------------

def _parse_ring(raw, label: str) -> list:
    if not isinstance(raw, list) or len(raw) < 4:
        raise ValueError(f"{label}: a ring needs at least 4 points")
    ring = []
    for pt in raw:
        ok = (isinstance(pt, (list, tuple)) and len(pt) >= 2
              and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v) for v in pt[:2]))
        if not ok:
            raise ValueError(f"{label}: bad point {pt!r}")
        ring.append((float(pt[0]), float(pt[1])))
    if ring[0] != ring[-1]:
        raise ValueError(f"{label}: ring is not closed")
    return ring


def load_neighborhoods(path) -> list:
    """Read a GeoJSON FeatureCollection into Neighborhood objects. Fails loudly on bad data."""
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        raise ValueError("expected a GeoJSON FeatureCollection")

    neighborhoods, seen = [], set()
    for i, feature in enumerate(doc.get("features") or []):
        name = (feature.get("properties") or {}).get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"feature {i} has no properties.name")
        if name in seen:
            raise ValueError(f"duplicate neighborhood name: {name}")
        seen.add(name)

        geometry = feature.get("geometry") or {}
        kind, coords = geometry.get("type"), geometry.get("coordinates")
        if kind == "Polygon":
            polys = [coords]
        elif kind == "MultiPolygon":
            polys = coords
        else:
            raise ValueError(f"{name}: unsupported geometry type {kind!r}")
        if not isinstance(polys, list) or not polys:
            raise ValueError(f"{name}: no coordinates")

        polygons = []
        for j, poly in enumerate(polys):
            if not isinstance(poly, list) or not poly:
                raise ValueError(f"{name} polygon {j}: no rings")
            rings = [_parse_ring(r, f"{name} polygon {j} ring {k}") for k, r in enumerate(poly)]
            polygons.append(Polygon(rings[0], rings[1:], _bbox(rings[0])))
        neighborhoods.append(Neighborhood(name, polygons))

    if not neighborhoods:
        raise ValueError("the file has no features")
    return neighborhoods


# ----------------------------------------------------------------------
# Tagging
# ----------------------------------------------------------------------

UPSERT = (
    "INSERT INTO station_neighborhoods (station_id, neighborhood, boundary_file, tagged_at) "
    "VALUES (?, ?, ?, ?) "
    "ON CONFLICT(station_id) DO UPDATE SET "
    "neighborhood = excluded.neighborhood, boundary_file = excluded.boundary_file, "
    "tagged_at = excluded.tagged_at "
    "WHERE station_neighborhoods.neighborhood IS NOT excluded.neighborhood "
    "OR station_neighborhoods.boundary_file IS NOT excluded.boundary_file"
)


@dataclass
class TagResult:
    total: int = 0
    counts: dict = field(default_factory=dict)       # neighborhood -> stations
    outside: int = 0                                  # stations tagged NULL
    conflicts: list = field(default_factory=list)     # (station_id, name, [neighborhoods])
    near_misses: list = field(default_factory=list)   # (station_id, name, neighborhood, meters, lat, lon)
    bad_coords: list = field(default_factory=list)    # (station_id, name)
    rows_changed: int = 0


def _valid_coord(lat, lon) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (lat, lon)) and -90 <= lat <= 90 and -180 <= lon <= 180


def tag_stations(conn: sqlite3.Connection, neighborhoods, boundary_file: str,
                 near_meters: float = DEFAULT_NEAR_METERS, dry_run: bool = False,
                 now: Optional[int] = None) -> TagResult:
    """Tag every station. All writes happen in one transaction."""
    now = int(time.time()) if now is None else now
    result = TagResult()
    writes = []

    rows = conn.execute("SELECT station_id, name, lat, lon FROM stations ORDER BY station_id").fetchall()
    for station_id, name, lat, lon in rows:
        result.total += 1
        if not _valid_coord(lat, lon):
            result.bad_coords.append((station_id, name))
            continue

        hits = [h.name for h in neighborhoods if h.contains(lon, lat)]   # lon first, always
        if len(hits) > 1:
            result.conflicts.append((station_id, name, hits))
            continue

        if hits:
            tag = hits[0]
            result.counts[tag] = result.counts.get(tag, 0) + 1
        else:
            tag = None
            result.outside += 1
            near = nearest_boundary(neighborhoods, lon, lat, near_meters)
            if near:
                result.near_misses.append((station_id, name, near[0], near[1], lat, lon))
        writes.append((station_id, tag, boundary_file, now))

    result.near_misses.sort(key=lambda m: m[3])
    if writes and not dry_run:
        before = conn.total_changes
        with poller.transaction(conn):
            conn.executemany(UPSERT, writes)
        result.rows_changed = conn.total_changes - before
    return result


def print_report(result: TagResult, dry_run: bool, near_meters: float) -> None:
    tagged = sum(result.counts.values())
    print(f"stations: {result.total}")
    print(f"  tagged with a neighborhood: {tagged}")
    print(f"  outside every polygon (NULL): {result.outside}")
    print(f"  conflicts, not tagged: {len(result.conflicts)}")
    print(f"  bad or missing coordinates, not tagged: {len(result.bad_coords)}")
    print("rows written: " + ("none, dry run" if dry_run else str(result.rows_changed)))

    if result.counts:
        print("\nper neighborhood:")
        for name, n in sorted(result.counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:4d}  {name}")
    if result.conflicts:
        print("\nCONFLICTS, in more than one polygon (needs a decision):")
        for sid, name, hoods in result.conflicts:
            print(f"  {sid}  {name}  ->  {', '.join(hoods)}")
    if result.bad_coords:
        print("\nbad or missing coordinates:")
        for sid, name in result.bad_coords:
            print(f"  {sid}  {name}")
    if result.near_misses:
        print(f"\nnear misses, tagged NULL but within {near_meters:g} m of a polygon (review only):")
        for sid, name, hood, meters, lat, lon in result.near_misses:
            print(f"  {meters:6.1f} m  {sid}  {name}  (next to {hood})  {lat:.5f}, {lon:.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=poller.DEFAULT_DB)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--near-meters", type=float, default=DEFAULT_NEAR_METERS)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    neighborhoods = load_neighborhoods(args.geojson)
    conn = poller.connect(args.db)
    try:
        result = tag_stations(conn, neighborhoods, args.geojson.name,
                              near_meters=args.near_meters, dry_run=args.dry_run)
    finally:
        conn.close()
    print_report(result, args.dry_run, args.near_meters)
    return 1 if result.conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
