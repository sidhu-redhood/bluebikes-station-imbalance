"""Point-in-neighborhood tagging from a GeoJSON of Boston neighborhood polygons.

Pure Python, no geo dependencies: a ray-casting point-in-polygon test over the
Polygon / MultiPolygon features in polling/reference/boston_neighborhoods.geojson.
Coordinates in that file are [longitude, latitude].

Used to scope the bike model to a named cluster of neighborhoods (e.g. the ones
around Northeastern) instead of a circle of a given radius.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GEOJSON = ROOT / "polling" / "reference" / "boston_neighborhoods.geojson"

# Kenmore is not its own polygon in this dataset; the reference README notes it is
# part of Fenway. So a request for "Kenmore" is satisfied by the Fenway polygon.
ALIASES = {"kenmore": "fenway"}


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    """Ray casting: is (lon, lat) inside this closed ring of [lon, lat] points?"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        # Does the horizontal ray at `lat` cross the edge (i, j)?
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, polygon: list) -> bool:
    """A GeoJSON Polygon: [outer_ring, hole1, hole2, ...]. Inside outer, outside holes."""
    if not polygon:
        return False
    if not _point_in_ring(lon, lat, polygon[0]):
        return False
    for hole in polygon[1:]:
        if _point_in_ring(lon, lat, hole):
            return False
    return True


def _point_in_feature(lon: float, lat: float, geometry: dict) -> bool:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        return _point_in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        return any(_point_in_polygon(lon, lat, poly) for poly in coords)
    return False


class Neighborhoods:
    """Loaded neighborhood polygons, keyed by lowercased name."""

    def __init__(self, features_by_name: dict[str, list[dict]]):
        self._by_name = features_by_name

    @classmethod
    def load(cls, path: Path = DEFAULT_GEOJSON) -> "Neighborhoods":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        by_name: dict[str, list[dict]] = {}
        for feat in data.get("features", []):
            name = (feat.get("properties") or {}).get("name")
            if not name:
                continue
            by_name.setdefault(name.strip().lower(), []).append(feat["geometry"])
        return cls(by_name)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def resolve(self, name: str) -> str:
        key = name.strip().lower()
        return ALIASES.get(key, key)

    def name_for(self, lon: float, lat: float, among: list[str] | None = None) -> str | None:
        """The neighborhood containing (lon, lat), optionally restricted to `among`.
        Returns the canonical lowercased polygon name, or None if outside all."""
        candidates = (
            [self.resolve(n) for n in among] if among is not None else list(self._by_name)
        )
        for key in candidates:
            for geom in self._by_name.get(key, []):
                if _point_in_feature(lon, lat, geom):
                    return key
        return None

    def contains(self, lon: float, lat: float, among: list[str]) -> bool:
        return self.name_for(lon, lat, among) is not None
