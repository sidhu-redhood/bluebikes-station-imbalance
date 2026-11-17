"""Tests for the point-in-neighborhood tagging (no network)."""

from __future__ import annotations

from coolboston.neighborhoods import Neighborhoods, _point_in_polygon, _point_in_ring


# A unit square ring (closed), coordinates [lon, lat].
SQUARE = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]


def test_point_in_ring_basic():
    assert _point_in_ring(5, 5, SQUARE) is True
    assert _point_in_ring(15, 5, SQUARE) is False
    assert _point_in_ring(-1, 5, SQUARE) is False


def test_polygon_with_hole():
    hole = [[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]]
    poly = [SQUARE, hole]
    assert _point_in_polygon(1, 1, poly) is True   # inside outer, outside hole
    assert _point_in_polygon(5, 5, poly) is False  # inside the hole


def test_loads_real_geojson_and_has_nu_neighborhoods():
    hoods = Neighborhoods.load()
    names = set(hoods.names())
    for n in ("fenway", "back bay", "mission hill", "south end", "roxbury", "jamaica plain"):
        assert n in names


def test_kenmore_alias_resolves_to_fenway():
    hoods = Neighborhoods.load()
    assert hoods.resolve("Kenmore") == "fenway"


def test_northeastern_is_in_the_nu_cluster():
    """Northeastern's main campus (~42.3398, -71.0892) should fall inside one of the
    NU-area neighborhoods (Fenway) and not, say, Dorchester."""
    hoods = Neighborhoods.load()
    nu = ["Fenway", "Back Bay", "Mission Hill", "South End", "Roxbury", "Jamaica Plain"]
    assert hoods.name_for(-71.0892, 42.3398, nu) is not None


def test_far_point_is_outside_the_cluster():
    hoods = Neighborhoods.load()
    nu = ["Fenway", "Back Bay", "Mission Hill", "South End", "Roxbury", "Jamaica Plain"]
    # Salem, far north of Boston, is not in any Boston neighborhood.
    assert hoods.name_for(-70.8807, 42.5219, nu) is None
