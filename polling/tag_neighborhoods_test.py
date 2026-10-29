"""Tests for tag_neighborhoods.py. Run from the repo root:
    python3 -m unittest polling.test_tag_neighborhoods -v
"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

try:
    from polling import poller, tag_neighborhoods as tn
except ImportError:          # run from inside polling/
    import poller
    import tag_neighborhoods as tn

SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def ring(x0, y0, x1, y1):
    """A closed rectangle as [lon, lat] points."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def hood(name, *rects):
    """A Neighborhood made of plain rectangles (x0, y0, x1, y1)."""
    polys = [tn.Polygon([tuple(p) for p in ring(*r)], [], tn._bbox(ring(*r))) for r in rects]
    return tn.Neighborhood(name, polys)


def feature(name, geometry):
    return {"type": "Feature", "properties": {"name": name}, "geometry": geometry}


def write_geojson(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


class GeometryTest(unittest.TestCase):
    def test_square_inside_and_outside(self):
        a = hood("A", (0, 0, 1, 1))
        self.assertTrue(a.contains(0.5, 0.5))
        self.assertFalse(a.contains(1.5, 0.5))
        self.assertFalse(a.contains(0.5, -0.1))

    def test_hole_is_outside(self):
        outer = tn._parse_ring(ring(0, 0, 10, 10), "o")
        hole = tn._parse_ring(ring(4, 4, 6, 6), "h")
        poly = tn.Polygon(outer, [hole], tn._bbox(outer))
        self.assertTrue(tn.polygon_contains(poly, 2, 2))        # in the ring around the hole
        self.assertFalse(tn.polygon_contains(poly, 5, 5))       # inside the hole
        self.assertTrue(tn.polygon_contains(poly, 4, 5))        # on the hole edge: still polygon

    def test_multipolygon_parts_and_gap(self):
        m = hood("M", (0, 0, 1, 1), (5, 5, 6, 6))
        self.assertTrue(m.contains(0.5, 0.5))
        self.assertTrue(m.contains(5.5, 5.5))
        self.assertFalse(m.contains(3, 3))

    def test_boundary_and_vertex_count_as_inside(self):
        a = hood("A", (0, 0, 1, 1))
        self.assertTrue(a.contains(0.5, 0.0))     # on an edge
        self.assertTrue(a.contains(1.0, 1.0))     # on a corner

    def test_concave_polygon(self):
        # An L shape: the notch at the top right is outside.
        pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)]
        poly = tn.Polygon(pts, [], tn._bbox(pts))
        self.assertTrue(tn.polygon_contains(poly, 0.5, 1.5))
        self.assertFalse(tn.polygon_contains(poly, 1.5, 1.5))

    def test_lon_lat_order(self):
        # Boston-ish square. Correct order is (lon, lat). Swapped must not match.
        b = hood("B", (-71.10, 42.30, -71.00, 42.40))
        self.assertTrue(b.contains(-71.05, 42.35))
        self.assertFalse(b.contains(42.35, -71.05))

    def test_distance_to_ring(self):
        r = tn._parse_ring(ring(-71.10, 42.30, -71.00, 42.40), "r")
        d = tn.distance_to_ring_m(-71.05, 42.4005, r)        # 0.0005 deg north of the top edge
        self.assertAlmostEqual(d, 0.0005 * tn.METERS_PER_DEG_LAT, delta=0.5)


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "hoods.geojson"

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_polygon_and_multipolygon(self):
        write_geojson(self.path, [
            feature("A", {"type": "Polygon", "coordinates": [ring(0, 0, 1, 1)]}),
            feature("B", {"type": "MultiPolygon", "coordinates": [[ring(2, 2, 3, 3)], [ring(4, 4, 5, 5)]]}),
        ])
        hoods = tn.load_neighborhoods(self.path)
        self.assertEqual([h.name for h in hoods], ["A", "B"])
        self.assertEqual(len(hoods[1].polygons), 2)

    def test_rejects_bad_files(self):
        good = ring(0, 0, 1, 1)
        open_ring = good[:-1] + [[0.5, 0.5]]
        cases = {
            "open ring": [feature("A", {"type": "Polygon", "coordinates": [open_ring]})],
            "no name": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [good]}}],
            "duplicate name": [feature("A", {"type": "Polygon", "coordinates": [good]})] * 2,
            "bad type": [feature("A", {"type": "Point", "coordinates": [0, 0]})],
            "too few points": [feature("A", {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 0]]]})],
            "no features": [],
        }
        for label, features in cases.items():
            with self.subTest(label):
                write_geojson(self.path, features)
                with self.assertRaises(ValueError):
                    tn.load_neighborhoods(self.path)

    def test_rejects_non_collection(self):
        self.path.write_text(json.dumps({"type": "Feature"}))
        with self.assertRaises(ValueError):
            tn.load_neighborhoods(self.path)


class TaggingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "test.db"
        sqlite3.connect(db).executescript(SCHEMA.read_text())
        self.conn = poller.connect(db)
        # A and B share the edge at lon -71.00.
        self.hoods = [hood("A", (-71.10, 42.30, -71.00, 42.40)),
                      hood("B", (-71.00, 42.30, -70.90, 42.40))]

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add(self, sid, lat, lon, name=None):
        self.conn.execute(
            "INSERT INTO stations (station_id, name, lat, lon, first_seen, last_seen) VALUES (?,?,?,?,1,1)",
            (sid, name or sid, lat, lon))

    def tags(self):
        return dict(self.conn.execute("SELECT station_id, neighborhood FROM station_neighborhoods"))

    def test_end_to_end(self):
        self.add("in_a", 42.35, -71.05)
        self.add("in_b", 42.35, -70.95)
        self.add("far", 42.60, -71.05)              # nowhere near
        self.add("near", 42.4005, -71.05)           # about 56 m north of A
        self.add("shared", 42.35, -71.00)           # exactly on the A and B edge
        self.add("nocoords", None, None)
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", now=1000)

        self.assertEqual(self.tags(), {"in_a": "A", "in_b": "B", "far": None, "near": None})
        self.assertEqual(r.total, 6)
        self.assertEqual(r.counts, {"A": 1, "B": 1})
        self.assertEqual(r.outside, 2)
        self.assertEqual([c[0] for c in r.conflicts], ["shared"])
        self.assertEqual(sorted(r.conflicts[0][2]), ["A", "B"])
        self.assertEqual([b[0] for b in r.bad_coords], ["nocoords"])
        self.assertEqual([(m[0], m[2]) for m in r.near_misses], [("near", "A")])
        self.assertAlmostEqual(r.near_misses[0][3], 55.6, delta=0.5)
        self.assertEqual(r.near_misses[0][4:], (42.4005, -71.05))   # lat, lon for the map check
        self.assertEqual(r.rows_changed, 4)

    def test_near_miss_respects_the_limit(self):
        self.add("a_bit_far", 42.4015, -71.05)      # about 167 m north
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", near_meters=100, now=1000)
        self.assertEqual(r.near_misses, [])
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", near_meters=200, now=1000)
        self.assertEqual(len(r.near_misses), 1)

    def test_second_run_changes_nothing(self):
        self.add("in_a", 42.35, -71.05)
        self.add("far", 42.60, -71.05)
        tn.tag_stations(self.conn, self.hoods, "test.geojson", now=1000)
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", now=2000)
        self.assertEqual(r.rows_changed, 0)
        self.assertEqual(self.conn.execute("SELECT DISTINCT tagged_at FROM station_neighborhoods").fetchall(), [(1000,)])

    def test_changed_boundary_rewrites_rows(self):
        self.add("in_a", 42.35, -71.05)
        tn.tag_stations(self.conn, self.hoods, "v1.geojson", now=1000)
        moved = [hood("A2", (-71.10, 42.30, -71.00, 42.40))]
        r = tn.tag_stations(self.conn, moved, "v2.geojson", now=2000)
        self.assertEqual(r.rows_changed, 1)
        self.assertEqual(self.conn.execute(
            "SELECT neighborhood, boundary_file, tagged_at FROM station_neighborhoods").fetchone(),
            ("A2", "v2.geojson", 2000))

    def test_conflict_leaves_old_row_alone(self):
        self.add("shared", 42.35, -71.00)
        self.conn.execute("INSERT INTO station_neighborhoods VALUES ('shared', 'Old', 'old.geojson', 5)")
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", now=1000)
        self.assertEqual(len(r.conflicts), 1)
        self.assertEqual(self.tags(), {"shared": "Old"})

    def test_dry_run_writes_nothing(self):
        self.add("in_a", 42.35, -71.05)
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", dry_run=True, now=1000)
        self.assertEqual(r.counts, {"A": 1})
        self.assertEqual(self.tags(), {})

    def test_station_in_a_hole_is_outside(self):
        outer = tn._parse_ring(ring(-71.10, 42.30, -71.00, 42.40), "o")
        hole = tn._parse_ring(ring(-71.06, 42.34, -71.04, 42.36), "h")
        donut = tn.Neighborhood("Donut", [tn.Polygon(outer, [hole], tn._bbox(outer))])
        self.add("in_hole", 42.35, -71.05)
        self.add("in_ring", 42.31, -71.05)
        tn.tag_stations(self.conn, [donut], "test.geojson", now=1000)
        self.assertEqual(self.tags(), {"in_hole": None, "in_ring": "Donut"})

    def test_swapped_coordinates_stay_outside(self):
        # lat and lon stored the wrong way round must not land in a Boston polygon.
        self.add("swapped", -71.05, 42.35)
        tn.tag_stations(self.conn, self.hoods, "test.geojson", now=1000)
        self.assertEqual(self.tags(), {"swapped": None})

    def test_out_of_range_coordinates_are_bad(self):
        self.add("bad", 142.0, -71.05)
        r = tn.tag_stations(self.conn, self.hoods, "test.geojson", now=1000)
        self.assertEqual([b[0] for b in r.bad_coords], ["bad"])
        self.assertEqual(self.tags(), {})


if __name__ == "__main__":
    unittest.main()
