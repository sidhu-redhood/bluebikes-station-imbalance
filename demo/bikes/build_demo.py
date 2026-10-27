"""Build the standalone Bluebikes demo page (demo/bikes/dist/bluebikes-markov-demo.html).

Run from the repo root:  python3 demo/bikes/build_demo.py
Inputs: web/data/bikes.json (coolboston.build_bikes), web/data/cells.json (land tiles), demo/bikes/markov.js.
The page is one self-contained HTML file: no Mapbox token, no network except Google Fonts.
Land: CoolBoston's 100 m tiles within 1.8 km of the stations, in State Plane metres. Full tiles are
stored as [i, j] and rebuilt as squares; clipped tiles carry their rings, offset from the tile corner.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MARGIN_M = 1800

# EPSG:26986 forward projection, same constants as web/lib/grid.ts.
A = 6378137; F = 1 / 298.257222101; E = math.sqrt(2 * F - F * F)
L1, L2, L0, LON0 = map(math.radians, (42 + 41 / 60, 41 + 43 / 60, 41, -71.5))
FE, FN = 200_000, 750_000
_m = lambda p: math.cos(p) / math.sqrt(1 - (E * math.sin(p)) ** 2)  # noqa: E731
_t = lambda p: math.tan(math.pi / 4 - p / 2) / ((1 - E * math.sin(p)) / (1 + E * math.sin(p))) ** (E / 2)  # noqa: E731
N = (math.log(_m(L1)) - math.log(_m(L2))) / (math.log(_t(L1)) - math.log(_t(L2)))
AF = A * _m(L1) / (N * _t(L1) ** N)
RHO0 = AF * _t(L0) ** N


def to_state_plane(lng, lat):
    rho, th = AF * _t(math.radians(lat)) ** N, N * (math.radians(lng) - LON0)
    return FE + rho * math.sin(th), FN + RHO0 - rho * math.cos(th)


def land_near(cells, stations):
    xs, ys = zip(*(to_state_plane(s["lon"], s["lat"]) for s in stations))
    x0, x1, y0, y1 = min(xs) - MARGIN_M, max(xs) + MARGIN_M, min(ys) - MARGIN_M, max(ys) + MARGIN_M
    full, clip = [], []
    for c in cells:
        i, j = map(int, c["id"].split("_"))
        if not (x0 <= i * 100 <= x1 and y0 <= j * 100 <= y1):
            continue
        if c["coords"] is None:
            full.append([i, j])
        else:
            rings = lambda poly: [[[round(to_state_plane(*p)[0] - i * 100, 1), round(to_state_plane(*p)[1] - j * 100, 1)] for p in r] for r in poly]  # noqa: E731
            clip.append([i, j, [rings(poly) for poly in c["coords"]]])
    return {"origin": "EPSG:26986", "full": full, "clip": clip, "bbox": [x0, y0, x1, y1]}


def main():
    bikes = (ROOT / "web/data/bikes.json").read_text()
    cells = json.loads((ROOT / "web/data/cells.json").read_text())["cells"]
    land = land_near(cells, json.loads(bikes)["stations"])
    page = ((HERE / "template.html").read_text()
            .replace("/*MARKOV*/", (HERE / "markov.js").read_text())
            .replace("/*BIKES*/", bikes)
            .replace("/*LAND*/", json.dumps(land, separators=(",", ":"))))
    out = HERE / "dist" / "bluebikes-markov-demo.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({len(page) // 1024} KB, {len(land['full'])} full + {len(land['clip'])} clipped tiles)")


if __name__ == "__main__":
    main()
