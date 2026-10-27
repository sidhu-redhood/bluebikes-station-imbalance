"use client";

import "mapbox-gl/dist/mapbox-gl.css";
import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { useEffect, useMemo, useState } from "react";
import Map, { Marker, NavigationControl, ScaleControl, useMap } from "react-map-gl/mapbox";
import { DeckOverlay, type InterleavedLayerProps } from "@/components/DeckOverlay";
import { BIKES_URL, parseBikes, type BikeStation, type BikesFile } from "@/lib/bikes";
import { BASEMAP_CONFIG, BOSTON_OVERVIEW, BOSTON_VIEW, MAP_STYLE, NORTHEASTERN } from "@/lib/mapConfig";
import { compile, hitEmpty, hitFull, rng, simulate, type Compiled, type DayTrace } from "@/lib/markov";
import {
  NEIGHBORHOODS_URL,
  NEIGHBORHOOD_COLORS,
  PLACES,
  neighborhoodColor,
  neighborhoodLabel,
  type NeighborhoodCollection,
  type PlaceTarget,
} from "@/lib/neighborhoods";

const END = 24 * 60;
type Rgba = [number, number, number, number];
const EMPTY: Rgba = [200, 36, 59, 235];
const MID: Rgba = [142, 151, 159, 235];
const FULL: Rgba = [31, 99, 180, 235];
const lerp = (a: Rgba, b: Rgba, t: number): Rgba => a.map((v, k) => Math.round(v + (b[k] - v) * t)) as Rgba;
const fillColor = (f: number) => (f < 0.5 ? lerp(EMPTY, MID, f / 0.5) : lerp(MID, FULL, (f - 0.5) / 0.5));
const hhmm = (m: number) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.floor(m % 60)).padStart(2, "0")}`;

// A station is "at risk" when the Markov forecast says it will likely run out or
// fill up within 30 minutes. This is the model's prediction, not the current count.
const RISK_ON = 0.35;

interface Live { st: BikeStation; c: Compiled; trace: DayTrace }
interface Row { live: Live; bikes: number; pE: number; pF: number }

/** Flies the camera to the 3D hero view or the wide overview when `mode` changes.
 *  Must be rendered inside <Map> so useMap() can reach the map instance. */
function CameraControl({ mode }: { mode: "3d" | "overview" }) {
  const { current: mapRef } = useMap();
  useEffect(() => {
    const map = mapRef?.getMap();
    if (!map) return;
    const v = mode === "3d" ? BOSTON_VIEW : BOSTON_OVERVIEW;
    map.flyTo({
      center: [v.longitude, v.latitude],
      zoom: v.zoom,
      pitch: v.pitch,
      bearing: v.bearing,
      duration: 1200,
      essential: true,
    });
  }, [mapRef, mode]);
  return null;
}

/** Flies the camera to a chosen place (neighborhood button), pitched like the 3D view. */
function FlyTo({ target }: { target: (PlaceTarget & { nonce: number }) | null }) {
  const { current: mapRef } = useMap();
  useEffect(() => {
    const map = mapRef?.getMap();
    if (!map || !target) return;
    map.flyTo({
      center: [target.longitude, target.latitude],
      zoom: target.zoom,
      pitch: 55,
      bearing: -17.5,
      duration: 1400,
      essential: true,
    });
  }, [mapRef, target]);
  return null;
}

/** A small labeled probability bar (0..1) for the station forecast panel. */
function RiskBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span>{label}</span>
        <span className="font-semibold tabular-nums">{pct}%</span>
      </div>
      <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

/** A ranked list of stations with their Markov risk, used in the analysis panel. */
function AnalysisList({
  title,
  dot,
  rows,
  prob,
  onPick,
}: {
  title: string;
  dot: string;
  rows: Row[];
  prob: (d: Row) => number;
  onPick: (d: Row) => void;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-xs font-semibold">
        <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: dot }} />
        {title}
      </div>
      {rows.length === 0 ? (
        <p className="mt-1 text-xs text-muted-foreground">None above 5% right now.</p>
      ) : (
        <ul className="mt-1 space-y-1">
          {rows.map((d) => (
            <li key={d.live.st.id}>
              <button
                type="button"
                onClick={() => onPick(d)}
                className="flex w-full items-center justify-between gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-muted"
              >
                <span className="truncate">{d.live.st.name}</span>
                <span className="shrink-0 font-semibold tabular-nums">{Math.round(prob(d) * 100)}%</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function useBikes() {
  const [state, setState] = useState<{ status: "loading" } | { status: "ready"; doc: BikesFile } | { status: "error"; message: string }>({ status: "loading" });
  useEffect(() => {
    fetch(BIKES_URL)
      .then((r) => r.json())
      .then((j) => setState({ status: "ready", doc: parseBikes(j) }))
      .catch((e: Error) => setState({ status: "error", message: e.message }));
  }, []);
  return state;
}

/** Loads the NU-area neighborhood polygons drawn under the stations. */
function useNeighborhoods() {
  const [geo, setGeo] = useState<NeighborhoodCollection | null>(null);
  useEffect(() => {
    fetch(NEIGHBORHOODS_URL)
      .then((r) => r.json())
      .then((j: NeighborhoodCollection) => setGeo(j))
      .catch(() => setGeo(null)); // the map still works without the overlay
  }, []);
  return geo;
}

/** The Bluebikes tracker on CoolBoston's map: Mapbox Standard basemap, deck.gl interleaved overlay. */
export default function BikesMapView({ token }: { token: string }) {
  const data = useBikes();
  const hoods = useNeighborhoods();
  const [seed, setSeed] = useState(4100);
  const [minute, setMinute] = useState(7.5 * 60);
  const [playing, setPlaying] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);
  const [view, setView] = useState<"3d" | "overview">("3d");
  const [pulse, setPulse] = useState(0); // 0..1 breathing phase for risk halos
  const [flyTarget, setFlyTarget] = useState<(PlaceTarget & { nonce: number }) | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(true);

  // Compile the hourly matrices once per file; simulate one day per seed.
  const live = useMemo<Live[] | null>(() => {
    if (data.status !== "ready") return null;
    const { doc } = data, r = rng(seed);
    return doc.stations.map((st) => ({
      st,
      c: compile(st, doc.stepMinutes),
      trace: simulate(st, st.initialBikes, r, doc.startMinute, END - doc.startMinute),
    }));
  }, [data, seed]);
  const start = data.status === "ready" ? data.doc.startMinute : 300;
  const step = data.status === "ready" ? data.doc.stepMinutes : 10;
  const destCount = data.status === "ready" ? data.doc.stations.filter((s) => s.isAmDestination).length : 0;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setMinute((m) => (m + 1 >= END ? (setPlaying(false), END - 1) : m + 1)), 1000 / 6);
    return () => clearInterval(id);
  }, [playing]);

  // A breathing phase so at-risk stations visibly pulse while the model runs. Only animate
  // while playing — a 60fps setState when idle was re-rendering the whole map for no reason.
  useEffect(() => {
    if (!playing) {
      setPulse(0.5);
      return;
    }
    let raf = 0;
    const tick = () => {
      setPulse((Math.sin(Date.now() / 450) + 1) / 2); // 0..1
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // Forecast from the last 10-minute poll, as the GBFS tracker would.
  const poll = Math.max(start, Math.floor(minute / step) * step);
  const rows = useMemo<Row[]>(() => {
    if (!live) return [];
    const k = Math.round(30 / step), at = (l: Live, m: number) => l.trace.bikes[Math.min(Math.round(m) - start, l.trace.bikes.length - 1)];
    return live.map((l) => {
      const b0 = at(l, poll);
      return { live: l, bikes: at(l, minute), pE: hitEmpty(l.c, b0, poll, k), pF: hitFull(l.c, b0, poll, k) };
    });
  }, [live, minute, poll, start, step]);

  // The model's dominant 30-min risk for a station right now: how big and what color the halo is.
  const riskOf = (d: Row) => {
    const emptyRisk = d.pE, fullRisk = d.pF;
    const p = Math.max(emptyRisk, fullRisk);
    const color: Rgba = emptyRisk >= fullRisk ? [200, 36, 59, 255] : [31, 99, 180, 255];
    return { p, color };
  };

  // Live counts of what the Markov model predicts at this minute.
  const atRisk = useMemo(() => {
    let empty = 0, full = 0;
    for (const d of rows) {
      if (Math.max(d.pE, d.pF) < RISK_ON) continue;
      if (d.pE >= d.pF) empty++; else full++;
    }
    return { empty, full };
  }, [rows]);

  // Markov analysis: the stations most likely to run out or fill up, ranked by the model.
  const analysis = useMemo(() => {
    const topEmpty = [...rows].filter((d) => d.pE >= 0.05).sort((a, b) => b.pE - a.pE).slice(0, 5);
    const topFull = [...rows].filter((d) => d.pF >= 0.05).sort((a, b) => b.pF - a.pF).slice(0, 5);
    return { topEmpty, topFull };
  }, [rows]);

  const layers = useMemo(
    () => [
      // NU-area neighborhood regions, tinted per the reference "Neighborhoods Near NU" map.
      ...(hoods
        ? [
            new GeoJsonLayer({
              id: "neighborhoods",
              slot: "bottom",
              // Runtime shape is valid GeoJSON; cast past deck.gl's stricter geometry union.
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              data: hoods as any,
              stroked: true,
              filled: true,
              getFillColor: (f: { properties?: { name?: string } }) => {
                const c = neighborhoodColor(f.properties?.name);
                return [c[0], c[1], c[2], 55];
              },
              getLineColor: (f: { properties?: { name?: string } }) => {
                const c = neighborhoodColor(f.properties?.name);
                return [c[0], c[1], c[2], 180];
              },
              lineWidthMinPixels: 1.5,
              pickable: false,
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
            } as any),
          ]
        : []),
      // Risk halos (under the dots): a pulsing ring whose size and opacity grow with the
      // model's forecast probability of running out / filling up in the next 30 min.
      new ScatterplotLayer<Row, InterleavedLayerProps>({
        id: "risk-halo",
        slot: "top",
        data: rows,
        getPosition: (d) => [d.live.st.lon, d.live.st.lat],
        getRadius: (d) => {
          const { p } = riskOf(d);
          if (p < RISK_ON) return 0;
          const base = 9 + Math.sqrt(d.live.st.capacity) * 3;
          return base + p * (26 + pulse * 22); // breathe with the pulse clock
        },
        radiusUnits: "meters",
        radiusMinPixels: 0,
        getFillColor: (d) => {
          const { p, color } = riskOf(d);
          if (p < RISK_ON) return [0, 0, 0, 0];
          return [color[0], color[1], color[2], Math.round(70 + p * 90)];
        },
        stroked: false,
        pickable: false,
        updateTriggers: { getRadius: [minute, rows, pulse], getFillColor: [minute, rows] },
      }),
      new ScatterplotLayer<Row, InterleavedLayerProps>({
        id: "bike-stations",
        slot: "top",
        data: rows,
        getPosition: (d) => [d.live.st.lon, d.live.st.lat],
        getRadius: (d) => 9 + Math.sqrt(d.live.st.capacity) * 3,
        radiusUnits: "meters",
        radiusMinPixels: 5,
        getFillColor: (d) => fillColor(d.bikes / d.live.st.capacity),
        // Morning-commute destinations get a gold outline; everything else white.
        getLineColor: (d) => (d.live.st.isAmDestination ? [245, 190, 70, 255] : [255, 255, 255, 255]),
        getLineWidth: (d) => (d.live.st.isAmDestination ? 2.6 : 1.2),
        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1.2,
        stroked: true,
        pickable: true,
        onClick: ({ index }) => setPicked(index),
        updateTriggers: {
          getFillColor: [minute, rows],
          getLineColor: [rows],
          getLineWidth: [rows],
        },
      }),
    ],
    [rows, minute, pulse, hoods],
  );

  const sel = picked !== null ? rows[picked] : null;

  return (
    <div className="relative h-full w-full">
      <Map
        mapboxAccessToken={token}
        mapStyle={MAP_STYLE}
        config={{ basemap: BASEMAP_CONFIG }}
        initialViewState={BOSTON_VIEW}
        maxPitch={75}
        style={{ width: "100%", height: "100%" }}
      >
        <DeckOverlay layers={layers} />
        <CameraControl mode={view} />
        <FlyTo target={flyTarget} />

        {/* Northeastern campus anchor: the destination the cluster is built around. */}
        <Marker longitude={NORTHEASTERN.longitude} latitude={NORTHEASTERN.latitude} anchor="bottom">
          <div className="flex flex-col items-center">
            <span className="whitespace-nowrap rounded-md bg-[#c8102e] px-2 py-0.5 text-xs font-semibold text-white shadow-md">
              {NORTHEASTERN.label}
            </span>
            <span className="-mt-0.5 text-[#c8102e]" aria-hidden>▾</span>
          </div>
        </Marker>

        <NavigationControl position="top-left" visualizePitch />
        <ScaleControl position="bottom-right" unit="imperial" />
      </Map>

      <button
        type="button"
        aria-pressed={view === "3d"}
        onClick={() => setView((v) => (v === "3d" ? "overview" : "3d"))}
        className="absolute left-3 top-24 rounded-md border border-border bg-card px-3 py-1.5 text-sm font-medium shadow-sm"
      >
        {view === "3d" ? "Zoom out to all stations" : "3D downtown view"}
      </button>

      {/* Place buttons: jump the camera to each neighborhood. */}
      <nav aria-label="Jump to neighborhood" className="absolute left-1/2 top-3 flex max-w-[92vw] -translate-x-1/2 flex-wrap justify-center gap-1.5 rounded-lg border border-border bg-card/95 p-1.5 shadow-sm backdrop-blur">
        {PLACES.map((place) => {
          const [r, g, b] = neighborhoodColor(place.key);
          return (
            <button
              key={place.label}
              type="button"
              onClick={() => setFlyTarget({ ...place, nonce: Date.now() })}
              className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs font-medium hover:bg-muted"
            >
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: `rgb(${r},${g},${b})` }} />
              {place.label}
            </button>
          );
        })}
      </nav>

      {/* Live model HUD: makes the Markov model visible as the day plays. */}
      <aside aria-label="Markov model status" className="absolute left-3 top-36 w-60 rounded-lg border border-border bg-card/95 p-3 text-sm shadow-sm backdrop-blur">
        <div className="flex items-center justify-between">
          <span className="font-semibold">Markov model</span>
          <span className="tabular-nums text-muted-foreground">{hhmm(minute)}</span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          {playing ? "Simulating the day, station by station…" : "Press Play to run a simulated day."}
        </p>
        <dl className="mt-2 space-y-1">
          <div className="flex items-center justify-between">
            <dt className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "rgb(200,36,59)" }} />
              Likely to run empty
            </dt>
            <dd className="font-semibold tabular-nums">{atRisk.empty}</dd>
          </div>
          <div className="flex items-center justify-between">
            <dt className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "rgb(31,99,180)" }} />
              Likely to fill up
            </dt>
            <dd className="font-semibold tabular-nums">{atRisk.full}</dd>
          </div>
        </dl>
        <div className="mt-2 flex items-center justify-between border-t border-border pt-2">
          <dt className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full border-2" style={{ borderColor: "rgb(245,190,70)" }} />
            Morning destinations
          </dt>
          <dd className="font-semibold tabular-nums">{destCount}</dd>
        </div>
        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
          Gold-ringed stations fill during the 6–10am commute (bikes arrive faster than they
          leave) — the network flows toward Northeastern. Halos pulse where the model forecasts a
          {" "}{Math.round(RISK_ON * 100)}%+ chance of running out or filling within 30 min.
        </p>
      </aside>

      {/* Neighborhood legend for the NU-area cluster. */}
      <div aria-label="Neighborhoods" className="absolute bottom-3 right-3 rounded-lg border border-border bg-card/95 p-3 text-xs shadow-sm backdrop-blur">
        <div className="mb-1 font-semibold">Neighborhoods near NU</div>
        <ul className="space-y-0.5">
          {Object.keys(NEIGHBORHOOD_COLORS).map((key) => {
            const [r, g, b] = NEIGHBORHOOD_COLORS[key];
            return (
              <li key={key} className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: `rgb(${r},${g},${b})` }} />
                {neighborhoodLabel(key)}
                {key === "fenway" ? <span className="text-muted-foreground"> (incl. Kenmore)</span> : null}
              </li>
            );
          })}
        </ul>
      </div>

      <section aria-label="Simulated day" className="absolute bottom-3 left-3 right-3 flex items-center gap-3 rounded-lg border border-border bg-card p-3 text-sm md:right-auto md:w-[520px]">
        <button type="button" className="rounded-md bg-foreground px-3 py-1.5 text-background" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play"}
        </button>
        <span className="w-14 text-lg font-semibold tabular-nums">{hhmm(minute)}</span>
        <input aria-label="Time of day" type="range" min={start} max={END - 1} value={minute} onChange={(e) => setMinute(+e.target.value)} className="flex-1" />
        <button type="button" className="rounded-md border border-border px-2 py-1" onClick={() => setSeed((s) => (s * 1103515245 + 12345) >>> 0)}>
          New day
        </button>
      </section>

      {/* Markov analysis: live ranking of the stations most likely to run out / fill up. */}
      <aside aria-label="Markov analysis" className={`absolute right-3 w-72 rounded-lg border border-border bg-card/95 p-3 text-sm shadow-sm backdrop-blur ${sel ? "top-[19rem]" : "top-3"}`}>
        <div className="flex items-center justify-between">
          <span className="font-semibold">At-risk stations · {hhmm(minute)}</span>
          <button type="button" className="text-xs text-muted-foreground" onClick={() => setShowAnalysis((v) => !v)}>
            {showAnalysis ? "hide" : "show"}
          </button>
        </div>
        {showAnalysis && (
          <div className="mt-2 space-y-3">
            <AnalysisList
              title="Most likely to run OUT (30 min)"
              dot="rgb(200,36,59)"
              rows={analysis.topEmpty}
              prob={(d) => d.pE}
              onPick={(d) => {
                const idx = rows.indexOf(d);
                if (idx >= 0) setPicked(idx);
                setFlyTarget({ key: d.live.st.neighborhood ?? "", label: d.live.st.name, longitude: d.live.st.lon, latitude: d.live.st.lat, zoom: 16, nonce: Date.now() });
              }}
            />
            <AnalysisList
              title="Most likely to FILL UP (30 min)"
              dot="rgb(31,99,180)"
              rows={analysis.topFull}
              prob={(d) => d.pF}
              onPick={(d) => {
                const idx = rows.indexOf(d);
                if (idx >= 0) setPicked(idx);
                setFlyTarget({ key: d.live.st.neighborhood ?? "", label: d.live.st.name, longitude: d.live.st.lon, latitude: d.live.st.lat, zoom: 16, nonce: Date.now() });
              }}
            />
            <p className="text-[11px] leading-snug text-muted-foreground">
              Probabilities are the Markov model&apos;s 30-minute run-out / fill-up forecast from the
              {" "}{hhmm(poll)} poll. Click a station to fly there.
            </p>
          </div>
        )}
      </aside>

      {sel && (
        <aside aria-label="Station" className="absolute right-3 top-3 w-72 rounded-lg border border-border bg-card p-4 text-sm">
          <div className="flex items-start justify-between gap-2">
            <h2 className="font-semibold">{sel.live.st.name}</h2>
            <button type="button" aria-label="Close" className="text-muted-foreground" onClick={() => setPicked(null)}>×</button>
          </div>
          {sel.live.st.neighborhood && (
            <p className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: `rgb(${neighborhoodColor(sel.live.st.neighborhood).join(",")})` }}
              />
              {neighborhoodLabel(sel.live.st.neighborhood)}
            </p>
          )}
          <p className="mt-1 text-2xl font-semibold tabular-nums">
            {sel.bikes} <span className="text-sm font-normal text-muted-foreground">of {sel.live.st.capacity} docks</span>
          </p>
          {/* Live fill bar: moves every minute while the simulation plays. */}
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted" aria-hidden>
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${Math.round((sel.bikes / sel.live.st.capacity) * 100)}%`,
                background: `rgb(${fillColor(sel.bikes / sel.live.st.capacity).slice(0, 3).join(",")})`,
              }}
            />
          </div>
          {sel.live.st.isAmDestination && (
            <p className="mt-2 flex items-center gap-1.5 rounded-md bg-[#fff6e0] px-2 py-1 text-xs text-[#7a5600]">
              <span className="inline-block h-2.5 w-2.5 rounded-full border-2" style={{ borderColor: "rgb(245,190,70)" }} />
              Morning commute destination
              {typeof sel.live.st.amNetFlow === "number" ? ` (+${sel.live.st.amNetFlow.toFixed(1)} bikes/hr in, 6–10am)` : ""}
            </p>
          )}
          <p className="mt-3 text-muted-foreground">Markov forecast from the {hhmm(poll)} poll (updates live):</p>
          <div className="mt-1 space-y-1.5">
            <RiskBar label="Runs out within 30 min" value={sel.pE} color="rgb(200,36,59)" />
            <RiskBar label="Fills up within 30 min" value={sel.pF} color="rgb(31,99,180)" />
          </div>
        </aside>
      )}
      {data.status === "error" && (
        <p role="alert" className="absolute left-1/2 top-2 -translate-x-1/2 rounded-md bg-destructive px-3 py-1 text-xs text-white">
          Could not load bike data: {data.message}
        </p>
      )}
    </div>
  );
}
