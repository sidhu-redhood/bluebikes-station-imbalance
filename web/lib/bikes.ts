/**
 * bikes.json contract (pipeline/src/coolboston/build_bikes.py, schema v1). Pure: no React.
 * Rates are mean WEEKDAY bikes per hour, clock hours 0..23. ratesIn = returns, ratesOut = pickups.
 */
export const BIKES_SCHEMA_VERSION = 1;
export const BIKES_URL = `/api/bikes?v=${BIKES_SCHEMA_VERSION}`;

export interface BikeStation {
  id: string;
  name: string;
  lat: number;
  lon: number;
  capacity: number;
  initialBikes: number;
  ratesIn: number[];
  ratesOut: number[];
  /** Optional neighborhood tag (e.g. "fenway"), set when the build was scoped to a cluster. */
  neighborhood?: string;
  /** Net bikes arriving 6-10am (returns minus pickups). Positive = morning commute destination. */
  amNetFlow?: number;
  /** True when this station fills during the morning commute (amNetFlow > 0). */
  isAmDestination?: boolean;
}

export interface BikesFile {
  version: typeof BIKES_SCHEMA_VERSION;
  stepMinutes: number;
  startMinute: number;
  source: { kind: string; month: string | null; weekdays: number | null; generated: string };
  stations: BikeStation[];
}

export class BikeDataError extends Error {
  override name = "BikeDataError";
}

const STATION_FIELDS = ["id", "name", "lat", "lon", "capacity", "initialBikes", "ratesIn", "ratesOut"] as const;
const isRates = (r: unknown) =>
  Array.isArray(r) && r.length === 24 && r.every((x) => typeof x === "number" && Number.isFinite(x) && x >= 0);

/** Validate an untrusted bikes.json. Throws BikeDataError naming the first bad field. */
export function parseBikes(input: unknown): BikesFile {
  const doc = input as Partial<BikesFile> | null;
  if (!doc || typeof doc !== "object") throw new BikeDataError("bikes.json: expected a JSON object");
  if (doc.version !== BIKES_SCHEMA_VERSION)
    throw new BikeDataError(`bikes.json: version ${JSON.stringify(doc.version)}, expected ${BIKES_SCHEMA_VERSION}`);
  if (!Array.isArray(doc.stations) || doc.stations.length === 0) throw new BikeDataError("bikes.json: no stations");
  doc.stations.forEach((s, i) => {
    const where = `stations[${i}]`;
    for (const k of STATION_FIELDS) if (!(k in s)) throw new BikeDataError(`${where}: missing field "${k}"`);
    if (!Number.isInteger(s.capacity) || s.capacity < 1) throw new BikeDataError(`${where}.capacity: expected a positive integer`);
    if (!(s.initialBikes >= 0 && s.initialBikes <= s.capacity)) throw new BikeDataError(`${where}.initialBikes: outside 0..capacity`);
    if (!isRates(s.ratesIn)) throw new BikeDataError(`${where}.ratesIn: expected 24 non-negative numbers`);
    if (!isRates(s.ratesOut)) throw new BikeDataError(`${where}.ratesOut: expected 24 non-negative numbers`);
  });
  return doc as BikesFile;
}
