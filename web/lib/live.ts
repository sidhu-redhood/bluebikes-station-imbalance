/**
 * Live Bluebikes GBFS readings. Pure: no React, no storage. Nothing here writes to disk.
 * The feed identifies stations by UUID or long number; bikes.json uses the short name (e.g. "B32062"),
 * so readings are joined through station_information.short_name.
 */
export const LIVE_URL = "/api/live";
export const GBFS_BASE = "https://gbfs.lyft.com/gbfs/2.3/bos/en";

export interface LiveSnapshot {
  /** Feed's own last_updated, epoch seconds. */
  updated: number;
  /** bikes.json station id -> bikes docked, or null when the station is offline or missing. */
  readings: Record<string, number | null>;
}

interface InfoFeed { data: { stations: { station_id: string; short_name?: string }[] } }
interface StatusFeed {
  last_updated: number;
  data: { stations: { station_id: string; num_bikes_available: number; is_installed?: number; is_renting?: number }[] };
}

/** Join the two feeds into one reading per wanted station id. Offline stations read as null. */
export function buildSnapshot(info: InfoFeed, status: StatusFeed, wanted: string[]): LiveSnapshot {
  const byFeedId = new Map<string, string>();
  for (const s of info.data.stations) if (s.short_name) byFeedId.set(s.station_id, s.short_name);
  const bySlot = new Map<string, number | null>();
  for (const s of status.data.stations) {
    const short = byFeedId.get(s.station_id);
    if (!short) continue;
    const online = s.is_installed !== 0 && s.is_renting !== 0;
    bySlot.set(short, online && Number.isFinite(s.num_bikes_available) ? s.num_bikes_available : null);
  }
  const readings: Record<string, number | null> = {};
  for (const id of wanted) readings[id] = bySlot.get(id) ?? null;
  return { updated: status.last_updated, readings };
}

export function parseSnapshot(input: unknown): LiveSnapshot {
  const doc = input as Partial<LiveSnapshot> | null;
  if (!doc || typeof doc.updated !== "number" || !doc.readings || typeof doc.readings !== "object")
    throw new Error("live: unexpected response");
  return doc as LiveSnapshot;
}

/** Minutes since midnight in Boston, regardless of the device's time zone. */
export function bostonMinute(now: Date): number {
  const p = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "numeric", hour12: false }).formatToParts(now);
  const get = (t: string) => Number(p.find((x) => x.type === t)?.value ?? 0);
  return (get("hour") % 24) * 60 + get("minute");
}
