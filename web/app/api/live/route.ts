import raw from "@/data/bikes.json";
import { parseBikes } from "@/lib/bikes";
import { GBFS_BASE, buildSnapshot, type LiveSnapshot } from "@/lib/live";

// Live readings are never persisted: fetches use cache: "no-store" (so Next writes nothing to
// .next/cache), responses are no-store (so the browser keeps no copy), and the only copy on the
// server is two module variables that live and die with the process.
export const dynamic = "force-dynamic";

const ids = parseBikes(raw).stations.map((s) => s.id);
const INFO_TTL_MS = 60 * 60 * 1000, STATUS_TTL_MS = 20 * 1000;

let info: { at: number; doc: Parameters<typeof buildSnapshot>[0] } | null = null;
let last: { at: number; snap: LiveSnapshot } | null = null;

async function getJson<T>(name: string): Promise<T> {
  const r = await fetch(`${GBFS_BASE}/${name}.json`, { cache: "no-store", signal: AbortSignal.timeout(10_000) });
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export async function GET() {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  try {
    const now = Date.now();
    if (last && now - last.at < STATUS_TTL_MS) return new Response(JSON.stringify(last.snap), { headers });
    if (!info || now - info.at > INFO_TTL_MS) info = { at: now, doc: await getJson("station_information") };
    const status = await getJson<Parameters<typeof buildSnapshot>[1]>("station_status");
    last = { at: now, snap: buildSnapshot(info.doc, status, ids) };
    return new Response(JSON.stringify(last.snap), { headers });
  } catch (e) {
    return new Response(JSON.stringify({ error: (e as Error).message }), { status: 502, headers });
  }
}
