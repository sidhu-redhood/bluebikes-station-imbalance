import raw from "@/data/bikes.json";
import { parseBikes } from "@/lib/bikes";

// Static import, like /api/cells: a malformed bikes.json fails `next build`, not a user's request.
const body = JSON.stringify(parseBikes(raw));

export function GET() {
  return new Response(body, {
    headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400" },
  });
}
