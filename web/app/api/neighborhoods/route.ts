import raw from "@/data/neighborhoods.json";

// Static import, like /api/bikes: the NU-area neighborhood polygons ship with the app.
const body = JSON.stringify(raw);

export function GET() {
  return new Response(body, {
    headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=86400, stale-while-revalidate=604800" },
  });
}
