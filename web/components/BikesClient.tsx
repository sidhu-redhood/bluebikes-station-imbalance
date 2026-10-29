"use client";

import dynamic from "next/dynamic";
import { checkMapboxToken } from "@/lib/mapConfig";

// Same pattern as MapClient: mapbox-gl needs `window`, so the map is client-only.
const BikesMapView = dynamic(() => import("@/components/BikesMapView"), {
  ssr: false,
  loading: () => <p className="p-6 text-sm text-muted-foreground">Loading map…</p>,
});

export function BikesClient() {
  const check = checkMapboxToken(process.env.NEXT_PUBLIC_MAPBOX_TOKEN);
  if (!check.ok) return <p role="alert" className="p-6 text-sm text-muted-foreground">{check.reason}</p>;
  return <BikesMapView token={check.token} />;
}
