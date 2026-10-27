/** NU-area neighborhood colors and the client-side geojson URL. Pure (no React). */

export const NEIGHBORHOODS_URL = "/api/neighborhoods";

type Rgb = [number, number, number];

/** Palette echoing the "Neighborhoods Near NU" reference map, keyed by lowercased name. */
export const NEIGHBORHOOD_COLORS: Record<string, Rgb> = {
  fenway: [120, 170, 110], // green (Kenmore is part of Fenway here)
  "back bay": [90, 130, 200], // blue
  "mission hill": [220, 130, 70], // orange
  "south end": [210, 90, 110], // pink/red
  roxbury: [160, 110, 200], // purple
  "jamaica plain": [90, 180, 180], // teal
};

const FALLBACK: Rgb = [150, 150, 150];

export function neighborhoodColor(name: string | undefined): Rgb {
  if (!name) return FALLBACK;
  return NEIGHBORHOOD_COLORS[name.trim().toLowerCase()] ?? FALLBACK;
}

/** Title-case a lowercased neighborhood key for display ("back bay" -> "Back Bay"). */
export function neighborhoodLabel(name: string | undefined): string {
  if (!name) return "";
  return name
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export interface NeighborhoodFeature {
  type: "Feature";
  properties: { name: string };
  geometry: { type: "Polygon" | "MultiPolygon"; coordinates: number[][][] | number[][][][] };
}

export interface NeighborhoodCollection {
  type: "FeatureCollection";
  features: NeighborhoodFeature[];
}

/** A place button: label plus the camera target to fly to. Centroids come from the
 *  station data; Kenmore is its own spot inside the Fenway polygon. */
export interface PlaceTarget {
  key: string; // matches a station's `neighborhood` tag (for filtering/highlight)
  label: string;
  longitude: number;
  latitude: number;
  zoom: number;
}

export const PLACES: PlaceTarget[] = [
  { key: "fenway", label: "Kenmore", longitude: -71.0956, latitude: 42.3489, zoom: 15.4 },
  { key: "fenway", label: "Fenway", longitude: -71.0967, latitude: 42.3458, zoom: 15 },
  { key: "back bay", label: "Back Bay", longitude: -71.0793, latitude: 42.3489, zoom: 15.2 },
  { key: "mission hill", label: "Mission Hill", longitude: -71.1025, latitude: 42.3338, zoom: 15.4 },
  { key: "south end", label: "South End", longitude: -71.0705, latitude: 42.3423, zoom: 15 },
  { key: "roxbury", label: "Roxbury", longitude: -71.0857, latitude: 42.326, zoom: 14.6 },
  { key: "jamaica plain", label: "Jamaica Plain", longitude: -71.1112, latitude: 42.3115, zoom: 14.6 },
];
