/**
 * Shared Mapbox map configuration and token validation. Pure (no React).
 *
 * The 3D Boston basemap (Mapbox Standard with 3D buildings + trees and the
 * cream/navy color scheme) is adapted from the CoolBoston project
 * (https://github.com/lvcasmadeit/coolboston, web/lib/mapConfig.ts).
 */

/** Mapbox Standard basemap: the one that ships real 3D building models. */
export const MAP_STYLE = "mapbox://styles/mapbox/standard";

/**
 * Standard config knobs. `show3dBuildings`/`show3dTrees` are what make the map
 * three-dimensional; the colors give Boston the CoolBoston cream-land / navy-water look.
 */
export const BASEMAP_CONFIG = {
  theme: "default",
  lightPreset: "day",
  colorLand: "#F4EFE4",
  colorWater: "#1F4A6E",
  colorGreenspace: "#D7E3C8",
  colorBuildings: "#F3F2EF",
  showPointOfInterestLabels: false,
  showLandmarkIcons: false,
  show3dBuildings: true,
  // Off: the procedural-facades tileset 404s on some downtown tiles (one console
  // error per tile); towers still get Standard's glassy 3D models.
  show3dFacades: false,
  show3dTrees: true,
} as const;

/**
 * Opening camera: a pitched 3D view over central Boston / Cambridge.
 *
 * Standard hides its 3D buildings below zoom ~15, so to actually see the 3D
 * cityscape the camera has to be zoomed in. The bike network spans several
 * towns, so this is a close "hero" view over downtown; users can zoom out to
 * see the whole cluster (buildings flatten to footprints when they do).
 */
/**
 * Opening camera: a pitched 3D view centered on Northeastern University, the middle of
 * the NU-area neighborhood cluster the bike model is scoped to.
 *
 * Standard hides its 3D buildings below zoom ~15, so the close view shows the 3D cityscape;
 * zoom out (the toggle) to see all the neighborhoods at once.
 */
export const BOSTON_VIEW = {
  longitude: -71.0892,
  latitude: 42.3398,
  zoom: 14.6,
  pitch: 55,
  bearing: -17.5,
} as const;

/** A wider view that frames the whole NU-area neighborhood cluster (buildings render flat here). */
export const BOSTON_OVERVIEW = {
  longitude: -71.0892,
  latitude: 42.3250,
  zoom: 12.6,
  pitch: 40,
  bearing: -17.5,
} as const;

/** Northeastern University main campus — the anchor the bike cluster is built around. */
export const NORTHEASTERN = {
  longitude: -71.0892,
  latitude: 42.3398,
  label: "Northeastern University",
} as const;

export type TokenCheck =
  | { ok: true; token: string }
  | { ok: false; reason: string };

/**
 * Validate a Mapbox access token read from the environment.
 * Public Mapbox tokens start with "pk." and are safe to ship to the browser;
 * a secret "sk." token must never go in a NEXT_PUBLIC_ var.
 */
export function checkMapboxToken(token: string | undefined): TokenCheck {
  if (!token || token.trim() === "") {
    return {
      ok: false,
      reason:
        "Map unavailable: set NEXT_PUBLIC_MAPBOX_TOKEN in web/.env.local to a Mapbox public token (pk...).",
    };
  }
  if (!token.startsWith("pk.")) {
    return {
      ok: false,
      reason:
        "Map unavailable: NEXT_PUBLIC_MAPBOX_TOKEN must be a public token (starts with \"pk.\"). Never put a secret (sk.) token in the browser.",
    };
  }
  return { ok: true, token };
}
