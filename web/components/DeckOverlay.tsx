"use client";

import { MapboxOverlay, type MapboxOverlayProps } from "@deck.gl/mapbox";
import { useEffect } from "react";
import { useMap } from "react-map-gl/mapbox";

/**
 * Per-layer props understood by MapboxOverlay in interleaved mode. deck.gl supports
 * these at runtime but doesn't export the type, so pass it as a layer's extra-props
 * generic: `new ScatterplotLayer<Datum, InterleavedLayerProps>({ slot: "top", ... })`.
 */
export type InterleavedLayerProps = {
  /** Mapbox Standard slot. Layers in the same slot keep their style order, bottom < middle < top. */
  slot?: "bottom" | "middle" | "top";
};

type MapInstance = NonNullable<ReturnType<typeof useMap>["current"]> extends { getMap(): infer M } ? M : never;

// One overlay per mapbox-gl instance, kept for the map's lifetime. react-map-gl's
// useControl adds/removes the control on every mount, and under React StrictMode the
// two overlapping deck instances race to attach to the same WebGL context
// ("deck: WebGL context already attached"), leaving no deck layers rendered in dev.
// A WeakMap keyed on the map instance guarantees a single overlay instead.
const overlays = new WeakMap<MapInstance, MapboxOverlay>();

/** The deck overlay attached to `map`, if any. */
export function getDeckOverlay(map: MapInstance): MapboxOverlay | undefined {
  return overlays.get(map);
}

/**
 * deck.gl rendered inside the Mapbox WebGL context (interleaved), so deck layers share
 * the depth buffer with the basemap's 3D buildings. Layers are placed by slot.
 *
 * Adapted from the CoolBoston project (https://github.com/lvcasmadeit/coolboston,
 * web/components/DeckOverlay.tsx).
 */
export function DeckOverlay(props: Omit<MapboxOverlayProps, "interleaved">) {
  const { current: mapRef } = useMap();

  useEffect(() => {
    const map = mapRef?.getMap();
    if (!map) return;
    let overlay = overlays.get(map);
    if (!overlay) {
      overlay = new MapboxOverlay({ ...props, interleaved: true });
      overlays.set(map, overlay);
      map.addControl(overlay);
    } else {
      overlay.setProps({ ...props, interleaved: true });
    }
  });

  // Clear layers (but keep the overlay attached) when this component unmounts.
  useEffect(() => {
    const map = mapRef?.getMap();
    return () => {
      if (map) overlays.get(map)?.setProps({ layers: [] });
    };
  }, [mapRef]);

  return null;
}
