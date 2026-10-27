# Capability Map: Square grid and composite Heat Score

Status: APPROVED 2026-10-24. Module specs: SPEC-square-grid.md, SPEC-heat-score.md (drafts).

| Module id | Responsibility | Depends on |
|---|---|---|
| square-grid | Replace the H3 res-10 hexes with a square grid: pipeline builds cells, joins Heat Watch air and Landsat surface onto them, writes `cells.json` (schema v4) with geometry. Web renders cells as a flat Mapbox `fill` layer under the 3D buildings (replacing the deck.gl hex slab). Buildings and areas map to cells by footprint overlap rather than a single centroid cell. | — |
| heat-score | Composite Heat Score (0–100): config module (weights, normalization anchors, categories, colour stops), pure scoring function, score layer and legend replacing the Air/Surface toggle, hover/select readout, before/after ΔH in the cooling model, dashboard cards, area statistics, charts re-based on ΔH. Air and surface stay available in detail views. | square-grid |

Build order: square-grid → heat-score

Why two modules: each ships and is verified on its own. The grid can go live with today's air/surface colouring; the score can be checked against fixed inputs without caring about cell shape. Scoring depends on the grid (not the reverse) because the score layer is a Mapbox fill over square cells, and building the fill layer twice would be waste.

Files once approved: `.kiro/specs/SPEC-square-grid.md`, `.kiro/specs/SPEC-heat-score.md`.

## Facts from the current data (checked 2026-10-24)

- Air: Heat Watch 3pm, July 2019, 682 source hexes of ~25 ha, area-weighted onto 8,044 H3 cells (~1.5 ha). Range 87.8–94.5 °F. Smaller tiles do not add air detail: every tile inside one source hex gets the same value.
- Surface: Landsat 7/8, 9 scenes, June–August 2019, acquired ~11:15 EDT (mid-morning), 30 m pixels. Range 46–131 °F, median 95 °F. Smaller tiles do add surface detail.
- Time mismatch: air is a 3pm July traverse; surface is a mid-morning summer median. The score must be labelled as combining non-contemporaneous inputs.
- With the given anchors (air 70–105, surface 75–150) today's cells score 31–67, with 2nd–98th percentile at 34–56. Nearly the whole city reads Moderate/High and the colour ramp uses about a quarter of its range.

## Decisions so far

- Tile size: 100 m squares (~12,000 cells). One static JSON file still works; no vector tiles.
- Normalization: fitted to Boston's 2019 data (2nd–98th percentile per input, ≈ air 88.9–92.9 °F, surface 74.8–108.2 °F), stored in config. Score means "relative to Boston, summer 2019", not comparable across years or cities.
- Effect spreading: a building's air effect spreads over its 500 m neighbourhood (5 × 5 tiles ≈ 25 ha, the Heat Watch resolution), share = treated area ÷ neighbourhood area. Surface effect stays on the tiles the footprint overlaps, share = overlap ÷ tile land area. Documented as a modelling assumption, not validated.

---

# Capability Map: Bluebikes station imbalance (FAI project)

Status: DRAFT 2026-10-27. Module specs: SPEC-bike-markov.md.

| Module id | Responsibility | Depends on |
|---|---|---|
| bike-markov | Per-station birth–death Markov chain from weekday hourly trip rates; 10-minute poll matrices; forecasts and run-out / fill-up probabilities; exact day simulation; `bikes.json` v1; `/bikes` map page; standalone demo. | — (reuses map config and land grid) |
| bike-hmm (planned) | Hidden counts from trip history (forward–backward, Viterbi); censored-demand correction; validation against GBFS polls. | bike-markov |
| bike-policy (planned) | MDP over vans, incentives and app nudges; simulator from bike-markov; baselines. | bike-markov, bike-hmm |

Build order: bike-markov → bike-hmm → bike-policy
