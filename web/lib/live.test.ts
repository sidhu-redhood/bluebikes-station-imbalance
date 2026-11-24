import { describe, expect, it } from "vitest";
import { bostonMinute, buildSnapshot } from "@/lib/live";

const info = { data: { stations: [
  { station_id: "uuid-1", short_name: "B1" },
  { station_id: "uuid-2", short_name: "B2" },
  { station_id: "uuid-3" },
] } };
const status = { last_updated: 1000, data: { stations: [
  { station_id: "uuid-1", num_bikes_available: 7, is_installed: 1, is_renting: 1 },
  { station_id: "uuid-2", num_bikes_available: 3, is_installed: 1, is_renting: 0 },
  { station_id: "uuid-3", num_bikes_available: 9 },
] } };

describe("live readings", () => {
  it("joins the feed to bikes.json ids through short_name", () => {
    expect(buildSnapshot(info, status, ["B1"]).readings).toEqual({ B1: 7 });
  });
  it("reads offline stations and unknown ids as null", () => {
    expect(buildSnapshot(info, status, ["B2", "B9"]).readings).toEqual({ B2: null, B9: null });
  });
  it("uses Boston time whatever the device zone is", () => {
    expect(bostonMinute(new Date("2026-08-05T12:30:00Z"))).toBe(8 * 60 + 30); // EDT = UTC-4
    expect(bostonMinute(new Date("2026-01-05T12:30:00Z"))).toBe(7 * 60 + 30); // EST = UTC-5
  });
});
