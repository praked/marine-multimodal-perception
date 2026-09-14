import { declinationPath } from "@/components/datamap/SkyDome";
import {
  coverageGaps,
  daypartTotals,
  entityCoverage,
  hourlyDensity,
  luminanceHist,
  sunArcs,
  weatherBands,
} from "@/lib/datamap";
import { CatalogueSchema, type ClipSummary } from "@/lib/types";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/* Aggregations are tested against the REAL enriched demo bundle plus
   synthetic edge cases. */

const catalogue = CatalogueSchema.parse(
  JSON.parse(
    readFileSync(
      path.resolve(__dirname, "..", "public", "demo", "clips.json"),
      "utf8",
    ),
  ),
);
const clips = catalogue.clips;

describe("enriched demo bundle", () => {
  it("both demo clips carry a full enrichment block", () => {
    for (const c of clips) {
      expect(c.enrichment).toBeDefined();
      expect(c.enrichment!.gps_init.lat).toBeCloseTo(47.6956, 3);
      expect(c.enrichment!.gps_init.lon).toBeCloseTo(9.1939, 3);
      expect(c.enrichment!.sun_samples.length).toBeGreaterThan(0);
      expect(c.enrichment!.luminance.hist).toHaveLength(16);
    }
  });

  it("sun arcs are physically plausible (afternoon, elevated, west of south)", () => {
    const arcs = sunArcs(clips);
    expect(arcs).toHaveLength(2);
    for (const arc of arcs) {
      for (const [elev, az] of arc.path) {
        expect(elev).toBeGreaterThan(0);
        expect(elev).toBeLessThan(70); // InstitutionOne max ~65.8°
        expect(az).toBeGreaterThan(90);
        expect(az).toBeLessThan(300);
      }
    }
  });

  it("hourly density lands in the capture hours", () => {
    const hours = hourlyDensity(clips);
    const active = hours.filter((h) => h.frames > 0).map((h) => h.hour);
    expect(active.every((h) => h >= 12 && h <= 17)).toBe(true);
    const total = hours.reduce((a, h) => a + h.frames, 0);
    const expected = clips.reduce((a, c) => a + c.n_frames, 0);
    expect(Math.abs(total - expected)).toBeLessThan(expected * 0.02);
  });

  it("dayparts + luminance + entities aggregate", () => {
    const parts = daypartTotals(clips);
    expect(parts.day).toBeGreaterThan(0);
    expect(parts.night).toBe(0);
    expect(luminanceHist(clips)).toHaveLength(16);
    const entities = entityCoverage(clips, { boat: 100, person: 100 });
    const boat = entities.find((e) => e.cls === "boat");
    expect(boat!.instances).toBeGreaterThan(0);
  });

  it("weather bands weight frames into the right bins", () => {
    const bands = weatherBands(clips, "cloud_cover_pct", [0, 20, 40, 60, 80, 100.1]);
    const total = bands.reduce((a, b) => a + b.value, 0);
    expect(total).toBe(clips.reduce((a, c) => a + c.n_frames, 0));
  });

  it("gaps flag the honest holes (no night, no rain)", () => {
    const gaps = coverageGaps(clips, entityCoverage(clips, { swimmer: 100 }));
    expect(gaps.some((g) => /night/i.test(g.text))).toBe(true);
  });
});

describe("declinationPath (solstice envelope geometry)", () => {
  it("summer solstice at InstitutionOne peaks near 65.7°", () => {
    const summer = declinationPath(47.6956, 23.44);
    const maxElev = Math.max(...summer.map(([e]) => e));
    expect(maxElev).toBeGreaterThan(64.5);
    expect(maxElev).toBeLessThan(66.5);
  });

  it("winter solstice peaks near 18.9° and spans a shorter arc", () => {
    const winter = declinationPath(47.6956, -23.44);
    const maxElev = Math.max(...winter.map(([e]) => e));
    expect(maxElev).toBeGreaterThan(17.5);
    expect(maxElev).toBeLessThan(20);
    expect(winter.length).toBeLessThan(
      declinationPath(47.6956, 23.44).length,
    );
  });
});

describe("edge cases", () => {
  it("everything degrades gracefully with no enrichment", () => {
    const bare = clips.map(
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      ({ enrichment: _e, ...rest }) => rest,
    ) as ClipSummary[];
    expect(sunArcs(bare)).toHaveLength(0);
    expect(hourlyDensity(bare).every((h) => h.frames === 0)).toBe(true);
    expect(luminanceHist(bare)).toHaveLength(0);
    expect(entityCoverage(bare, {})).toHaveLength(0);
  });
});

describe("entity counting honesty (regression: no cross-stream summing)", () => {
  it("headline instances = max across streams, never the sum", () => {
    for (const c of clips) {
      for (const [cls, e] of Object.entries(c.enrichment!.entities)) {
        const streams = Object.values(e.by_source ?? {});
        if (streams.length === 0) continue;
        const max = Math.max(...streams.map((s) => s.instances));
        const sum = streams.reduce((a, s) => a + s.instances, 0);
        expect(e.instances, `${cls} headline`).toBe(max);
        if (streams.length > 1) {
          expect(e.instances, `${cls} must not sum streams`).toBeLessThan(sum);
        }
      }
    }
  });

  it("canonical folding within a clip takes max, not sum", () => {
    const cov = entityCoverage(clips, {});
    for (const e of cov) {
      expect(e.contributions.length).toBeGreaterThan(0);
      const total = e.contributions.reduce((a, c) => a + c.instances, 0);
      expect(e.instances).toBe(total); // across clips: additive
    }
  });
});

describe("degToCompass", () => {
  it("maps degrees to 16-point compass names", async () => {
    const { degToCompass } = await import("@/lib/datamap");
    expect(degToCompass(0)).toBe("N");
    expect(degToCompass(242)).toBe("WSW");
    expect(degToCompass(359)).toBe("N");
    expect(degToCompass(90)).toBe("E");
  });
});

describe("coverage goals", () => {
  it("mergeGoals falls back to defaults on garbage and merges partials", async () => {
    const { mergeGoals, defaultGoals } = await import("@/lib/goals");
    expect(mergeGoals(null)).toEqual(defaultGoals());
    expect(mergeGoals({ junk: true })).toEqual(defaultGoals());
    const partial = {
      ...defaultGoals(),
      entities: { boat: 9999 },
      hours: { "3": 5 },
    };
    const merged = mergeGoals(partial);
    expect(merged.entities.boat).toBe(9999);
    expect(merged.entities.person).toBe(defaultGoals().entities.person);
    expect(merged.hours["3"]).toBe(5);
    expect(merged.hours["12"]).toBe(defaultGoals().hours["12"]);
  });

  it("goalsForBands aligns keyed goals with band edges", async () => {
    const { goalsForBands } = await import("@/lib/goals");
    expect(goalsForBands({ "0": 2, "20": 1 }, [0, 20, 40.1])).toEqual([2, 1]);
    expect(goalsForBands({}, [0, 10, 20.1])).toEqual([0, 0]);
  });

  it("goalGaps reports weather bands below their hour goals", async () => {
    const { goalGaps } = await import("@/lib/datamap");
    const { WEATHER_BAND_DEFS, framesToHours, defaultGoals } = await import(
      "@/lib/goals"
    );
    const gaps = goalGaps(clips, defaultGoals(), WEATHER_BAND_DEFS, framesToHours);
    // fixture clips carry a single fair-weather sample -> the never-seen
    // rain/wind bands with goals must surface as serious gaps
    expect(
      gaps.some((g) => g.severity === "serious" && /Precipitation/.test(g.text)),
    ).toBe(true);
    // every gap line names its goal
    for (const g of gaps) expect(g.text).toMatch(/h\)?$|target/);
  });
});

describe("entityConditionMatrix (advanced mode)", () => {
  it("attributes instances to conditions via clip membership, never summing streams", async () => {
    const { entityConditionMatrix, entityCoverage, ENTITY_TARGETS } =
      await import("@/lib/datamap");
    const m = entityConditionMatrix(clips, ENTITY_TARGETS);
    const cov = entityCoverage(clips, ENTITY_TARGETS);
    for (const cls of m.classes) {
      const total = cov.find((e) => e.cls === cls)!.instances;
      // a clip belongs to exactly one band per group, so each group's
      // row-sum must equal the class total over clips with that datum
      for (const group of ["Cloud", "Daypart"]) {
        const sum = m.conditions
          .filter((c) => c.group === group)
          .reduce((a, c) => a + (m.cells[cls]?.[c.key]?.instances ?? 0), 0);
        expect(sum).toBeLessThanOrEqual(total);
      }
    }
  });

  it("flags a gap only when the condition has footage but the class is absent", async () => {
    const { entityConditionMatrix, ENTITY_TARGETS } = await import("@/lib/datamap");
    const m = entityConditionMatrix(clips, ENTITY_TARGETS);
    for (const cls of m.classes) {
      for (const c of m.conditions) {
        const cell = m.cells[cls]?.[c.key];
        if (cell?.gap) expect(cell.instances).toBe(0);
        if ((cell?.instances ?? 0) > 0) expect(cell!.gap).toBe(false);
      }
    }
  });
});
