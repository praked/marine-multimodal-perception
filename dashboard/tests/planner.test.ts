import { describe, expect, it } from "vitest";
import {
  candidatesForClip,
  conditionCells,
  evalExpr,
  selectPlan,
  type ExprRow,
  type PlanCandidate,
} from "@/lib/annotate/planner";
import type { ClipSummary } from "@/lib/types";

function cand(over: Partial<PlanCandidate>): PlanCandidate {
  return {
    clipKey: "s__t", clipTitle: "t", ts: "12:00:00.0", idx: 0, u: 0.5,
    classes: [], cells: [], labelled: true, ...over,
  };
}

describe("annotation planner", () => {
  it("is deterministic: same inputs, identical plan", () => {
    const cands = Array.from({ length: 60 }, (_, i) =>
      cand({ ts: `12:00:${String(i).padStart(2, "0")}.0`, idx: i * 10,
             u: (i * 7919) % 100 / 100,
             classes: i % 3 === 0 ? ["person"] : ["boat"] }));
    const a = selectPlan(cands, 20);
    const b = selectPlan([...cands].reverse(), 20);
    expect(a.frames.map((f) => f.ts)).toEqual(b.frames.map((f) => f.ts));
  });

  it("clamps N to what exists, never pads", () => {
    const cands = [cand({}), cand({ ts: "12:00:09.0", idx: 20 })];
    const p = selectPlan(cands, 500);
    expect(p.frames).toHaveLength(2);
    expect(p.clamped).toBe(true);
    expect(p.requested).toBe(500);
  });

  it("enforces temporal spacing within a clip", () => {
    const cands = Array.from({ length: 10 }, (_, i) =>
      cand({ ts: `12:00:0${i}.0`, idx: i, u: 1 - i * 0.01 }));
    const p = selectPlan(cands, 10);
    // all 10 are within 8 frames of each other -> only one survives
    expect(p.frames).toHaveLength(2); // idx 0 and idx 9 (spacing >= 8)
  });

  it("rarity weighting pulls scarce classes in", () => {
    const cands = [
      ...Array.from({ length: 30 }, (_, i) =>
        cand({ ts: `12:0${Math.floor(i / 6)}:0${i % 6}.0`, idx: i * 10,
               u: 0.9, classes: ["boat"] })),
      cand({ ts: "13:00:00.0", idx: 900, u: 0.1, classes: ["duck"] }),
    ];
    const p = selectPlan(cands, 5);
    expect(p.frames.some((f) => f.classes.includes("duck"))).toBe(true);
  });

  it("evalExpr: NOT/AND/OR/XOR left-fold", () => {
    const rows = (r: [string, string, string, boolean][]): ExprRow[] =>
      r.map(([op, kind, value, negate]) => ({
        op: op as ExprRow["op"],
        pred: { kind: kind as "class" | "condition", value, negate },
      }));
    // person AND NOT precip:wet
    const e1 = rows([["AND", "class", "person", false],
                     ["AND", "condition", "precip:wet", true]]);
    expect(evalExpr(e1, ["person"], ["precip:dry"])).toBe(true);
    expect(evalExpr(e1, ["person"], ["precip:wet"])).toBe(false);
    // duck XOR person
    const e2 = rows([["AND", "class", "duck", false],
                     ["XOR", "class", "person", false]]);
    expect(evalExpr(e2, ["duck"], [])).toBe(true);
    expect(evalExpr(e2, ["duck", "person"], [])).toBe(false);
    expect(evalExpr(e2, [], [])).toBe(false);
    // empty expression selects everything
    expect(evalExpr([], [], [])).toBe(true);
  });

  it("conditionCells mirrors the python selector's derivation", () => {
    const clip = {
      scene: "s", triplet_ts: "t", title: "t", clip_id: "s/t", n_frames: 1,
      enrichment: {
        dayparts: { day: 100, golden: 5 },
        weather: { cloud_cover_pct: 45, wind_speed_kmh: 12,
                   precipitation_mm: 0.0, temperature_c: 20,
                   wind_dir_deg: 180 },
        luminance: { mean: 130, hist: [] },
        gps_init: { lat: 0, lon: 0, source: "s" },
        sun_samples: [], entities: {},
      },
    } as unknown as ClipSummary;
    expect(conditionCells(clip)).toEqual(
      ["part:day", "cloud:2", "wind:1", "precip:dry", "luma:1"]);
  });

  it("candidatesForClip: uncertainty favours label/scorer conflict", () => {
    const clip = { scene: "s", triplet_ts: "t", title: "t", clip_id: "s/t",
                   n_frames: 2, enrichment: null } as unknown as ClipSummary;
    const sectors = {
      "12:00:00.0": { p_obstacle: [0.9, 0.9], bin_centers_deg: [-15, 15] },
      "12:00:00.3": { p_obstacle: [0.9, 0.9], bin_centers_deg: [-15, 15] },
    } as never;
    const labels = {
      "12:00:00.0": { obstacle_bins_fisheye: [-15, 15], fisheye_bboxes: [] },
      // second frame: labels say EMPTY but scorer says 0.9 -> conflict
      "12:00:00.3": { obstacle_bins_fisheye: [], fisheye_bboxes: [] },
    } as never;
    const c = candidatesForClip(clip, sectors, labels);
    expect(c[1]!.u).toBeGreaterThan(c[0]!.u);
  });
});

describe("centroid round-trip", () => {
  it("BoxSchema accepts an optional centroid and JSONL export carries it", async () => {
    const { BoxSchema } = await import("@/lib/types");
    const withC = BoxSchema.parse({ cls: "boat", xyxy: [0.1, 0.1, 0.4, 0.4],
                                    centroid: [0.2, 0.3] });
    expect(withC.centroid).toEqual([0.2, 0.3]);
    expect(BoxSchema.parse({ cls: "boat", xyxy: [0, 0, 1, 1] }).centroid)
      .toBeUndefined();
    const { auditsToJsonl } = await import("@/lib/audit/backend");
    const jsonl = auditsToJsonl(
      [{
        id: "1", clip_key: "s__t", frame_ts: "12:00:00.0",
        frame_id: "s/t/ts=12-00-00.0", verdict: "edit",
        boxes: [withC, { cls: "duck", xyxy: [0.5, 0.5, 0.6, 0.6] }],
        source: "dashboard-web", created_at: "2026-08-25T00:00:00Z",
      }],
      { width: 864, height: 648 },
    );
    const rec = JSON.parse(jsonl.trim());
    expect(rec.fisheye_bboxes[0].centroid).toEqual([0.2, 0.3]);
    expect(rec.fisheye_bboxes[1].centroid).toBeUndefined();
  });
});

describe("plan persistence", () => {
  it("storePlan/loadPlan round-trips frames, cursor and stats", async () => {
    const { storePlan, loadPlan, savePlanCursor, clearPlan } = await import(
      "@/lib/annotate/planner");
    const plan = selectPlan(
      [cand({ ts: "12:00:00.0", idx: 0 }), cand({ ts: "12:00:10.0", idx: 30 })],
      2);
    storePlan(plan);
    const back = loadPlan();
    expect(back).not.toBeNull();
    expect(back!.frames).toHaveLength(plan.frames.length);
    expect(back!.cursor).toBe(0);
    expect(back!.stats).toEqual(plan.stats);
    expect(back!.clamped).toBe(plan.clamped);
    savePlanCursor(1);
    expect(loadPlan()!.cursor).toBe(1);
    clearPlan();
    expect(loadPlan()).toBeNull();
  });
});

describe("planner honours curation cuts", () => {
  it("frames inside a cut are never candidates; idx stays the raw position", async () => {
    const { candidatesForClip } = await import("@/lib/annotate/planner");
    const clip = { scene: "s", triplet_ts: "t", title: "t", clip_id: "s/t" } as unknown as ClipSummary;
    const sectors = Object.fromEntries(
      Array.from({ length: 6 }, (_, i) => [
        `12:00:0${i}.0`,
        { p_obstacle: [0.5, 0.5], bin_centers_deg: [-7.5, 7.5] },
      ]),
    ) as never;
    const all = candidatesForClip(clip, sectors, {});
    expect(all).toHaveLength(6);
    const cut = candidatesForClip(clip, sectors, {}, [{ start_ts: "12:00:01.0", end_ts: "12:00:03.0" }]);
    expect(cut.map((c) => c.ts)).toEqual(["12:00:00.0", "12:00:04.0", "12:00:05.0"]);
    expect(cut.map((c) => c.idx)).toEqual([0, 4, 5]);
  });
});

describe("plan library (multi-plan, named)", () => {
  it("keeps multiple named plans, newest active; cursor follows the active plan", async () => {
    const { storePlan, loadPlan, loadLibrary, savePlanCursor, setActivePlan, deletePlan, renamePlan } =
      await import("@/lib/annotate/planner");
    window.localStorage.clear();
    const p1 = selectPlan([cand({ ts: "12:00:00.0" }), cand({ ts: "12:00:10.0", idx: 30 })], 2);
    const p2 = selectPlan([cand({ ts: "13:00:00.0" })], 1);
    const s1 = storePlan(p1, "night tranche");
    const s2 = storePlan(p2, "negatives");
    const lib = loadLibrary();
    expect(lib.plans.map((p) => p.name)).toEqual(["night tranche", "negatives"]);
    expect(lib.activeId).toBe(s2.id);
    expect(loadPlan()!.name).toBe("negatives");
    savePlanCursor(0); // clamps within the 1-frame active plan
    setActivePlan(s1.id!);
    expect(loadPlan()!.name).toBe("night tranche");
    savePlanCursor(1);
    expect(loadPlan()!.cursor).toBe(1);
    expect(loadLibrary().plans.find((p) => p.id === s2.id)!.cursor).toBe(0); // untouched
    renamePlan(s1.id!, "low light");
    expect(loadPlan()!.name).toBe("low light");
    deletePlan(s1.id!);
    expect(loadLibrary().plans).toHaveLength(1);
    expect(loadPlan()!.name).toBe("negatives"); // active falls back to the survivor
  });

  it("migrates a legacy single-plan slot with its cursor intact", async () => {
    const { loadLibrary, loadPlan } = await import("@/lib/annotate/planner");
    window.localStorage.clear();
    window.localStorage.setItem("asvproject.annotate.plan.v1", JSON.stringify({
      version: 1, createdAt: "2026-08-31T12:00:00Z", requested: 2, cursor: 1,
      frames: [{ clipKey: "a__b", ts: "12:00:00.0" }, { clipKey: "a__b", ts: "12:00:10.0" }],
    }));
    const lib = loadLibrary();
    expect(lib.plans).toHaveLength(1);
    expect(lib.activeId).toBe("legacy");
    expect(loadPlan()!.cursor).toBe(1);
    expect(window.localStorage.getItem("asvproject.annotate.plan.v1")).toBeNull();
  });
});

describe("shared-plan import", () => {
  it("adds new shared plans, follows shared names/deletions, keeps local cursors and local-only plans", async () => {
    const { storePlan, importSharedPlans, savePlanCursor, loadLibrary, adoptPlanId } =
      await import("@/lib/annotate/planner");
    window.localStorage.clear();
    const local = storePlan(selectPlan([cand({})], 1), "local only");           // base36 id
    const adopted = storePlan(selectPlan([cand({})], 1), "mine, shared");
    adoptPlanId(adopted.id!, "11111111-aaaa-bbbb-cccc-222222222222");
    savePlanCursor(0);
    const shared = [
      { id: "11111111-aaaa-bbbb-cccc-222222222222", name: "renamed remotely", requested: 1,
        frames: adopted.frames, clamped: false, created_at: "2026-09-01T00:00:00Z" },
      { id: "33333333-dddd-eeee-ffff-444444444444", name: "built elsewhere", requested: 2,
        frames: [{ clipKey: "x__y", ts: "01:00:00.0" }], clamped: false, created_at: "2026-09-01T01:00:00Z" },
    ];
    const lib = importSharedPlans(shared);
    expect(lib.plans.map((p) => p.name)).toEqual(["local only", "renamed remotely", "built elsewhere"]);
    // remote deletion drops the local copy of a shared plan, keeps local-only
    const lib2 = importSharedPlans([shared[1]!]);
    expect(lib2.plans.map((p) => p.name)).toEqual(["local only", "built elsewhere"]);
    expect(loadLibrary().plans.find((p) => p.name === "local only")!.id).toBe(local.id);
  });
});

describe("hitTestPrioritised (selected box wins)", () => {
  it("prefers the selected box's corner over another box's body/corner", async () => {
    const { hitTestPrioritised } = await import("@/lib/annotate/boxes");
    const a = { cls: "boat", xyxy: [0.1, 0.1, 0.4, 0.4] as [number, number, number, number] };
    const b = { cls: "boat", xyxy: [0.35, 0.35, 0.7, 0.7] as [number, number, number, number] };
    // click on a's bottom-right corner, inside b's body (b is topmost)
    const hit = hitTestPrioritised([a, b], 0, 0.4, 0.4, 0.02);
    expect(hit).toEqual({ index: 0, corner: "br" });
    // without selection the base scan applies (corners already have global
    // priority there, so a's exact corner wins even under b's body)
    const plain = hitTestPrioritised([a, b], null, 0.4, 0.4, 0.02);
    expect(plain).toEqual({ index: 0, corner: "br" });
  });
  it("selected box's body beats an overlapping neighbour", async () => {
    const { hitTestPrioritised } = await import("@/lib/annotate/boxes");
    const a = { cls: "boat", xyxy: [0.1, 0.1, 0.5, 0.5] as [number, number, number, number] };
    const b = { cls: "boat", xyxy: [0.2, 0.2, 0.6, 0.6] as [number, number, number, number] };
    expect(hitTestPrioritised([a, b], 0, 0.3, 0.3, 0.02)?.index).toBe(0);
    expect(hitTestPrioritised([a, b], null, 0.3, 0.3, 0.02)?.index).toBe(1);
  });
});

describe("boxesInMarquee", () => {
  it("selects boxes whose centre is inside; ignores mere overlap", async () => {
    const { boxesInMarquee } = await import("@/lib/annotate/boxes");
    const boxes = [
      { cls: "boat", xyxy: [0.10, 0.10, 0.20, 0.20] as [number, number, number, number] }, // centre .15
      { cls: "duck", xyxy: [0.18, 0.18, 0.30, 0.30] as [number, number, number, number] }, // centre .24
      { cls: "buoy", xyxy: [0.05, 0.05, 0.60, 0.60] as [number, number, number, number] }, // centre .325 (big overlapping)
    ];
    expect(boxesInMarquee(boxes, [0.08, 0.08, 0.26, 0.26])).toEqual([0, 1]);
    expect(boxesInMarquee(boxes, [0.3, 0.3, 0.4, 0.4])).toEqual([2]);
    // inverted drag direction normalises
    expect(boxesInMarquee(boxes, [0.26, 0.26, 0.08, 0.08])).toEqual([0, 1]);
  });
});
