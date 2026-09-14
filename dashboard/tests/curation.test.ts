import {
  applyCuration,
  applyCuts,
  applyDeleted,
  createLocalCurationBackend,
  cutRuns,
  cutStats,
  daysRemaining,
  frameKept,
  inCut,
  isDeleted,
  keptFrames,
  nextKeptIndex,
  normaliseCuts,
  purgeCandidates,
  type CurationRecord,
} from "@/lib/curation";
import type { ClipSummary } from "@/lib/types";
import { beforeEach, describe, expect, it } from "vitest";

function rec(over: Partial<CurationRecord> = {}): CurationRecord {
  return {
    clip_key: "scene__2026-08-26_16-28-59",
    deleted_at: null,
    restored_at: null,
    purged_at: null,
    cuts: [],
    note: null,
    updated_at: "2026-08-28T10:00:00Z",
    updated_by: "test",
    ...over,
  };
}

const frames = Array.from({ length: 12 }, (_, i) => ({
  ts: `16:29:${String(i).padStart(2, "0")}.0`,
}));

describe("curation: deleted semantics", () => {
  it("deleted_at without a later restore is deleted; restore flips it", () => {
    expect(isDeleted(rec())).toBe(false);
    expect(isDeleted(rec({ deleted_at: "2026-08-28T10:00:00Z" }))).toBe(true);
    expect(isDeleted(rec({ deleted_at: "2026-08-28T10:00:00Z", restored_at: "2026-08-28T11:00:00Z" }))).toBe(false);
    // deleted again after a restore
    expect(isDeleted(rec({ deleted_at: "2026-08-28T12:00:00Z", restored_at: "2026-08-28T11:00:00Z" }))).toBe(true);
    // purged is final
    expect(isDeleted(rec({ deleted_at: "2026-07-01T00:00:00Z", restored_at: "2026-07-02T00:00:00Z", purged_at: "2026-08-01T00:00:00Z" }))).toBe(true);
  });

  it("applyDeleted / applyCuts are the only transitions", () => {
    const d = applyDeleted(rec(), true, "2026-08-28T10:00:00Z", "authortwo");
    expect(isDeleted(d)).toBe(true);
    expect(d.updated_by).toBe("authortwo");
    const r = applyDeleted(d, false, "2026-08-28T10:05:00Z", "authortwo");
    expect(isDeleted(r)).toBe(false);
    expect(r.deleted_at).toBe("2026-08-28T10:00:00Z"); // history kept
    const c = applyCuts(r, [{ start_ts: "16:29:05.0", end_ts: "16:29:02.0" }], "2026-08-28T10:06:00Z", "authortwo");
    expect(c.cuts).toEqual([{ start_ts: "16:29:02.0", end_ts: "16:29:05.0" }]); // normalised
  });

  it("splits a catalogue into active + deleted by clip key", () => {
    const clips = ["a__1", "b__2", "c__3"].map((k) => ({
      clip_id: k.replace("__", "/"),
    })) as ClipSummary[];
    const records = new Map<string, CurationRecord>([
      ["b__2", rec({ clip_key: "b__2", deleted_at: "2026-08-28T10:00:00Z" })],
      ["c__3", rec({ clip_key: "c__3", cuts: [{ start_ts: "1:0:0.0", end_ts: "1:0:1.0" }] })],
    ]);
    const { active, deleted } = applyCuration(clips, records);
    expect(active.map((c) => c.clip_id)).toEqual(["a/1", "c/3"]);
    expect(deleted.map((d) => d.clip.clip_id)).toEqual(["b/2"]);
  });

  it("retention: days remaining + purge candidates", () => {
    const now = Date.parse("2026-09-10T12:00:00Z");
    expect(daysRemaining("2026-09-01T00:00:00Z", now)).toBe(21);
    expect(daysRemaining("2026-08-01T00:00:00Z", now)).toBe(0);
    const old = rec({ clip_key: "old", deleted_at: "2026-08-01T00:00:00Z" });
    const fresh = rec({ clip_key: "fresh", deleted_at: "2026-09-05T00:00:00Z" });
    const restored = rec({ clip_key: "back", deleted_at: "2026-08-01T00:00:00Z", restored_at: "2026-08-02T00:00:00Z" });
    const purged = rec({ clip_key: "gone", deleted_at: "2026-07-01T00:00:00Z", purged_at: "2026-08-01T00:00:00Z" });
    expect(purgeCandidates([old, fresh, restored, purged], now).map((r) => r.clip_key)).toEqual(["old"]);
  });
});

describe("curation: cuts", () => {
  const cuts = [
    { start_ts: "16:29:03.0", end_ts: "16:29:05.0" },
    { start_ts: "16:29:09.0", end_ts: "16:29:09.0", note: "single frame" },
  ];

  it("inCut is inclusive both ends and tolerates safe_ts dashes", () => {
    expect(inCut(cuts, "16:29:02.9")).toBe(false);
    expect(inCut(cuts, "16:29:03.0")).toBe(true);
    expect(inCut(cuts, "16:29:05.0")).toBe(true);
    expect(inCut(cuts, "16:29:05.1")).toBe(false);
    expect(inCut(cuts, "16-29-09.0")).toBe(true);
    expect(inCut([], "16:29:09.0")).toBe(false);
  });

  it("normaliseCuts sorts, swaps reversed ends and drops garbage", () => {
    const out = normaliseCuts([
      { start_ts: "16:29:09.0", end_ts: "16:29:07.0" },
      { start_ts: "nope", end_ts: "16:29:07.0" },
      { start_ts: "16:29:01.0", end_ts: "16:29:02.0" },
    ]);
    expect(out).toEqual([
      { start_ts: "16:29:01.0", end_ts: "16:29:02.0" },
      { start_ts: "16:29:07.0", end_ts: "16:29:09.0" },
    ]);
  });

  it("frameKept / keptFrames honour deletion and cuts", () => {
    const r = rec({ cuts });
    expect(frameKept(r, "16:29:04.0")).toBe(false);
    expect(frameKept(r, "16:29:06.0")).toBe(true);
    expect(frameKept(undefined, "16:29:04.0")).toBe(true);
    expect(keptFrames(frames, r).map((f) => f.ts)).toEqual(
      frames.filter((f) => !["16:29:03.0", "16:29:04.0", "16:29:05.0", "16:29:09.0"].includes(f.ts)).map((f) => f.ts),
    );
    expect(keptFrames(frames, rec({ deleted_at: "2026-08-28T10:00:00Z" }))).toEqual([]);
  });

  it("cutStats counts frames and seconds (gap-capped)", () => {
    const s = cutStats(frames, cuts);
    expect(s.cut).toBe(4);
    expect(s.kept).toBe(8);
    // 1 s per frame here (capped at 2.5), last frame nominal 1/3 s
    expect(s.cutSeconds).toBeCloseTo(4, 5);
    expect(s.keptSeconds).toBeCloseTo(7 + 1 / 3, 5);
  });

  it("nextKeptIndex skips cut runs in both directions, with optional wrap", () => {
    expect(nextKeptIndex(frames, cuts, 3, 1)).toBe(6);
    expect(nextKeptIndex(frames, cuts, 5, -1)).toBe(2);
    expect(nextKeptIndex(frames, cuts, 9, 1)).toBe(10);
    expect(nextKeptIndex(frames, cuts, 2, 1)).toBe(2);
    const allCut = [{ start_ts: "16:29:00.0", end_ts: "16:29:11.0" }];
    expect(nextKeptIndex(frames, allCut, 0, 1)).toBe(-1);
    const tailCut = [{ start_ts: "16:29:10.0", end_ts: "16:29:11.0" }];
    expect(nextKeptIndex(frames, tailCut, 10, 1)).toBe(-1);
    expect(nextKeptIndex(frames, tailCut, 10, 1, true)).toBe(0);
  });

  it("cutRuns compresses cut frames into timeline fractions", () => {
    expect(cutRuns(frames, cuts)).toEqual([
      { start: 3 / 12, width: 3 / 12 },
      { start: 9 / 12, width: 1 / 12 },
    ]);
    expect(cutRuns(frames, [])).toEqual([]);
  });
});

describe("local curation backend (demo mode)", () => {
  beforeEach(() => window.localStorage.clear());

  it("round-trips delete / restore / cuts through localStorage", async () => {
    const b = createLocalCurationBackend();
    await b.setDeleted("x__1", true);
    expect(isDeleted((await b.list()).get("x__1"))).toBe(true);
    await b.setDeleted("x__1", false);
    expect(isDeleted((await b.list()).get("x__1"))).toBe(false);
    await b.setCuts("x__1", [{ start_ts: "10:00:01.0", end_ts: "10:00:00.0" }]);
    expect((await b.list()).get("x__1")!.cuts).toEqual([{ start_ts: "10:00:00.0", end_ts: "10:00:01.0" }]);
    window.localStorage.setItem("asvproject.curation.v1", "garbage");
    expect((await b.list()).size).toBe(0);
  });
});

describe("data-link export filter", () => {
  it("drops per-frame objects inside cuts, keeps JSON side files and other sets", async () => {
    const { filterCutObjects } = await import("@/lib/curation/exportFilter");
    const objects = [
      { key: "a__1/meta.json" },
      { key: "a__1/frames/ts=10-00-00.0.jpg" },
      { key: "a__1/frames/ts=10-00-01.0.jpg" },
      { key: "a__1/thermal/ts=10-00-01.0.jpg" },
      { key: "a__1/seg/ts=10-00-01.0.png" },
      { key: "b__2/frames/ts=10-00-01.0.jpg" },
    ];
    const records = new Map([["a__1", rec({ clip_key: "a__1", cuts: [{ start_ts: "10:00:01.0", end_ts: "10:00:02.0" }] })]]);
    const { kept, cut } = filterCutObjects(objects, records);
    expect(cut).toBe(3);
    expect(kept.map((o) => o.key)).toEqual([
      "a__1/meta.json", "a__1/frames/ts=10-00-00.0.jpg", "b__2/frames/ts=10-00-01.0.jpg",
    ]);
  });
});
