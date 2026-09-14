import {
  cropAssetPath,
  isGiant,
  jumpIndex,
  orderForReview,
  parsePredictions,
  predictionsAssetPath,
  queueOrder,
  firstUnlabelled,
  foldCropLabels,
  manifestAssetPath,
  overlayRect,
  parseManifest,
  type CropLabelRecord,
} from "@/lib/crops";
import { describe, expect, it } from "vitest";

const rec = (
  crop_id: string,
  label: CropLabelRecord["label"],
  created_at: string,
): CropLabelRecord => ({ id: crypto.randomUUID(), crop_id, label, created_at });

describe("crops manifest", () => {
  it("parses compact rows preserving order", () => {
    const rows = parseManifest([
      ["aa", "2026-08-26_17-06-04", "17:06:05.0", 120, 80],
      ["bb", "2026-08-26_17-06-04", "17:06:05.3", 40, 30],
    ]);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toEqual({
      crop_id: "aa",
      chunk: "2026-08-26_17-06-04",
      ts: "17:06:05.0",
      crop_w: 120,
      crop_h: 80,
      window: null,
      xyxy: null,
    });
  });

  it("parses extended rows (window + xyxy) and legacy 5-field rows", () => {
    const rows = parseManifest([
      ["aa", "c", "t", 130, 66, [85, 92, 215, 158], [0.1157, 0.1543, 0.2315, 0.2315]],
      ["bb", "c", "t", 40, 30],
    ]);
    expect(rows[0]!.window).toEqual([85, 92, 215, 158]);
    expect(rows[0]!.xyxy).toEqual([0.1157, 0.1543, 0.2315, 0.2315]);
    expect(rows[1]!.window).toBeNull();
    expect(rows[1]!.xyxy).toBeNull();
  });

  it("rejects malformed manifests", () => {
    expect(() => parseManifest({ not: "an array" })).toThrow();
    expect(() => parseManifest([["aa", "chunk"]])).toThrow();
  });

  it("asset paths pass the /api/assets validator shape", () => {
    // 2-4 safe segments ending in a known extension (lib/r2 isValidAssetPath)
    expect(cropAssetPath("2026-08-26", "abc123")).toBe(
      "swarm_crops/2026-08-26/crops/abc123.jpg",
    );
    expect(manifestAssetPath("2026-08-26")).toBe(
      "swarm_crops/2026-08-26/manifest.json",
    );
  });
});

describe("crop label folding", () => {
  it("latest label per crop wins", () => {
    const folded = foldCropLabels([
      rec("aa", "accept", "2026-09-03T10:00:00Z"),
      rec("aa", "deny", "2026-09-03T10:01:00Z"),
      rec("bb", "skip", "2026-09-03T09:00:00Z"),
    ]);
    expect(folded.get("aa")?.label).toBe("deny");
    expect(folded.get("bb")?.label).toBe("skip");
  });

  it("firstUnlabelled resumes mid-session and saturates when done", () => {
    const rows = parseManifest([
      ["aa", "c", "t", 20, 20],
      ["bb", "c", "t", 20, 20],
      ["cc", "c", "t", 20, 20],
    ]);
    const none = foldCropLabels([]);
    expect(firstUnlabelled(rows, none)).toBe(0);
    const some = foldCropLabels([
      rec("aa", "accept", "2026-09-03T10:00:00Z"),
      // out-of-order labelling (g-jump backwards) must not confuse resume
      rec("cc", "deny", "2026-09-03T10:02:00Z"),
    ]);
    expect(firstUnlabelled(rows, some)).toBe(1);
    const all = foldCropLabels([
      rec("aa", "accept", "1"),
      rec("bb", "skip", "2"),
      rec("cc", "deny", "3"),
    ]);
    expect(firstUnlabelled(rows, all)).toBe(3);
  });
});

describe("overlayRect", () => {
  it("maps the source box into the padded crop window", () => {
    // 100x50 px box at (100,100)-(200,150), window padded 15% per axis:
    // (85,92)-(215,158) — exactly what crop_rect produces (mirrored in
    // tests/test_swarm_crops.py on the python side)
    const r = overlayRect({
      window: [85, 92, 215, 158],
      xyxy: [100 / 864, 100 / 648, 200 / 864, 150 / 648],
    });
    expect(r).not.toBeNull();
    expect(r!.x).toBeCloseTo((100 - 85) / 130, 5);
    expect(r!.y).toBeCloseTo((100 - 92) / 66, 5);
    expect(r!.w).toBeCloseTo(100 / 130, 5);
    expect(r!.h).toBeCloseTo(50 / 66, 5);
  });

  it("clamps a box that spills past a clamped window to the crop edge", () => {
    // box touching the frame edge: window clamped at 0, box starts at 0 too
    const r = overlayRect({ window: [0, 0, 100, 100], xyxy: [-0.01, -0.01, 50 / 864, 50 / 648] });
    expect(r!.x).toBe(0);
    expect(r!.y).toBe(0);
  });

  it("returns null without window data (legacy manifest) or degenerate boxes", () => {
    expect(overlayRect({ window: null, xyxy: [0, 0, 1, 1] })).toBeNull();
    expect(overlayRect({ window: [0, 0, 10, 10], xyxy: null })).toBeNull();
    expect(overlayRect({ window: [10, 10, 10, 20], xyxy: [0, 0, 1, 1] })).toBeNull();
  });
});

describe("orderForReview", () => {
  const row = (id: string, window: [number, number, number, number] | null) =>
    ({ crop_id: id, chunk: "c", ts: "t", crop_w: 20, crop_h: 20, window, xyxy: null });

  it("flags near-frame windows as giants (> 55% of 864x648)", () => {
    expect(isGiant(row("g", [0, 0, 864, 648]))).toBe(true);
    expect(isGiant(row("g", [50, 40, 800, 600]))).toBe(true); // ~75%
    expect(isGiant(row("n", [0, 0, 400, 300]))).toBe(false); // ~21%
    expect(isGiant(row("legacy", null))).toBe(false); // legacy rows stay put
  });

  it("demotes all giants to the end, keeping relative order (stable)", () => {
    const rows = [
      row("g1", [0, 0, 864, 648]),
      row("n1", [0, 0, 100, 100]),
      row("g2", [10, 10, 860, 640]),
      row("n2", [0, 0, 50, 50]),
      row("n3", null),
    ];
    expect(orderForReview(rows).map((r) => r.crop_id)).toEqual([
      "n1",
      "n2",
      "n3",
      "g1",
      "g2",
    ]);
  });
});

describe("jumpIndex (type-to-jump, review-order positions)", () => {
  it("maps 1-based positions to clamped 0-based indices", () => {
    expect(jumpIndex(1, 100)).toBe(0);
    expect(jumpIndex(100, 100)).toBe(99);
    expect(jumpIndex(500, 100)).toBe(99); // over the end clamps
    expect(jumpIndex(0, 100)).toBe(0);
    expect(jumpIndex(-5, 100)).toBe(0);
    expect(jumpIndex(NaN, 100)).toBeNull();
    expect(jumpIndex(3, 0)).toBeNull();
  });

  it("position N addresses the REVIEW order (post giant demotion)", () => {
    const row = (id: string, window: [number, number, number, number]) =>
      ({ crop_id: id, chunk: "c", ts: "t", crop_w: 20, crop_h: 20, window, xyxy: null });
    const ordered = orderForReview([
      row("giant", [0, 0, 864, 648]),
      row("tight1", [0, 0, 100, 100]),
      row("tight2", [0, 0, 50, 50]),
    ]);
    // jumping to #1 lands on the first TIGHT crop, not the manifest's giant
    expect(ordered[jumpIndex(1, ordered.length)!]!.crop_id).toBe("tight1");
    expect(ordered[jumpIndex(3, ordered.length)!]!.crop_id).toBe("giant");
  });
});

describe("model predictions", () => {
  const row = (id: string) =>
    ({ crop_id: id, chunk: "c", ts: "t", crop_w: 20, crop_h: 20, window: null, xyxy: null });

  it("parses {crop_id: {label, p}} and drops malformed entries", () => {
    const m = parsePredictions({
      aa: { label: "accept", p: 0.97 },
      bb: { label: "junk", p: 0.55 },
      zz: { label: "bogus", p: 0.5 },
      yy: { label: "accept" },
    });
    expect(m.size).toBe(2);
    expect(m.get("aa")).toEqual({ label: "accept", p: 0.97 });
    expect(() => parsePredictions([1, 2])).toThrow();
  });

  it("asset path passes the /api/assets validator shape", () => {
    expect(predictionsAssetPath("2026-08-26")).toBe(
      "swarm_crops/2026-08-26/predictions.json",
    );
  });

  it("queueOrder: model-accepts, then uncertainty band, then rest — stable", () => {
    const preds = parsePredictions({
      a1: { label: "accept", p: 0.99 },
      u1: { label: "deny", p: 0.6 },     // uncertain (p < 0.8)
      r1: { label: "deny", p: 0.95 },    // confident non-accept -> rest
      a2: { label: "accept", p: 0.55 },  // accept, even uncertain -> bucket a
      u2: { label: "junk", p: 0.7 },
    });
    const ordered = queueOrder(
      [row("r1"), row("u1"), row("a1"), row("nopred"), row("u2"), row("a2")],
      preds,
    );
    expect(ordered.map((r) => r.crop_id)).toEqual([
      "a1", "a2",          // accepts, original relative order
      "u1", "u2",          // uncertainty band
      "r1", "nopred",      // rest (incl. rows without a prediction)
    ]);
  });
});
