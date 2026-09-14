/* Swarm self-recognition crop review (/crops): manifest + label folding.

   The extraction script (tools/swarm_crops/extract_crops.py) uploads a
   compact manifest to R2 (`swarm_crops/<set>/manifest.json`): an array of
   [crop_id, chunk, ts, crop_w, crop_h] rows, already ordered
   chunk-chronological / largest-first-within-chunk, plus the full JSONL
   beside it for offline joins. Labels append to public.sail_crop_labels
   (never updated or deleted); readers fold to latest-per-crop. */

export const CROP_SET = "2026-08-26";

export type CropVerdict = "accept" | "deny" | "skip" | "junk";

export type Rect4 = [number, number, number, number];

export interface CropRow {
  crop_id: string;
  chunk: string;
  ts: string;
  crop_w: number;
  crop_h: number;
  /** The padded crop window in FRAME pixels — the jpg is exactly this region. */
  window: Rect4 | null;
  /** The source detection box, normalised on the 864x648 frame. */
  xyxy: Rect4 | null;
}

export interface CropLabelRecord {
  id: string;
  crop_id: string;
  frame_id?: string;
  label: CropVerdict;
  created_at: string;
}

export type CompactManifest = [
  string,
  string,
  string,
  number,
  number,
  Rect4 | null,
  Rect4 | null,
][];

export function parseManifest(compact: unknown): CropRow[] {
  if (!Array.isArray(compact)) throw new Error("manifest: not an array");
  return compact.map((r) => {
    if (!Array.isArray(r) || r.length < 5) throw new Error("manifest: bad row");
    const [crop_id, chunk, ts, crop_w, crop_h, window, xyxy] =
      r as CompactManifest[number];
    return {
      crop_id,
      chunk,
      ts,
      crop_w,
      crop_h,
      // pre-window manifests carried 5 fields; the overlay just hides
      window: Array.isArray(window) && window.length === 4 ? window : null,
      xyxy: Array.isArray(xyxy) && xyxy.length === 4 ? xyxy : null,
    };
  });
}

export const FRAME_W = 864;
export const FRAME_H = 648;

/** Window-area share of the frame above which a crop is a "giant": a
    near-frame window is almost never a tight detection of one boat —
    it is a whole-scene box that happens to CONTAIN boats. */
export const GIANT_FRACTION = 0.55;

export function isGiant(
  row: Pick<CropRow, "window">,
  frameW = FRAME_W,
  frameH = FRAME_H,
): boolean {
  if (!row.window) return false; // legacy rows: cannot tell, keep in place
  const [x0, y0, x1, y1] = row.window;
  return ((x1 - x0) * (y1 - y0)) / (frameW * frameH) > GIANT_FRACTION;
}

/** Review order: manifest order (chunk-chronological, largest-first within
    chunk) but with every giant demoted to the very end, relative order
    kept — the session starts on tight, decidable crops and the loose
    whole-scene boxes come last. Pure, stable — unit-tested. */
export function orderForReview(rows: CropRow[]): CropRow[] {
  const normal: CropRow[] = [];
  const giants: CropRow[] = [];
  for (const r of rows) (isGiant(r) ? giants : normal).push(r);
  return [...normal, ...giants];
}

/** Where the source detection box sits INSIDE the displayed crop, as a
    normalised {x, y, w, h} on the crop image — the jpg is exactly the
    `window` region of the frame, so the box maps by translating into the
    window origin and dividing by the window size. Null when the manifest
    predates the window field. Pure — unit-tested. */
export function overlayRect(
  row: Pick<CropRow, "window" | "xyxy">,
  frameW = FRAME_W,
  frameH = FRAME_H,
): { x: number; y: number; w: number; h: number } | null {
  if (!row.window || !row.xyxy) return null;
  const [wx0, wy0, wx1, wy1] = row.window;
  const ww = wx1 - wx0;
  const wh = wy1 - wy0;
  if (ww <= 0 || wh <= 0) return null;
  const bx0 = row.xyxy[0] * frameW;
  const by0 = row.xyxy[1] * frameH;
  const bx1 = row.xyxy[2] * frameW;
  const by1 = row.xyxy[3] * frameH;
  const x = Math.max(0, Math.min(1, (bx0 - wx0) / ww));
  const y = Math.max(0, Math.min(1, (by0 - wy0) / wh));
  const x1 = Math.max(0, Math.min(1, (bx1 - wx0) / ww));
  const y1 = Math.max(0, Math.min(1, (by1 - wy0) / wh));
  if (x1 <= x || y1 <= y) return null;
  return { x, y, w: x1 - x, h: y1 - y };
}

/** Latest label per crop_id (append-only log -> current state). */
export function foldCropLabels(
  records: CropLabelRecord[],
): Map<string, CropLabelRecord> {
  const byCrop = new Map<string, CropLabelRecord>();
  for (const rec of records) {
    const prev = byCrop.get(rec.crop_id);
    if (!prev || rec.created_at >= prev.created_at) byCrop.set(rec.crop_id, rec);
  }
  return byCrop;
}

/** Index of the first manifest row without a label; rows.length when done. */
export function firstUnlabelled(
  rows: CropRow[],
  labels: Map<string, CropLabelRecord>,
): number {
  for (let i = 0; i < rows.length; i++) {
    if (!labels.has(rows[i]!.crop_id)) return i;
  }
  return rows.length;
}

/** R2 asset key for one crop image (served via /api/assets). */
export function cropAssetPath(set: string, cropId: string): string {
  return `swarm_crops/${set}/crops/${cropId}.jpg`;
}

export function manifestAssetPath(set: string): string {
  return `swarm_crops/${set}/manifest.json`;
}

/** 1-based position in the review order -> clamped 0-based index (the
    type-to-jump input). NaN/absurd input returns null (no jump). */
export function jumpIndex(n: number, total: number): number | null {
  if (!Number.isFinite(n) || total <= 0) return null;
  return Math.max(0, Math.min(Math.trunc(n) - 1, total - 1));
}

// ---- model predictions (additive: no predictions.json -> no change) -------

export interface CropPrediction {
  label: "accept" | "deny" | "junk";
  p: number;
}

export function predictionsAssetPath(set: string): string {
  return `swarm_crops/${set}/predictions.json`;
}

export function parsePredictions(data: unknown): Map<string, CropPrediction> {
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("predictions: not an object");
  }
  const out = new Map<string, CropPrediction>();
  for (const [cid, v] of Object.entries(data as Record<string, unknown>)) {
    const rec = v as { label?: unknown; p?: unknown };
    if (
      (rec.label === "accept" || rec.label === "deny" || rec.label === "junk") &&
      typeof rec.p === "number"
    ) {
      out.set(cid, { label: rec.label, p: rec.p });
    }
  }
  return out;
}

/** Probability below which a prediction counts as uncertain (review band). */
export const UNCERTAIN_P = 0.8;

/** Model-assisted review queue: (a) every model-accept, then (b) the
    uncertainty band (max-p < UNCERTAIN_P, not already in a), then (c) the
    rest. Relative (giant-demoted review) order is kept inside each bucket;
    rows without a prediction fall to (c). Pure, stable — unit-tested. */
export function queueOrder(
  rows: CropRow[],
  preds: Map<string, CropPrediction>,
  band = UNCERTAIN_P,
): CropRow[] {
  const accepts: CropRow[] = [];
  const uncertain: CropRow[] = [];
  const rest: CropRow[] = [];
  for (const r of rows) {
    const p = preds.get(r.crop_id);
    if (p?.label === "accept") accepts.push(r);
    else if (p && p.p < band) uncertain.push(r);
    else rest.push(r);
  }
  return [...accepts, ...uncertain, ...rest];
}
