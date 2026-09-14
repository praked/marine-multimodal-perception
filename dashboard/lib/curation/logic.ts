import type { ClipSummary, FrameEntry } from "@/lib/types";
import { RETENTION_DAYS, type Cut, type CurationRecord } from "./types";

/* Pure curation semantics, shared by every consumer (clips list, viewer,
   planner, offline packs, the export route, the CLI tools) and mirrored
   by scripts/utils/curation.py on the Python side. Tested in
   tests/curation.test.ts. */

/** "HH:MM:SS.f" -> seconds of day. Tolerates "HH-MM-SS.f" (safe_ts). */
export function tsSeconds(ts: string): number {
  const [h, m, s] = ts.replaceAll("-", ":").split(":");
  return Number(h) * 3600 + Number(m) * 60 + Number(s);
}

/** Deleted := deleted_at set and not restored since, or purged. */
export function isDeleted(rec: CurationRecord | undefined | null): boolean {
  if (!rec) return false;
  if (rec.purged_at) return true;
  if (!rec.deleted_at) return false;
  return !rec.restored_at || rec.restored_at < rec.deleted_at;
}

export function normaliseCut(cut: Cut): Cut {
  const a = tsSeconds(cut.start_ts);
  const b = tsSeconds(cut.end_ts);
  return a <= b ? { ...cut } : { ...cut, start_ts: cut.end_ts, end_ts: cut.start_ts };
}

/** Sorted, start<=end, dropping malformed timestamps. */
export function normaliseCuts(cuts: Cut[]): Cut[] {
  return cuts
    .filter((c) => Number.isFinite(tsSeconds(c.start_ts)) && Number.isFinite(tsSeconds(c.end_ts)))
    .map(normaliseCut)
    .sort((x, y) => tsSeconds(x.start_ts) - tsSeconds(y.start_ts));
}

/** True when `ts` lies inside any cut (inclusive both ends). */
export function inCut(cuts: readonly Cut[], ts: string): boolean {
  if (cuts.length === 0) return false;
  const t = tsSeconds(ts);
  for (const c of cuts) {
    const a = tsSeconds(c.start_ts);
    const b = tsSeconds(c.end_ts);
    if (t >= Math.min(a, b) && t <= Math.max(a, b)) return true;
  }
  return false;
}

/** Frame kept := set not deleted and frame not cut. */
export function frameKept(rec: CurationRecord | undefined | null, ts: string): boolean {
  if (!rec) return true;
  if (isDeleted(rec)) return false;
  return !inCut(rec.cuts, ts);
}

export interface CutStats {
  kept: number;
  cut: number;
  keptSeconds: number;
  cutSeconds: number;
}

/** Per-frame duration = gap to the next frame, capped (chunk rolls and the
    2026-08-19 10 s/frame collapse would otherwise count as footage);
    the last frame gets the nominal 1/3 s. */
export function cutStats(frames: readonly { ts: string }[], cuts: readonly Cut[], maxGapS = 2.5): CutStats {
  const out: CutStats = { kept: 0, cut: 0, keptSeconds: 0, cutSeconds: 0 };
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]!;
    let dur = 1 / 3;
    if (i + 1 < frames.length) {
      let d = tsSeconds(frames[i + 1]!.ts) - tsSeconds(f.ts);
      if (d < 0) d += 86400;
      dur = Math.min(Math.max(d, 0), maxGapS);
    }
    if (inCut(cuts, f.ts)) {
      out.cut++;
      out.cutSeconds += dur;
    } else {
      out.kept++;
      out.keptSeconds += dur;
    }
  }
  return out;
}

/** Seconds covered by the cut ranges themselves (no frame list needed —
    the catalogue card's "trimmed" chip). A single-frame cut counts one
    nominal frame (1/3 s). Overlaps are not merged; normaliseCuts keeps
    them rare and the chip is an estimate, the viewer shows exact counts. */
export function cutRangeSeconds(cuts: readonly Cut[]): number {
  let s = 0;
  for (const c of cuts) {
    s += Math.abs(tsSeconds(c.end_ts) - tsSeconds(c.start_ts)) + 1 / 3;
  }
  return s;
}

/** Whole days until a deleted set becomes prunable (0 = prunable now). */
export function daysRemaining(deletedAtIso: string, nowMs: number, retentionDays = RETENTION_DAYS): number {
  const end = Date.parse(deletedAtIso) + retentionDays * 86400_000;
  return Math.max(0, Math.ceil((end - nowMs) / 86400_000));
}

/** Purge candidates: deleted longer than the retention window, not purged. */
export function purgeCandidates(
  records: readonly CurationRecord[],
  nowMs: number,
  retentionDays = RETENTION_DAYS,
): CurationRecord[] {
  return records.filter(
    (r) => isDeleted(r) && !r.purged_at && r.deleted_at != null &&
      daysRemaining(r.deleted_at, nowMs, retentionDays) === 0,
  );
}

export interface DeletedClip {
  clip: ClipSummary;
  record: CurationRecord;
}

/** Split a catalogue into active + deleted according to the records. */
export function applyCuration(
  clips: readonly ClipSummary[],
  records: ReadonlyMap<string, CurationRecord>,
): { active: ClipSummary[]; deleted: DeletedClip[] } {
  const active: ClipSummary[] = [];
  const deleted: DeletedClip[] = [];
  for (const clip of clips) {
    const key = clip.clip_id.replace("/", "__");
    const rec = records.get(key);
    if (rec && isDeleted(rec)) deleted.push({ clip, record: rec });
    else active.push(clip);
  }
  return { active, deleted };
}

/** Index of the nearest kept frame at/after (dir=+1) or at/before (dir=-1)
    `from`; -1 when none. With `wrap`, continues from the other end. */
export function nextKeptIndex(
  frames: readonly { ts: string }[],
  cuts: readonly Cut[],
  from: number,
  dir: 1 | -1 = 1,
  wrap = false,
): number {
  const n = frames.length;
  if (n === 0) return -1;
  let i = Math.min(Math.max(from, 0), n - 1);
  for (let step = 0; step < n; step++) {
    if (!inCut(cuts, frames[i]!.ts)) return i;
    i += dir;
    if (i < 0 || i >= n) {
      if (!wrap) return -1;
      i = (i + n) % n;
    }
  }
  return -1;
}

/** Cut regions as fractions of the frame timeline (for scrub-bar overlays
    and the health bars): contiguous cut frames become one run. */
export function cutRuns(frames: readonly { ts: string }[], cuts: readonly Cut[]): { start: number; width: number }[] {
  const n = frames.length;
  const runs: { start: number; width: number }[] = [];
  if (n === 0 || cuts.length === 0) return runs;
  let runStart = -1;
  for (let i = 0; i <= n; i++) {
    const cut = i < n && inCut(cuts, frames[i]!.ts);
    if (cut && runStart < 0) runStart = i;
    if (!cut && runStart >= 0) {
      runs.push({ start: runStart / n, width: (i - runStart) / n });
      runStart = -1;
    }
  }
  return runs;
}

/** Frames that survive curation (the offline pack + planner contract). */
export function keptFrames<T extends { ts: string }>(frames: readonly T[], rec: CurationRecord | undefined | null): T[] {
  if (!rec) return [...frames];
  if (isDeleted(rec)) return [];
  return frames.filter((f) => !inCut(rec.cuts, f.ts));
}

export type { FrameEntry };
