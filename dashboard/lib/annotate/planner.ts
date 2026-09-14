import { inCut } from "@/lib/curation/logic";
import type { Cut } from "@/lib/curation/types";
import type { ClipSummary, LabelRecord, SectorRecord } from "@/lib/types";

/* Deterministic annotation planner: pick the best N frames to annotate.
   A faithful port of scripts/eval/audit_frame_selector.py — greedy
   budgeted selection balancing scorer/label disagreement + entropy,
   rarity-weighted class quotas, condition-cell coverage and temporal
   spacing, with a standing bonus for unlabelled frames. Fully
   deterministic: stable inputs -> identical plan (ties break on
   frame id). If N exceeds the candidate pool the plan is simply the
   maximum available — never padded, never resampled. */

export const CLASS_WEIGHTS: Record<string, number> = {
  person: 3, duck: 3, animal: 3, buoy: 2, float: 2, other: 1.5,
  structure: 1, boat: 1,
};
const CLASS_CANON: Record<string, string> = {
  boat_ship: "boat", row_boats: "boat", swimmer: "person",
  paddle_board: "float",
};
export const MIN_SPACING_FRAMES = 8; // ±7-frame target dilation => near-dupes

export interface PlanCandidate {
  clipKey: string;
  clipTitle: string;
  ts: string;
  idx: number; // frame position within the clip (spacing metric)
  u: number; // uncertainty
  classes: string[];
  cells: string[];
  labelled: boolean;
}

export interface PlanFrame extends PlanCandidate {
  rank: number;
  gain: number;
}

export interface Plan {
  requested: number;
  frames: PlanFrame[];
  clamped: boolean;
  stats: PlanStats;
}

export interface PlanStats {
  classes: Record<string, number>; // frames containing the class
  cells: Record<string, number>;
  clips: Record<string, number>;
  labelled: number;
  unlabelled: number;
}

/* ---- condition cells (must mirror scripts/eval/audit_frame_selector) -- */

export function conditionCells(clip: ClipSummary): string[] {
  const e = clip.enrichment;
  if (!e) return [];
  const cells: string[] = [];
  const parts = e.dayparts ?? {};
  const keys = Object.keys(parts);
  if (keys.length) {
    const top = keys.reduce((a, b) => ((parts[a] ?? 0) >= (parts[b] ?? 0) ? a : b));
    cells.push(`part:${top}`);
  }
  const w = e.weather;
  if (w) {
    if (w.cloud_cover_pct != null) cells.push(`cloud:${Math.floor(w.cloud_cover_pct / 20)}`);
    if (w.wind_speed_kmh != null) cells.push(`wind:${Math.floor(w.wind_speed_kmh / 10)}`);
    if (w.precipitation_mm != null) cells.push(w.precipitation_mm >= 0.1 ? "precip:wet" : "precip:dry");
  }
  if (e.luminance?.mean != null) cells.push(`luma:${Math.floor(e.luminance.mean / 85)}`);
  return cells;
}

/* ---- boolean condition builder ---------------------------------------- */

export interface Predicate {
  kind: "class" | "condition";
  value: string; // canonical class or condition cell token
  negate: boolean;
}

export type ExprOp = "AND" | "OR" | "XOR";

export interface ExprRow {
  op: ExprOp; // ignored on the first row
  pred: Predicate;
}

function predTrue(p: Predicate, classes: string[], cells: string[]): boolean {
  const v = p.kind === "class" ? classes.includes(p.value) : cells.includes(p.value);
  return p.negate ? !v : v;
}

/** Left-fold evaluation: ((p1 op2 p2) op3 p3) … — simple, order-explicit. */
export function evalExpr(rows: ExprRow[], classes: string[], cells: string[]): boolean {
  if (rows.length === 0) return true;
  let acc = predTrue(rows[0]!.pred, classes, cells);
  for (const row of rows.slice(1)) {
    const v = predTrue(row.pred, classes, cells);
    acc = row.op === "AND" ? acc && v : row.op === "OR" ? acc || v : acc !== v;
  }
  return acc;
}

/* ---- candidates -------------------------------------------------------- */

function uncertainty(p: number[], labelBins: Set<number> | null, centers: number[]): number {
  const n = p.length || 1;
  let ent = 0;
  for (const pi of p) ent += 1 - Math.abs(2 * pi - 1);
  ent /= n;
  if (labelBins == null) return 0.6 * ent;
  let dis = 0;
  for (let i = 0; i < p.length; i++) {
    dis += Math.abs(p[i]! - (labelBins.has(centers[i]!) ? 1 : 0));
  }
  return dis / n + 0.4 * ent;
}

export function candidatesForClip(
  clip: ClipSummary,
  sectors: Record<string, SectorRecord>,
  labels: Record<string, LabelRecord>,
  cuts: readonly Cut[] = [],
): PlanCandidate[] {
  const cells = conditionCells(clip);
  const out: PlanCandidate[] = [];
  const tss = Object.keys(sectors).sort();
  tss.forEach((ts, idx) => {
    // curation cut: never a candidate (idx stays the raw position so the
    // spacing metric matches scripts/eval/audit_frame_selector.py)
    if (cuts.length && inCut(cuts, ts)) return;
    const r = sectors[ts]!;
    const p = r.p_obstacle;
    const centers = r.bin_centers_deg;
    if (!p || !p.length || !centers) return;
    const rec = labels[ts];
    const labelBins = rec ? new Set(rec.obstacle_bins_fisheye ?? []) : null;
    const classes = [
      ...new Set(
        (rec?.fisheye_bboxes ?? []).map((b) => CLASS_CANON[b.cls] ?? b.cls),
      ),
    ];
    out.push({
      clipKey: `${clip.scene}__${clip.triplet_ts}`,
      clipTitle: clip.title,
      ts,
      idx,
      u: uncertainty(p, labelBins, centers),
      classes,
      cells,
      labelled: rec != null,
    });
  });
  return out;
}

/* ---- greedy deterministic selection ------------------------------------ */

export function selectPlan(
  cands: PlanCandidate[],
  n: number,
  weights = { uncertainty: 1.0, cls: 1.0, condition: 0.8 },
): Plan {
  const requested = Math.max(0, Math.floor(n));
  // deterministic ordering: uncertainty desc, then clipKey/ts for ties
  const sorted = [...cands].sort(
    (a, b) => b.u - a.u
      || a.clipKey.localeCompare(b.clipKey)
      || a.ts.localeCompare(b.ts),
  );
  const pool = sorted.slice(0, Math.max(requested * 20, 5000));
  const remaining = [...pool];
  const clsCount: Record<string, number> = {};
  const cellCount: Record<string, number> = {};
  const pickedIdx: Record<string, number[]> = {};
  const chosen: PlanFrame[] = [];

  const target = Math.min(requested, pool.length);
  while (chosen.length < target) {
    let best: PlanCandidate | null = null;
    let bestGain = -Infinity;
    for (const c of remaining) {
      const picked = pickedIdx[c.clipKey];
      if (picked && picked.some((i) => Math.abs(c.idx - i) < MIN_SPACING_FRAMES)) continue;
      let g = weights.uncertainty * c.u;
      for (const cls of c.classes) {
        g += (weights.cls * (CLASS_WEIGHTS[cls] ?? 1)) / Math.sqrt(1 + (clsCount[cls] ?? 0));
      }
      const cellDiv = c.cells.length || 1;
      for (const cell of c.cells) {
        g += weights.condition / cellDiv / Math.sqrt(1 + (cellCount[cell] ?? 0));
      }
      if (!c.labelled) g += 0.15;
      if (
        g > bestGain ||
        (g === bestGain && best != null &&
          (c.clipKey + c.ts).localeCompare(best.clipKey + best.ts) < 0)
      ) {
        best = c;
        bestGain = g;
      }
    }
    if (best == null) break; // spacing exhausted the pool: deliver what fits
    chosen.push({ ...best, rank: chosen.length + 1, gain: bestGain });
    remaining.splice(remaining.indexOf(best), 1);
    (pickedIdx[best.clipKey] ??= []).push(best.idx);
    for (const cls of best.classes) clsCount[cls] = (clsCount[cls] ?? 0) + 1;
    for (const cell of best.cells) cellCount[cell] = (cellCount[cell] ?? 0) + 1;
  }

  const stats: PlanStats = {
    classes: clsCount,
    cells: cellCount,
    clips: {},
    labelled: chosen.filter((c) => c.labelled).length,
    unlabelled: chosen.filter((c) => !c.labelled).length,
  };
  for (const c of chosen) stats.clips[c.clipKey] = (stats.clips[c.clipKey] ?? 0) + 1;
  return { requested, frames: chosen, clamped: chosen.length < requested, stats };
}

/* ---- persisted plan ----------------------------------------------------- */

export interface StoredPlan {
  version: 1;
  /** stable id within the library (absent only in the legacy v1 slot) */
  id?: string;
  /** user-visible name (editable in the library) */
  name?: string;
  createdAt: string;
  requested: number;
  cursor: number;
  frames: { clipKey: string; ts: string }[];
  /** Coverage stats snapshot so the planner page can re-render the plan
      panel after a reload (older stored plans may not carry them). */
  stats?: Plan["stats"];
  clamped?: boolean;
}

const PLAN_KEY = "asvproject.annotate.plan.v1"; // legacy single-plan slot (migrated)
const LIB_KEY = "asvproject.annotate.plans.v2";

export interface PlanLibrary {
  version: 2;
  activeId: string | null;
  plans: StoredPlan[];
}

function newId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function writeLib(lib: PlanLibrary): void {
  try {
    window.localStorage.setItem(LIB_KEY, JSON.stringify(lib));
  } catch {
    /* storage unavailable: plans still work for this page's session */
  }
}

/** The plan library (multiple named plans, one active). Migrates the legacy
    single-plan slot on first read so a mid-audit plan survives the upgrade. */
export function loadLibrary(): PlanLibrary {
  const empty: PlanLibrary = { version: 2, activeId: null, plans: [] };
  try {
    const raw = window.localStorage.getItem(LIB_KEY);
    if (raw) {
      const lib = JSON.parse(raw) as PlanLibrary;
      if (lib.version === 2 && Array.isArray(lib.plans)) return lib;
      return empty;
    }
    const legacy = window.localStorage.getItem(PLAN_KEY);
    if (legacy) {
      const p = JSON.parse(legacy) as StoredPlan;
      if (p.version === 1 && Array.isArray(p.frames)) {
        const migrated: PlanLibrary = {
          version: 2,
          activeId: "legacy",
          plans: [{ ...p, id: "legacy", name: p.name ?? "migrated plan" }],
        };
        writeLib(migrated);
        window.localStorage.removeItem(PLAN_KEY);
        return migrated;
      }
    }
    return empty;
  } catch {
    return empty;
  }
}

/** Append a plan to the library (named) and make it the active one. */
export function storePlan(plan: Plan, name?: string): StoredPlan {
  const stored: StoredPlan = {
    version: 1,
    id: newId(),
    name: name ?? `plan ${loadLibrary().plans.length + 1}`,
    createdAt: new Date().toISOString(),
    requested: plan.requested,
    cursor: 0,
    frames: plan.frames.map((f) => ({ clipKey: f.clipKey, ts: f.ts })),
    stats: plan.stats,
    clamped: plan.clamped,
  };
  const lib = loadLibrary();
  lib.plans.push(stored);
  lib.activeId = stored.id ?? null;
  writeLib(lib);
  return stored;
}

/** The ACTIVE plan — what the annotate stage navigates. */
export function loadPlan(): StoredPlan | null {
  const lib = loadLibrary();
  return lib.plans.find((p) => p.id === lib.activeId) ?? null;
}

export function savePlanCursor(cursor: number): void {
  const lib = loadLibrary();
  const p = lib.plans.find((x) => x.id === lib.activeId);
  if (!p) return;
  p.cursor = Math.max(0, Math.min(cursor, p.frames.length - 1));
  writeLib(lib);
}

export function setActivePlan(id: string): void {
  const lib = loadLibrary();
  if (lib.plans.some((p) => p.id === id)) {
    lib.activeId = id;
    writeLib(lib);
  }
}

export function deletePlan(id: string): void {
  const lib = loadLibrary();
  lib.plans = lib.plans.filter((p) => p.id !== id);
  if (lib.activeId === id) lib.activeId = lib.plans.at(-1)?.id ?? null;
  writeLib(lib);
}

export function renamePlan(id: string, name: string): void {
  const lib = loadLibrary();
  const p = lib.plans.find((x) => x.id === id);
  if (p) {
    p.name = name;
    writeLib(lib);
  }
}

/** True for ids minted by the shared store (uuid with dashes); local-only
    ids are compact base36. */
function isSharedId(id: string | undefined): boolean {
  return !!id && id.includes("-");
}

/** Merge the shared plan list into the local library. Shared is the source
    of truth for shared plans: new ones are added (cursor 0), names follow
    shared, and local copies of REMOTELY DELETED shared plans are dropped.
    Local-only plans (demo mode / offline builds) and cursors are kept. */
export function importSharedPlans(shared: {
  id: string; name: string; requested: number;
  frames: { clipKey: string; ts: string }[];
  stats?: Plan["stats"]; clamped: boolean; created_at: string;
}[]): PlanLibrary {
  const lib = loadLibrary();
  const sharedIds = new Set(shared.map((s) => s.id));
  lib.plans = lib.plans.filter((p) => !isSharedId(p.id) || sharedIds.has(p.id!));
  for (const s of shared) {
    const existing = lib.plans.find((p) => p.id === s.id);
    if (existing) {
      existing.name = s.name;
    } else {
      lib.plans.push({
        version: 1, id: s.id, name: s.name, createdAt: s.created_at,
        requested: s.requested, cursor: 0, frames: s.frames,
        stats: s.stats, clamped: s.clamped,
      });
    }
  }
  if (!lib.plans.some((p) => p.id === lib.activeId)) {
    lib.activeId = lib.plans.at(-1)?.id ?? null;
  }
  writeLib(lib);
  return lib;
}

/** Swap a plan's id (a local build adopting its shared uuid). */
export function adoptPlanId(oldId: string, newId: string): void {
  const lib = loadLibrary();
  const p = lib.plans.find((x) => x.id === oldId);
  if (!p) return;
  p.id = newId;
  if (lib.activeId === oldId) lib.activeId = newId;
  writeLib(lib);
}

/** Delete the ACTIVE plan (legacy name, kept for the planner page). */
export function clearPlan(): void {
  const lib = loadLibrary();
  if (lib.activeId != null) deletePlan(lib.activeId);
}
