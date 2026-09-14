"use client";

import {
  CLASS_WEIGHTS,
  candidatesForClip,
  deletePlan,
  evalExpr,
  loadLibrary,
  renamePlan,
  selectPlan,
  setActivePlan,
  storePlan,
  type ExprOp,
  type ExprRow,
  type Plan,
  type PlanCandidate,
  adoptPlanId,
  importSharedPlans,
  type PlanLibrary,
  type StoredPlan,
} from "@/lib/annotate/planner";
import {
  deleteSharedPlan,
  listSharedPlans,
  pushSharedPlan,
  renameSharedPlan,
} from "@/lib/annotate/planStore";
import type { CurationRecord } from "@/lib/curation";
import { getProvider, listCuratedClips } from "@/lib/data";
import { colourForClass } from "@/lib/palette";
import { clipKeyFromId, type ClipSummary } from "@/lib/types";
import { ChevronDown, ChevronRight, ChevronUp, Minus, Plus, SquarePen, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, type ReactNode } from "react";

/* Annotation planner: pick N, optionally constrain with a stackable boolean
   expression over instances and conditions, build the deterministic optimal
   split, inspect its coverage, then annotate it frame by frame. */

const CONDITION_OPTIONS = [
  "part:day", "part:golden", "part:twilight", "part:night",
  "precip:dry", "precip:wet",
  "cloud:0", "cloud:1", "cloud:2", "cloud:3", "cloud:4",
  "wind:0", "wind:1", "wind:2", "wind:3",
  "luma:0", "luma:1", "luma:2",
];
const CLASS_OPTIONS = Object.keys(CLASS_WEIGHTS);

/** Hover tooltip: instant, styled, keyboard-reachable via the title too. */
function Tip({ tip, children }: { tip: string; children: ReactNode }) {
  return (
    <span className="group/tip relative inline-flex" title={tip}>
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 z-30 mb-1.5 hidden w-72 rounded-sm border border-border-strong bg-surface-2 p-2 font-sans text-[11px] font-normal normal-case leading-snug tracking-normal text-foreground shadow-lg group-hover/tip:block"
      >
        {tip}
      </span>
    </span>
  );
}

const TIPS = {
  op: "How this row combines with everything before it: AND needs both sides, OR needs either, XOR needs exactly one. Rows fold left to right: ((row1 op row2) op row3).",
  not: "Invert this row: keep frames where the predicate does NOT hold (e.g. NOT instance person = frames without a labelled person).",
  kind: "instance: a class present in the frame's labels (audited or GroundingDINO). condition: the capture conditions of the frame's clip.",
  value: "Condition codes: part = daypart (day / golden hour / civil twilight / night), precip = dry or wet, cloud = cloud-cover band 0 (clear) to 4 (overcast), wind = wind band 0 (calm) to 3 (strong), luma = frame-luminance tercile 0 (dark) to 2 (bright).",
  add: "Each added row is one predicate; the operator in front of it folds it onto the result of the rows above.",
} as const;

function StatsGrid({ stats }: { stats: Plan["stats"] }) {
  return (
    <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-3">
      <div>
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-subtle">instances (frames containing)</div>
        <ul className="space-y-0.5 font-mono text-[11px] tabular-nums">
          {Object.entries(stats.classes).sort((a, b) => b[1] - a[1]).map(([c, v]) => (
            <li key={c} className="flex justify-between gap-2">
              <span className="flex items-center gap-1.5">
                <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: colourForClass(c) }} />
                {c}
              </span>
              <span>{v}</span>
            </li>
          ))}
          {Object.keys(stats.classes).length === 0 && (
            <li className="text-subtle">none labelled yet</li>
          )}
        </ul>
        <div className="mt-2 font-mono text-[11px] text-subtle">
          labelled {stats.labelled} · unlabelled {stats.unlabelled}
        </div>
      </div>
      <div>
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-subtle">conditions</div>
        <ul className="space-y-0.5 font-mono text-[11px] tabular-nums">
          {Object.entries(stats.cells).sort((a, b) => b[1] - a[1]).map(([c, v]) => (
            <li key={c} className="flex justify-between gap-2"><span className="text-muted">{c}</span><span>{v}</span></li>
          ))}
        </ul>
      </div>
      <div>
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-subtle">clips</div>
        <ul className="space-y-0.5 font-mono text-[11px] tabular-nums">
          {Object.entries(stats.clips).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([c, v]) => (
            <li key={c} className="flex justify-between gap-2"><span className="truncate text-muted">{c}</span><span>{v}</span></li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function AnnotatePlanPage() {
  const router = useRouter();
  const [clips, setClips] = useState<ClipSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [n, setN] = useState(200);
  const [advanced, setAdvanced] = useState(false);
  const [rows, setRows] = useState<ExprRow[]>([]);
  const [building, setBuilding] = useState<string | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planName, setPlanName] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [lib, setLib] = useState<PlanLibrary | null>(null);
  const [candidates, setCandidates] = useState<PlanCandidate[] | null>(null);
  const [curation, setCuration] = useState<Map<string, CurationRecord>>(new Map());

  useEffect(() => {
    // localStorage is a browser-only external store; read it post-render
    void Promise.resolve().then(() => setLib(loadLibrary()));
    // shared plans (sail_plans) are the cross-browser source of truth;
    // cursors stay local. Failure keeps the local library working.
    listSharedPlans()
      .then((shared) => setLib(importSharedPlans(shared)))
      .catch(() => undefined);
    // deleted sets never reach the planner; cuts drop frames below
    listCuratedClips()
      .then(({ active, records }) => {
        setClips(active);
        setCuration(records);
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  async function loadCandidates(): Promise<PlanCandidate[]> {
    if (candidates) return candidates;
    const provider = getProvider();
    const all: PlanCandidate[] = [];
    const list = clips ?? [];
    for (let i = 0; i < list.length; i++) {
      const clip = list[i]!;
      const key = `${clip.scene}__${clip.triplet_ts}`;
      setBuilding(`loading ${i + 1}/${list.length}: ${clip.title}`);
      try {
        const [sectors, labels] = await Promise.all([
          provider.getSectors(key),
          provider.getLabels(key),
        ]);
        all.push(...candidatesForClip(clip, sectors, labels, curation.get(key)?.cuts ?? []));
      } catch {
        // clip without sectors: nothing to rank there — skipped honestly
      }
    }
    setCandidates(all);
    return all;
  }

  async function build() {
    setBuilding("starting…");
    try {
      let cands = await loadCandidates();
      if (advanced && rows.length > 0) {
        cands = cands.filter((c) => evalExpr(rows, c.classes, c.cells));
      }
      setBuilding("selecting…");
      // yield a frame so the progress text paints before the greedy loop
      await new Promise((r) => setTimeout(r, 30));
      const p = selectPlan(cands, n);
      setPlan(p);
      // persist immediately so the plan survives leaving the page
      const autoName = advanced && rows.length > 0
        ? rows.map((r) => `${r.pred.negate ? "NOT " : ""}${r.pred.value}`).join(" ").slice(0, 48)
        : "auto";
      const stored = storePlan(p, planName.trim() || `${autoName} · ${p.frames.length}f`);
      setPlanName("");
      setLib(loadLibrary());
      // publish so the plan appears in every browser; adopt the shared uuid
      try {
        const sharedId = await pushSharedPlan(stored);
        if (sharedId && stored.id) {
          adoptPlanId(stored.id, sharedId);
          setLib(loadLibrary());
        }
      } catch {
        /* offline/demo: the plan stays local */
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBuilding(null);
    }
  }

  function startAnnotating() {
    if (!plan || plan.frames.length === 0) return;
    // the freshly built plan is already stored + active
    const first = plan.frames[0]!;
    router.push(`/annotate/${first.clipKey}?plan=1`);
  }

  function continuePlan(p: StoredPlan) {
    if (p.frames.length === 0 || !p.id) return;
    setActivePlan(p.id);
    const at = p.frames[Math.min(p.cursor, p.frames.length - 1)]!;
    router.push(`/annotate/${at.clipKey}?plan=1`);
  }

  function removePlan(p: StoredPlan) {
    if (!p.id) return;
    deletePlan(p.id);
    setLib(loadLibrary());
    if (p.id.includes("-")) void deleteSharedPlan(p.id).catch(() => undefined);
  }

  const availableNote = useMemo(() => {
    if (!plan) return null;
    if (!plan.clamped) return null;
    return `only ${plan.frames.length} frames satisfy the constraints — the plan is the maximum available (requested ${plan.requested})`;
  }, [plan]);

  if (error) return <div className="p-8 font-mono text-sm text-status-serious">{error}</div>;
  if (!clips) return <div className="p-8 font-mono text-sm text-subtle">loading…</div>;

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        Annotation · optimal split planner
      </div>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">Plan an audit session</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Deterministic best-N selection across the corpus: scorer/label
        disagreement, rarity-weighted class quotas, condition coverage and
        temporal spacing. Same inputs, same plan — and if fewer frames exist
        than you ask for, you get exactly what exists.
      </p>

      <section className="mt-6 rounded-sm border border-border bg-surface-1 p-4">
        <div className="flex flex-wrap items-end gap-4">
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-subtle">
              frames to annotate
            </span>
            <input
              type="number" min={1} max={20000} value={n}
              onChange={(e) => setN(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
              className="w-28 rounded-sm border border-border bg-surface-1 px-2 py-1.5 text-right font-mono text-sm tabular-nums focus:border-seeblau-100 focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-subtle">
              plan name (optional)
            </span>
            <input
              type="text" value={planName} placeholder="auto-named from conditions"
              onChange={(e) => setPlanName(e.target.value)}
              className="w-56 rounded-sm border border-border bg-surface-1 px-2 py-1.5 font-mono text-sm focus:border-seeblau-100 focus:outline-none"
            />
          </label>
          <button
            type="button"
            onClick={() => setAdvanced((a) => !a)}
            className={`rounded-sm border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider ${advanced ? "border-seeblau-100 bg-seeblau-100/10 text-seeblau-100" : "border-border text-muted hover:text-foreground"}`}
          >
            advanced conditions
          </button>
          <button
            type="button" onClick={() => void build()} disabled={building != null}
            className="rounded-sm bg-seeblau-100 px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-inverse hover:bg-seeblau-deep disabled:opacity-50"
          >
            {building ?? "build plan"}
          </button>
        </div>

        {advanced && (
          <div className="mt-4 space-y-2 border-t border-border pt-3">
            <div className="font-mono text-[10px] uppercase tracking-wider text-subtle">
              constraint expression · left-to-right fold: ((r1 op r2) op r3)…
            </div>
            {rows.map((row, i) => (
              <div key={i} className="flex flex-wrap items-center gap-2 font-mono text-xs">
                {i > 0 ? (
                  <Tip tip={TIPS.op}>
                    <select
                      value={row.op}
                      onChange={(e) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, op: e.target.value as ExprOp } : r)))}
                      className="rounded-sm border border-border bg-surface-1 px-1.5 py-1"
                      aria-label="operator"
                    >
                      <option>AND</option><option>OR</option><option>XOR</option>
                    </select>
                  </Tip>
                ) : (
                  <Tip tip="Only frames satisfying the whole expression become plan candidates.">
                    <span className="w-14 text-subtle">WHERE</span>
                  </Tip>
                )}
                <Tip tip={TIPS.not}>
                  <button
                    type="button"
                    onClick={() => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, pred: { ...r.pred, negate: !r.pred.negate } } : r)))}
                    className={`rounded-sm border px-1.5 py-1 ${row.pred.negate ? "border-status-warn text-status-warn" : "border-border text-subtle"}`}
                    aria-pressed={row.pred.negate}
                  >
                    NOT
                  </button>
                </Tip>
                <Tip tip={TIPS.kind}>
                  <select
                    value={row.pred.kind}
                    onChange={(e) => {
                      const kind = e.target.value as "class" | "condition";
                      setRows((rs) => rs.map((r, j) => (j === i ? { ...r, pred: { kind, value: kind === "class" ? CLASS_OPTIONS[0]! : CONDITION_OPTIONS[0]!, negate: r.pred.negate } } : r)));
                    }}
                    className="rounded-sm border border-border bg-surface-1 px-1.5 py-1"
                    aria-label="predicate kind"
                  >
                    <option value="class">instance</option>
                    <option value="condition">condition</option>
                  </select>
                </Tip>
                <Tip tip={row.pred.kind === "class"
                  ? "The labelled class the frame must contain (or must not, with NOT)."
                  : TIPS.value}>
                  <select
                    value={row.pred.value}
                    onChange={(e) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, pred: { ...r.pred, value: e.target.value } } : r)))}
                    className="rounded-sm border border-border bg-surface-1 px-1.5 py-1"
                    aria-label="predicate value"
                  >
                    {(row.pred.kind === "class" ? CLASS_OPTIONS : CONDITION_OPTIONS).map((o) => (
                      <option key={o}>{o}</option>
                    ))}
                  </select>
                </Tip>
                <button
                  type="button" aria-label="move row up" disabled={i === 0}
                  onClick={() => setRows((rs) => { const c = [...rs]; [c[i - 1], c[i]] = [c[i]!, c[i - 1]!]; return c; })}
                  className="text-subtle hover:text-foreground disabled:opacity-25"
                >
                  <ChevronUp size={14} />
                </button>
                <button
                  type="button" aria-label="move row down" disabled={i === rows.length - 1}
                  onClick={() => setRows((rs) => { const c = [...rs]; [c[i], c[i + 1]] = [c[i + 1]!, c[i]!]; return c; })}
                  className="text-subtle hover:text-foreground disabled:opacity-25"
                >
                  <ChevronDown size={14} />
                </button>
                <button
                  type="button" aria-label="remove row"
                  onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}
                  className="text-subtle hover:text-status-serious"
                >
                  <Minus size={14} />
                </button>
              </div>
            ))}
            <Tip tip={TIPS.add}>
              <button
                type="button"
                onClick={() => setRows((rs) => [...rs, { op: "AND", pred: { kind: "class", value: "person", negate: false } }])}
                className="flex items-center gap-1 font-mono text-[11px] text-seeblau-100 hover:text-seeblau-deep"
              >
                <Plus size={13} /> add condition
              </button>
            </Tip>
            <p className="text-[11px] text-subtle">
              instance predicates match a frame&apos;s labelled classes; condition
              predicates match the clip&apos;s capture conditions (daypart, cloud/
              wind bands, wet/dry, luminance tercile).
            </p>
          </div>
        )}
      </section>

      {plan && (
        <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
          <div className="flex items-baseline justify-between">
            <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
              Plan · {plan.frames.length} frames
            </h2>
            <button
              type="button" onClick={startAnnotating}
              disabled={plan.frames.length === 0}
              className="flex items-center gap-1 rounded-sm bg-seeblau-100 px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-inverse hover:bg-seeblau-deep disabled:opacity-50"
            >
              start annotating <ChevronRight size={13} />
            </button>
          </div>
          {availableNote && (
            <p className="mt-1 font-mono text-[11px] text-status-warn">{availableNote}</p>
          )}
          <StatsGrid stats={plan.stats} />
        </section>
      )}

      {lib && lib.plans.length > 0 && (
        <section className="mt-4 rounded-sm border border-seeblau-100/40 bg-surface-1 p-4">
          <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
            Saved plans · {lib.plans.length} · stored in this browser
          </h2>
          <ul className="mt-2 space-y-2">
            {lib.plans.map((p) => {
              const pid = p.id ?? p.createdAt;
              const isOpen = expanded.has(pid);
              return (
              <li key={pid} className="rounded-sm border border-border/60 px-2 py-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button" aria-expanded={isOpen}
                    aria-label={`${isOpen ? "collapse" : "expand"} plan breakdown`}
                    onClick={() => setExpanded((e) => { const c = new Set(e); if (c.has(pid)) c.delete(pid); else c.add(pid); return c; })}
                    className="text-subtle hover:text-foreground"
                  >
                    {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
                  {p.id === lib.activeId && (
                    <span title="active plan: the one the annotate stage navigates"
                          className="h-2 w-2 shrink-0 rounded-full bg-seeblau-100" aria-label="active plan" />
                  )}
                  <input
                    type="text" value={p.name ?? ""}
                    aria-label="plan name"
                    onChange={(e) => { if (p.id) { renamePlan(p.id, e.target.value); setLib(loadLibrary()); } }}
                    onBlur={(e) => { if (p.id?.includes("-")) void renameSharedPlan(p.id, e.target.value).catch(() => undefined); }}
                    className="min-w-56 flex-1 rounded-sm border border-transparent bg-transparent px-1 py-0.5 font-mono text-sm hover:border-border focus:border-seeblau-100 focus:outline-none"
                  />
                  <span className="ml-auto text-right font-mono text-[11px] tabular-nums text-subtle">
                    at {Math.min(p.cursor + 1, p.frames.length)}/{p.frames.length} · built {p.createdAt.slice(0, 16).replace("T", " ")}
                  </span>
                  <button
                    type="button" aria-label={`delete plan ${p.name ?? ""}`}
                    onClick={() => removePlan(p)}
                    className="text-subtle hover:text-status-serious"
                  >
                    <Trash2 size={14} />
                  </button>
                  <button
                    type="button" onClick={() => continuePlan(p)}
                    className="flex items-center gap-1 rounded-sm bg-seeblau-100 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-inverse hover:bg-seeblau-deep"
                  >
                    {p.cursor > 0 ? "continue" : "annotate"} <ChevronRight size={13} />
                  </button>
                </div>
                {isOpen && (p.stats
                  ? <StatsGrid stats={p.stats} />
                  : <p className="mt-2 font-mono text-[11px] text-subtle">no coverage snapshot stored for this plan</p>)}
              </li>
              );
            })}
          </ul>
          <p className="mt-2 font-mono text-[11px] text-subtle">
            plans are shared across browsers (progress stays per-browser); the
            dot marks the active one (arrow keys in the annotate stage walk it).
            Audits are append-only across plans.
          </p>
        </section>
      )}

      <section className="mt-8">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
          Or annotate a clip directly
        </h2>
        <ul className="mt-3 space-y-2">
          {clips.map((clip) => {
            const key = clipKeyFromId(clip.clip_id);
            return (
              <li key={clip.clip_id}>
                <Link
                  href={`/annotate/${key}`}
                  className="flex items-center gap-3 rounded-sm border border-border bg-surface-1 px-4 py-3 hover:border-border-strong hover:bg-surface-2"
                >
                  <SquarePen size={16} className="text-accent-strong" aria-hidden />
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{clip.title}</div>
                    <div className="font-mono text-[11px] text-subtle">{clip.clip_id}</div>
                  </div>
                  <span className="ml-auto font-mono text-xs tabular-nums text-muted">
                    {clip.n_labelled ?? 0} labelled frames
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
