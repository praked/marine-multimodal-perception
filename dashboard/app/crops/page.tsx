"use client";

import {
  CROP_SET,
  cropAssetPath,
  firstUnlabelled,
  foldCropLabels,
  jumpIndex,
  manifestAssetPath,
  orderForReview,
  overlayRect,
  parseManifest,
  parsePredictions,
  predictionsAssetPath,
  queueOrder,
  type CropLabelRecord,
  type CropPrediction,
  type CropRow,
  type CropVerdict,
} from "@/lib/crops";
import { getProvider } from "@/lib/data";
import { colourForClass } from "@/lib/palette";
import { clipKey as clipKeyFor, type ClipSummary } from "@/lib/types";
import {
  appendCropLabel,
  drainCropQueue,
  listCropLabels,
  makeRecord,
  queuedCropLabels,
} from "@/lib/cropLabels";
import { Ban, Check, HelpCircle, RotateCcw, SkipForward, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* eslint-disable @next/next/no-img-element -- R2 crop assets */

/* Swarm self-recognition crop review: one boat-detection crop at a time,
   big on screen, keyboard-first — a (ours) / x (not ours) / s (unsure),
   u or Cmd+Z undo, arrows navigate, g jumps to the first unlabelled.
   Crops come from the 2026-08-26 outing's union boat boxes, ordered
   chunk-chronological and largest-first within each chunk (the big/near
   crops are the decidable ones). Labels append to sail_crop_labels;
   latest-per-crop wins, so a refresh resumes exactly where you left off.
   Feeds the D.1 crop classifier (docs/plans/swarm_self_recognition.md). */

const VERDICT_STYLE: Record<CropVerdict, string> = {
  accept: "border-status-good text-status-good",
  deny: "border-status-serious text-status-serious",
  skip: "border-status-warn text-status-warn",
  junk: "border-border-strong text-muted",
};

const VERDICT_NAME: Record<CropVerdict, string> = {
  accept: "ours",
  deny: "other boat",
  skip: "unsure",
  junk: "bad box",
};

const MODEL_NAME: Record<CropPrediction["label"], string> = {
  accept: "OURS",
  deny: "OTHER",
  junk: "BAD BOX",
};

const ORDER_LS = "asvproject.crops.order.v1";

function frameIdFor(row: CropRow): string {
  return `2026-08-26_afloat/${row.chunk}/ts=${row.ts.replace(/:/g, "-")}`;
}

export default function CropsPage() {
  const [rows, setRows] = useState<CropRow[] | null>(null);
  const [labels, setLabels] = useState<Map<string, CropLabelRecord>>(new Map());
  const [index, setIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [queued, setQueued] = useState(0);
  const [imgFailed, setImgFailed] = useState<Set<string>>(new Set());
  // session action stack for undo: the indices we labelled, in order
  // (ref for the data, state mirror for the button's disabled flag)
  const undoStack = useRef<number[]>([]);
  const [undoDepth, setUndoDepth] = useState(0);
  const [landed, setLanded] = useState(false);
  // measured stage: the crop is drawn at an exact integer of its own
  // aspect (scale = fit-to-stage, upscaling allowed) so the box overlay
  // aligns with the pixels with no object-fit letterbox guesswork
  const stageRef = useRef<HTMLDivElement>(null);
  const [stageDims, setStageDims] = useState<[number, number]>([0, 0]);
  // context thumbnail: activity clipKey per capture chunk (from the clip
  // summaries' `chunks` membership), null = join failed, hide thumbnail
  const [chunkClips, setChunkClips] = useState<Map<string, string> | null>(null);
  const [thumbFailed, setThumbFailed] = useState<Set<string>>(new Set());
  const [showBox, setShowBox] = useState(true);
  // model predictions (additive: absent file -> page behaves as before)
  const [preds, setPreds] = useState<Map<string, CropPrediction> | null>(null);
  const [orderPref, setOrderPref] = useState<"queue" | "chrono">("queue");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // manifest lives in the private R2 bucket: resolve a signed URL via
        // the assets route's json mode (a cross-origin redirect would fail
        // the bucket's CORS policy), then fetch it directly.
        const res = await fetch(
          `/api/assets/${manifestAssetPath(CROP_SET)}?json=1`,
        );
        if (!res.ok) throw new Error(`manifest sign failed: ${res.status}`);
        const { url } = (await res.json()) as { url: string };
        const mRes = await fetch(url);
        if (!mRes.ok) throw new Error(`manifest fetch failed: ${mRes.status}`);
        // giants (near-frame windows: loose whole-scene boxes) go to the
        // very end so the session starts on tight, decidable crops
        const manifest = orderForReview(parseManifest(await mRes.json()));
        // model predictions are optional: 404/parse failure -> no model UI
        let p: Map<string, CropPrediction> | null = null;
        try {
          const pRes = await fetch(
            `/api/assets/${predictionsAssetPath(CROP_SET)}?json=1`,
          );
          if (pRes.ok) {
            const { url: pUrl } = (await pRes.json()) as { url: string };
            const pData = await fetch(pUrl);
            if (pData.ok) p = parsePredictions(await pData.json());
          }
        } catch {
          p = null;
        }
        const recs = await listCropLabels();
        if (cancelled) return;
        try {
          const stored = window.localStorage.getItem(ORDER_LS);
          if (stored === "chrono" || stored === "queue") setOrderPref(stored);
        } catch {}
        setPreds(p);
        setRows(manifest);
        setLabels(foldCropLabels(recs));
        setQueued(queuedCropLabels().length);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // chunk -> activity clipKey, from the catalogue's chunk membership (the
  // bundle concatenates successive capture chunks into activities). Failing
  // here only costs the thumbnail, never the review itself.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const clips: ClipSummary[] = await getProvider().listClips();
        if (cancelled) return;
        const map = new Map<string, string>();
        for (const c of clips) {
          if (c.scene !== "2026-08-26_afloat") continue;
          const key = clipKeyFor(c.scene, c.triplet_ts);
          for (const chunk of c.chunks ?? [c.triplet_ts]) map.set(chunk, key);
        }
        setChunkClips(map);
      } catch {
        if (!cancelled) setChunkClips(new Map());
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // stage size (ResizeObserver): drives the exact-scale crop box
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const update = () =>
      setStageDims([el.clientWidth, el.clientHeight]);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [rows]);

  // Display order: the model-assisted review queue (model-accepts, then the
  // uncertainty band, then the rest) or the plain review order.
  const view = useMemo(() => {
    if (!rows) return null;
    return preds && orderPref === "queue" ? queueOrder(rows, preds) : rows;
  }, [rows, preds, orderPref]);

  // Land on the first unlabelled crop once both loads are in (one-shot).
  if (view && !landed) {
    setLanded(true);
    setIndex(Math.min(firstUnlabelled(view, labels), view.length - 1));
  }

  const total = view?.length ?? 0;
  const row = view?.[index] ?? null;
  const labelled = useMemo(() => {
    if (!rows) return 0;
    let n = 0;
    for (const r of rows) if (labels.has(r.crop_id)) n++;
    return n;
  }, [rows, labels]);
  const current = row ? labels.get(row.crop_id) ?? null : null;
  const prediction = row && preds ? preds.get(row.crop_id) ?? null : null;
  // "model agrees with your existing labels k/n" (skip = abstain, excluded)
  const agreement = useMemo(() => {
    if (!preds) return null;
    let k = 0;
    let n = 0;
    for (const [cid, rec] of labels) {
      if (rec.label === "skip") continue;
      const pr = preds.get(cid);
      if (!pr) continue;
      n++;
      if (pr.label === rec.label) k++;
    }
    return { k, n };
  }, [preds, labels]);

  const jumpUnlabelled = useCallback(() => {
    if (!view) return;
    setIndex(Math.min(firstUnlabelled(view, labels), view.length - 1));
  }, [view, labels]);

  const switchOrder = useCallback(
    (mode: "queue" | "chrono") => {
      setOrderPref(mode);
      try {
        window.localStorage.setItem(ORDER_LS, mode);
      } catch {}
      if (rows) {
        const next =
          preds && mode === "queue" ? queueOrder(rows, preds) : rows;
        setIndex(Math.min(firstUnlabelled(next, labels), next.length - 1));
      }
    },
    [rows, preds, labels],
  );

  const applyLabel = useCallback(
    (verdict: CropVerdict) => {
      if (!row || !view) return;
      const rec = makeRecord(row.crop_id, frameIdFor(row), verdict);
      undoStack.current.push(index);
      setUndoDepth(undoStack.current.length);
      // optimistic fold, then advance
      setLabels((m) => new Map(m).set(rec.crop_id, rec));
      setIndex((i) => Math.min(i + 1, view.length - 1));
      void appendCropLabel(rec).then((outcome) => {
        setQueued(queuedCropLabels().length);
        if (outcome === "queued") {
          // keep the optimistic label — the queue banner says it's pending
        }
      });
    },
    [row, view, index],
  );

  const undo = useCallback(() => {
    const prev = undoStack.current.pop();
    setUndoDepth(undoStack.current.length);
    if (prev != null) setIndex(prev);
  }, []);

  const retryQueue = useCallback(() => {
    void drainCropQueue().then((n) => setQueued(n));
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "SELECT" ||
          target.tagName === "TEXTAREA")
      ) {
        return;
      }
      const mod = e.metaKey || e.ctrlKey;
      if (mod && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        undo();
      } else if (e.key === "a") {
        applyLabel("accept");
      } else if (e.key === "x") {
        applyLabel("deny");
      } else if (e.key === "s") {
        applyLabel("skip");
      } else if (e.key === "f") {
        applyLabel("junk");
      } else if (e.key === "u") {
        undo();
      } else if (e.key === "g") {
        jumpUnlabelled();
      } else if (e.key === "b") {
        setShowBox((v) => !v);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setIndex((i) => Math.min(i + 1, total - 1));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [applyLabel, undo, jumpUnlabelled, total]);

  // Preload the next few crops so labelling never waits on the network.
  useEffect(() => {
    if (!view) return;
    for (let k = 1; k <= 6; k++) {
      const next = view[index + k];
      if (!next) break;
      const img = new Image();
      img.src = `/api/assets/${cropAssetPath(CROP_SET, next.crop_id)}`;
    }
  }, [view, index]);

  if (error) {
    return (
      <div className="p-8 font-mono text-sm text-status-serious">
        {error}
        <p className="mt-2 text-subtle">
          The crop manifest lives in R2 under swarm_crops/{CROP_SET}/ — run
          dashboard/tools/swarm_crops/upload_crops.mjs if it is missing.
        </p>
      </div>
    );
  }
  if (!view || !row) {
    return <div className="p-8 font-mono text-sm text-subtle">loading…</div>;
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-3 border-b border-border bg-surface-1 px-4 py-2">
        <h1 className="truncate text-sm font-semibold tracking-tight">
          Our boat? · {CROP_SET} boat crops
        </h1>
        <span className="group relative">
          <button
            type="button"
            aria-label="About this review"
            className="flex h-5 w-5 items-center justify-center rounded-full border border-border text-subtle hover:border-border-strong hover:text-foreground"
          >
            <HelpCircle size={12} aria-hidden />
          </button>
          <span className="pointer-events-none absolute left-0 top-full z-20 mt-1 hidden w-96 rounded-sm border border-border bg-surface-1 p-3 text-left text-[11px] leading-snug text-muted shadow-sm group-hover:block group-focus-within:block">
            <b className="text-foreground">
              Is this crop OUR ASVProject boat (Vessel A)?
            </b>{" "}
            Every union boat detection from the 2026-08-26 outing, one crop at
            a time — big/near crops first within each chunk, the hopeless
            specks last. <b className="text-foreground">a</b> = tight box on OUR boat,{" "}
            <b className="text-foreground">x</b> = tight box on another real
            boat, <b className="text-foreground">f</b> = bad box (the
            detection itself is wrong — loose/whole-frame/not a boat),{" "}
            <b className="text-foreground">s</b> = unsure. Below ~16 px
            nothing is decidable and those crops were never extracted; if you
            can&apos;t tell, that&apos;s what s is for. Labels feed the D.1
            crop classifier (swarm self-recognition plan, decision D2).
          </span>
        </span>
        {queued > 0 && (
          <button
            type="button"
            onClick={retryQueue}
            data-queued-labels
            className="rounded-sm border border-status-warn bg-status-warn/10 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-status-warn"
            title="These labels failed to reach the database and are queued in this browser. Click to retry now; they also retry on the next successful write."
          >
            {queued} queued — retry
          </button>
        )}
        {preds && (
          <span className="flex items-center gap-1" data-order-toggle>
            {(["queue", "chrono"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => switchOrder(m)}
                aria-pressed={orderPref === m}
                title={
                  m === "queue"
                    ? "Model-assisted review queue: every model-OURS first, then the uncertain band (p < 0.8), then the rest"
                    : "Plain review order: chunk-chronological, largest first, giants last"
                }
                className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] ${orderPref === m ? "border-seeblau-100/60 bg-seeblau-100/10 text-seeblau-deep" : "border-border text-subtle hover:text-foreground"}`}
              >
                {m === "queue" ? "review queue" : "chronological"}
              </button>
            ))}
          </span>
        )}
        {agreement && agreement.n > 0 && (
          <span
            data-model-agree
            className="font-mono text-[10px] tabular-nums text-subtle"
            title="On crops you have already labelled (skip excluded), how often the model's prediction matches your verdict"
          >
            model agrees {agreement.k}/{agreement.n}
          </span>
        )}
        {prediction && (
          <span
            data-model-chip
            title={`Model prediction (calibrated p) — your key is the truth; agreeing is just pressing the same verdict`}
            className="rounded-sm border border-seeblau-100/60 bg-seeblau-100/10 px-1.5 py-0.5 font-mono text-[11px] tabular-nums text-seeblau-deep"
          >
            model: {MODEL_NAME[prediction.label]} {prediction.p.toFixed(2)}
          </span>
        )}
        {current && (
          <span
            data-verdict-check
            title={`This crop already has a label (latest: ${VERDICT_NAME[current.label]})`}
            className={`rounded-sm border px-1.5 py-0.5 font-mono text-[11px] ${VERDICT_STYLE[current.label]}`}
          >
            {current.label === "accept" ? "✓" : VERDICT_NAME[current.label]}
          </span>
        )}
        <input
          data-goto-frame
          type="text"
          inputMode="numeric"
          placeholder={`#/${total}`}
          title="Type a crop number (review order) and press Enter to jump"
          className="w-20 rounded-sm border border-border bg-surface-2 px-1.5 py-0.5 text-center font-mono text-[11px] tabular-nums placeholder:text-subtle focus:border-seeblau-100 focus:outline-none"
          onKeyDown={(e) => {
            if (e.key !== "Enter") return;
            const el = e.currentTarget;
            const idx = jumpIndex(parseInt(el.value, 10), total);
            if (idx == null) return;
            setIndex(idx);
            el.value = "";
            el.blur();
          }}
        />
        <span
          className="ml-auto font-mono text-xs tabular-nums text-muted"
          data-progress
        >
          labelled {labelled}/{total}
        </span>
      </div>

      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-surface-2 p-4">
        <div
          ref={stageRef}
          className="flex h-[70vh] w-full items-center justify-center"
          data-crop-stage
          data-crop-id={row.crop_id}
        >
          {imgFailed.has(row.crop_id) ? (
            <div className="font-mono text-xs text-status-serious">
              crop image failed to load — {row.crop_id}.jpg
            </div>
          ) : (
            (() => {
              // exact fit-to-stage scale (upscaling allowed, aspect kept),
              // so the SVG overlay shares the img's box pixel-for-pixel
              const [sw, sh] = stageDims;
              const scale =
                sw > 0 && sh > 0
                  ? Math.min(sw / row.crop_w, sh / row.crop_h)
                  : 0;
              const box = overlayRect(row);
              return (
                <div
                  className="relative"
                  style={
                    scale > 0
                      ? {
                          width: row.crop_w * scale,
                          height: row.crop_h * scale,
                        }
                      : { visibility: "hidden" }
                  }
                >
                  <img
                    key={row.crop_id}
                    src={`/api/assets/${cropAssetPath(CROP_SET, row.crop_id)}`}
                    onError={() =>
                      setImgFailed((f) => new Set(f).add(row.crop_id))
                    }
                    alt={`Boat crop ${row.crop_id} (${row.crop_w}×${row.crop_h} px)`}
                    className="h-full w-full border border-border"
                    style={{ imageRendering: "pixelated" }}
                    draggable={false}
                  />
                  {showBox && box && (
                    <svg
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      viewBox="0 0 1 1"
                      preserveAspectRatio="none"
                      data-overlay
                    >
                      {/* white halo under the class colour so the box reads
                          on dark water and bright glint alike */}
                      <rect
                        x={box.x}
                        y={box.y}
                        width={box.w}
                        height={box.h}
                        fill="none"
                        stroke="#fff"
                        strokeOpacity={0.9}
                        strokeWidth={4}
                        vectorEffect="non-scaling-stroke"
                      />
                      <rect
                        data-overlay-box
                        x={box.x}
                        y={box.y}
                        width={box.w}
                        height={box.h}
                        fill="none"
                        stroke={colourForClass("boat")}
                        strokeWidth={2}
                        vectorEffect="non-scaling-stroke"
                      />
                    </svg>
                  )}
                </div>
              );
            })()
          )}
        </div>

        {/* context thumbnail: the whole bundle frame with the same box —
            multi-boat frames stop being ambiguous. Best-effort: hidden when
            the chunk->activity join or the frame asset is unavailable. */}
        {chunkClips &&
          chunkClips.has(row.chunk) &&
          row.xyxy &&
          !thumbFailed.has(`${row.chunk}|${row.ts}`) && (
            <div
              className="absolute right-6 top-6 w-56 overflow-hidden rounded-sm border border-border bg-surface-1 shadow-sm"
              data-context-thumb
              title="The full frame this crop came from; the highlighted box is the detection you are judging"
            >
              <div className="relative">
                <img
                  src={getProvider().frameUrl(
                    chunkClips.get(row.chunk)!,
                    "frames",
                    row.ts,
                  )}
                  onError={() =>
                    setThumbFailed((f) =>
                      new Set(f).add(`${row.chunk}|${row.ts}`),
                    )
                  }
                  alt={`Frame ${row.ts} context`}
                  className="block w-full"
                  draggable={false}
                />
                <svg
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                >
                  <rect
                    x={row.xyxy[0]}
                    y={row.xyxy[1]}
                    width={row.xyxy[2] - row.xyxy[0]}
                    height={row.xyxy[3] - row.xyxy[1]}
                    fill="none"
                    stroke="#fff"
                    strokeOpacity={0.9}
                    strokeWidth={3}
                    vectorEffect="non-scaling-stroke"
                  />
                  <rect
                    x={row.xyxy[0]}
                    y={row.xyxy[1]}
                    width={row.xyxy[2] - row.xyxy[0]}
                    height={row.xyxy[3] - row.xyxy[1]}
                    fill="none"
                    stroke={colourForClass("boat")}
                    strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
              </div>
              <div className="px-1.5 py-0.5 font-mono text-[9px] text-subtle">
                full frame · box = this detection
              </div>
            </div>
          )}
      </div>

      <div className="flex items-center gap-2 border-t border-border bg-surface-1 px-3 py-2">
        <button
          type="button"
          onClick={() => applyLabel("accept")}
          className="flex items-center gap-1.5 rounded-sm bg-status-good px-3 py-1.5 text-xs font-medium text-white hover:opacity-90"
        >
          <Check size={14} aria-hidden /> ours (a)
        </button>
        <button
          type="button"
          onClick={() => applyLabel("deny")}
          className="flex items-center gap-1.5 rounded-sm border border-status-serious px-3 py-1.5 text-xs font-medium text-status-serious hover:bg-status-serious/10"
        >
          <X size={14} aria-hidden /> other (x)
        </button>
        <button
          type="button"
          onClick={() => applyLabel("junk")}
          className="flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-border-strong"
          title="The detection itself is wrong: loose/whole-frame box, dock chunk, glint — not a tight boat detection"
        >
          <Ban size={14} aria-hidden /> bad box (f)
        </button>
        <button
          type="button"
          onClick={() => applyLabel("skip")}
          className="flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-border-strong"
        >
          <SkipForward size={14} aria-hidden /> unsure (s)
        </button>
        <button
          type="button"
          onClick={undo}
          disabled={undoDepth === 0}
          className="flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-border-strong disabled:opacity-40"
        >
          <RotateCcw size={14} aria-hidden /> undo (u / ⌘Z)
        </button>
        <button
          type="button"
          onClick={jumpUnlabelled}
          className="rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-border-strong"
        >
          first unlabelled (g)
        </button>
        <span className="ml-auto flex items-center gap-3 font-mono text-xs tabular-nums text-muted">
          <span title="capture chunk">{row.chunk}</span>
          <span title="frame timestamp">{row.ts}</span>
          <span title="crop size in native pixels">
            {row.crop_w}×{row.crop_h} px
          </span>
          <span data-index>
            {index + 1}/{total}
          </span>
        </span>
      </div>

      <div
        data-progress-bar
        className="flex items-center gap-3 border-t border-border bg-surface-1 px-4 py-1.5"
      >
        <div className="h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-3">
          <div
            className="h-full bg-status-good transition-[width] duration-300"
            style={{ width: `${total > 0 ? (labelled / total) * 100 : 0}%` }}
          />
        </div>
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted">
          {labelled}/{total} labelled
        </span>
      </div>
      <div className="space-y-0.5 border-t border-border bg-surface-1 px-4 py-1.5 font-mono text-[10px] leading-relaxed text-subtle">
        <div>
          <b className="text-foreground">a</b> ours — box is tight on OUR
          ASVProject boat · <b className="text-foreground">x</b> other — box is
          tight on a real boat that is NOT ours ·{" "}
          <b className="text-foreground">f</b> bad box — the detection itself
          is wrong (loose/whole-frame/not a boat) ·{" "}
          <b className="text-foreground">s</b> unsure — can&apos;t tell
          (small/far/ambiguous)
        </div>
        <div>
          u/⌘Z undo · ←/→ navigate · g first unlabelled · b toggle box — the
          highlighted box is the detection being judged (crops are padded 15%
          around it); loose whole-scene boxes are queued last; labelling
          advances automatically; refresh resumes at the first unlabelled crop
        </div>
      </div>
    </div>
  );
}
