"use client";

import { auditsToJsonl, foldAudits, getAuditBackend } from "@/lib/audit/backend";
import {
  boxesInMarquee,
  hitTestPrioritised,
  isValidBox,
  moveBox,
  normaliseBox,
  resizeBox,
  type Corner,
} from "@/lib/annotate/boxes";
import { getProvider } from "@/lib/data";
import { packedClip, packedFrames, refreshPackedClip } from "@/lib/offline/packs";
import { CLASS_KEYS, colourForClass, LABEL_CLASSES, type LabelClass } from "@/lib/palette";
import type { AuditRecord, Box, ClipMeta, LabelRecord } from "@/lib/types";
import { frameIdFor } from "@/lib/types";
import { Info, ArrowLeft, Check, Download, Pencil, Trash2, X } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadPlan, savePlanCursor, type StoredPlan } from "@/lib/annotate/planner";

/* eslint-disable @next/next/no-img-element -- bundle/storage assets */

type Drag =
  | { kind: "move"; index: number; lastX: number; lastY: number }
  | { kind: "resize"; index: number; corner: Corner }
  | { kind: "draw"; startX: number; startY: number }
  | { kind: "marquee"; startX: number; startY: number };

interface Props {
  clipKey: string;
}

/** Keyboard-first audit stage, the web successor of the Tkinter dashboard's
    --pseudo audit mode + label tool:
    {plan
      ? "a accept · r reject · ←→ plan · ⌥←→ frame · ⇧←→ ±5s · x/⌫ delete · ⇧X delete all · ⌘A select all · d draw · ⌘Z undo · ⇧⌘Z redo · c centroid · space play ·"
      : "a accept · r reject · x/⌫ delete · ⇧X delete all · ⌘A select all · d draw · ⌘Z undo · ⇧⌘Z redo · c centroid · space play · ⇧←→ ±5s ·"}
    b/d(uck via D)/B/p/m/o class keys · ←/→ navigate.
    The log is append-only; the export button downloads folded JSONL in the
    repo's training-frames shape. */
export function AnnotateStage({ clipKey }: Props) {
  const provider = getProvider();
  const backend = getAuditBackend();

  const [meta, setMeta] = useState<ClipMeta | null>(null);
  const [labels, setLabels] = useState<Record<string, LabelRecord>>({});
  const [audits, setAudits] = useState<AuditRecord[]>([]);
  const [index, setIndex] = useState(0);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  // marquee multi-select (outside draw mode): drag on empty space, then
  // class keys / delete apply to every box whose centre was inside
  const [multiSel, setMultiSel] = useState<number[]>([]);
  const [marquee, setMarquee] = useState<[number, number, number, number] | null>(null);
  const [cls, setCls] = useState<LabelClass>("boat");
  const [drawMode, setDrawMode] = useState(false);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [draft, setDraft] = useState<Box | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();
  const undoStack = useRef<Map<string, Box[][]>>(new Map());
  const redoStack = useRef<Map<string, Box[][]>>(new Map());
  const [playing, setPlaying] = useState(false);
  const [planActive, setPlanActive] = useState(true);
  const [centroidMode, setCentroidMode] = useState(false);
  // Night boost: display-only gamma for near-black fisheye frames (the 2026-09-08
  // night trials); the stored frame and the boxes are untouched. Key: b.
  const [nightBoost, setNightBoost] = useState(false);
  // Thermal assist (⌘T): the Lepton frame of the same instant, warped column by
  // column onto the fisheye by bearing (thermal linear model cx 77 / 2.82 px/deg,
  // fisheye pinhole K at native 864 px), drawn as a translucent band at the
  // horizon. Boxes are still drawn on the fisheye, so audits stay bearing-space
  // truth; this only makes the warm targets visible where the RGB is black.
  // Thermal mode (t cycles): off -> assist (band on the fisheye) -> frame
  // (pure-thermal annotation, 2026-09-12: the Lepton frame IS the stage and
  // boxes are stored in thermal space, `space: "thermal"`, normalised to the
  // recorded 160x120 frame; fisheye-space boxes are hidden while it is on and
  // vice versa, so one audit can carry both sets).
  const [thermalMode, setThermalMode] = useState<"off" | "assist" | "frame">("off");
  const thermalAssist = thermalMode === "assist";
  const curSpace: "fisheye" | "thermal" = thermalMode === "frame" ? "thermal" : "fisheye";
  const inSpace = useCallback((b: Box) => (b.space ?? "fisheye") === curSpace, [curSpace]);
  const thermalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [plan, setPlan] = useState<StoredPlan | null>(null);
  const [planCursor, setPlanCursor] = useState(0);
  // Offline packs: when the clip's JSON side files come from a pack (no
  // network, or the network failed) the stage is confined to the packed
  // frames so nothing it shows can be a broken image.
  const [packed, setPacked] = useState<Set<string> | null>(null);

  const [fromPack, setFromPack] = useState(false);
  // true only when we fell back to the pack DESPITE being online (the
  // network labels fetch failed) — the pack may be stale, so shout.
  const [packStale, setPackStale] = useState(false);
  // Teacher switch (union label sets carry per-box source: grounding-dino /
  // dart-sam3 / both). Filters only the SUGGESTION SEED of unaudited frames;
  // audited frames always show their audited boxes.
  const [teacher, setTeacher] = useState<"union" | "grounding-dino" | "dart-sam3" | "both">("union");
  const [showMasks, setShowMasks] = useState(true);
  // Full-resolution frames (864x648) exist in R2 for audit-plan frames as
  // `frames_hd`; prefer them and fall back to the baked 432x324 on 404.
  const [hdFailed, setHdFailed] = useState<Set<string>>(new Set());
  // Lossless 160x120 thermal PNGs (`thermal_hd`) exist for audit-plan frames;
  // the baked thermal JPEG (~5 KB) blocks the 3-6 px night targets. Prefer the
  // PNG in thermal-frame mode, fall back to the JPEG on 404.
  const [thermalHdFailed, setThermalHdFailed] = useState<Set<string>>(new Set());
  const teacherSeed = useCallback(
    (bs: Box[]) =>
      teacher === "union"
        ? bs
        : bs.filter((b) =>
            teacher === "both"
              ? b.source === "both"
              : b.source === teacher || b.source === "both" || b.source === undefined,
          ),
    [teacher],
  );
  const stageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    // One transient failure anywhere in the load chain (Supabase meta,
    // ?json signing, cross-origin labels fetch) used to pivot the whole
    // stage silently onto the offline pack — whose labels are as old as the
    // pack. Ride out blips with short retries instead.
    const retry = async <T,>(fn: () => Promise<T>, tries = 3): Promise<T> => {
      let last: unknown;
      for (let i = 0; i < tries; i++) {
        try {
          return await fn();
        } catch (e) {
          last = e;
          await new Promise((r) => setTimeout(r, 500 * (i + 1)));
        }
      }
      throw last;
    };
    (async () => {
      try {
        const pack = await packedClip(clipKey);
        const packedTs = pack ? await packedFrames(clipKey) : null;
        let m: ClipMeta;
        let l: Record<string, LabelRecord>;
        let usedPack = false;
        // Network first even when navigator.onLine claims offline (it lies
        // under some VPN/adapter setups, and a genuinely offline fetch
        // rejects immediately anyway). The pack is strictly a fallback.
        try {
          m = await retry(() => provider.getMeta(clipKey));
          l = await retry(() => provider.getLabels(clipKey));
        } catch (e) {
          if (!pack) throw e;
          m = pack.meta;
          l = pack.labels;
          usedPack = true;
          // online-but-failed -> the pack may be stale: shout.
          if (!cancelled && navigator.onLine !== false) setPackStale(true);
        }
        // backend.list already falls back to the pack's audit snapshot and
        // merges the still-queued offline audits
        const a = await backend.list(clipKey);
        // Converge the pack to what the network just served, so the next
        // fallback shows current labels, not download-day ones.
        if (pack && !usedPack) {
          void refreshPackedClip(clipKey, {
            title: m.title,
            meta: m,
            labels: l,
            audits: a,
          }).catch(() => {});
        }
        if (cancelled) return;
        setMeta(m);
        setLabels(l);
        setAudits(a);
        setPacked(packedTs);
        setFromPack(usedPack);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [clipKey, provider, backend]);

  const frames = useMemo(() => {
    const all = meta?.frames.filter((f) => f.fisheye) ?? [];
    return fromPack && packed ? all.filter((f) => packed.has(f.ts)) : all;
  }, [meta, fromPack, packed]);

  // Thermal assist rendering: bearing-aligned warp of the 160x120 Lepton frame.
  useEffect(() => {
    const cv = thermalCanvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!thermalAssist || !meta) return;
    const ts = frames[index]?.ts;
    if (!ts) return;
    const [w, h] = meta.image_size;
    cv.width = w; cv.height = h;
    const img = new Image();
    // no crossOrigin: the signed asset URL carries no ACAO header (2026-08-28 gotcha);
    // a tainted canvas is fine here because nothing reads pixels back
    img.onload = () => {
      if (!thermalCanvasRef.current) return;
      const c = thermalCanvasRef.current.getContext("2d");
      if (!c) return;
      c.clearRect(0, 0, w, h);
      // fisheye pinhole at native 864x648 (lib/viewer/radarProjection.ts) scaled to the bundle frame
      const sc = w / 864, fx = 418.51 * sc, cx = 444.11 * sc, cy = 300.5 * sc;
      const T_CX = 77, T_PXDEG = 2.82, TW = img.naturalWidth || 160, TH = img.naturalHeight || 120;
      const tScale = TW / 160;
      const pxPerDegF = fx * Math.PI / 180;            // fisheye px per degree near the axis
      const bandH = TH / tScale * (pxPerDegF / T_PXDEG); // thermal rows in fisheye px
      const y0 = cy - bandH / 2;
      c.globalAlpha = 0.6;
      for (let x = 0; x < w; x++) {
        const bearing = Math.atan((x + 0.5 - cx) / fx) * 180 / Math.PI;
        const xt = (T_CX + T_PXDEG * bearing) * tScale;
        if (xt < 0 || xt >= TW) continue;
        c.drawImage(img, Math.floor(xt), 0, 1, TH, x, y0, 1, bandH);
      }
      c.globalAlpha = 1;
      c.strokeStyle = "rgba(255,255,255,0.7)"; c.lineWidth = 1;
      c.strokeRect(0.5, y0 + 0.5, w - 1, bandH - 1);
    };
    img.src = provider.frameUrl(clipKey, "thermal", ts);
  }, [thermalAssist, index, frames, meta, clipKey, provider]);

  // Plan mode: when arriving via /annotate/<clip>?plan=1, drive the frame
  // index from the stored plan and expose next/prev-in-plan navigation
  // (crossing clips navigates to the next clip's annotate page).
  // Render-time state adjustment (house pattern, mirrors seededTs):
  // read the ?plan flag + stored plan exactly once per clipKey.
  const [planKey, setPlanKey] = useState<string | null>(null);
  if (searchParams.has("plan") && planKey !== clipKey) {
    setPlanKey(clipKey);
    const stored = loadPlan();
    if (stored && stored.frames.some((f) => f.clipKey === clipKey)) {
      setPlan(stored);
      setPlanCursor(stored.cursor);
    }
  }

  const gotoPlan = useCallback(
    (cursor: number) => {
      if (!plan) return;
      const c = Math.max(0, Math.min(cursor, plan.frames.length - 1));
      const target = plan.frames[c]!;
      setPlanCursor(c);
      savePlanCursor(c);
      if (target.clipKey !== clipKey) {
        router.push(`/annotate/${target.clipKey}?plan=1`);
        return;
      }
      const idx = frames.findIndex((f) => f.ts === target.ts);
      if (idx >= 0) setIndex(idx);
    },
    [plan, clipKey, frames, router],
  );

  // Land on the plan's current frame once frames are loaded.
  // Land on the plan's current frame once frames exist (render-time
  // adjustment, one-shot per clip — same pattern).
  const [planLanded, setPlanLanded] = useState(false);
  if (plan && frames.length > 0 && !planLanded) {
    setPlanLanded(true);
    const target = plan.frames[planCursor];
    if (target && target.clipKey === clipKey) {
      const idx = frames.findIndex((f) => f.ts === target.ts);
      if (idx >= 0 && idx !== index) setIndex(idx);
    }
  }

  // Playback (space): step at the capture cadence (~3 fps).
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setIndex((i) => {
        if (i + 1 >= frames.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 333);
    return () => window.clearInterval(id);
  }, [playing, frames.length]);
  const frame = frames[index] ?? null;
  const ts = frame?.ts ?? "";
  const folded = useMemo(() => foldAudits(audits), [audits]);
  // Plan completion across ALL the plan's clips: other clips are counted
  // once per plan (one backend.list each); the current clip recounts live
  // from the folded audits so the bar moves as AuthorTwo works.
  const [otherClipDone, setOtherClipDone] = useState<Record<string, number>>({});
  useEffect(() => {
    if (!plan) return;
    let cancelled = false;
    (async () => {
      const others = [...new Set(plan.frames.map((f) => f.clipKey))].filter((k) => k !== clipKey);
      const out: Record<string, number> = {};
      for (const k of others) {
        try {
          const a = foldAudits(await backend.list(k));
          out[k] = plan.frames.filter((f) => f.clipKey === k && a.has(f.ts)).length;
        } catch {
          out[k] = 0;
        }
      }
      if (!cancelled) setOtherClipDone(out);
    })();
    return () => { cancelled = true; };
  }, [plan?.id, clipKey, backend]);
  const planDone = useMemo(() => {
    if (!plan) return 0;
    const here = plan.frames.filter((f) => f.clipKey === clipKey && folded.has(f.ts)).length;
    return here + Object.values(otherClipDone).reduce((a, b) => a + b, 0);
  }, [plan, clipKey, folded, otherClipDone]);
  const verdict = ts ? folded.get(ts)?.verdict ?? null : null;

  // Seed editable boxes from the audit log (if any) else the pseudo labels.
  // Render-time state adjustment (not an effect): re-seeds exactly when the
  // frame under edit changes.
  const [seededTs, setSeededTs] = useState<string | null>(null);
  const seedKey = ts ? `${ts}|${teacher}` : null;
  if (seedKey && seedKey !== seededTs) {
    const audited = folded.get(ts);
    const seed = audited
      ? audited.boxes
      : teacherSeed(labels[ts]?.fisheye_bboxes ?? []);
    setSeededTs(seedKey);
    setMultiSel([]);
    setBoxes(seed.map((b) => ({ ...b, xyxy: [...b.xyxy] as Box["xyxy"] })));
    setSelected(null);
    setDraft(null);
  }


  const pushUndo = useCallback(
    (current: Box[]) => {
      if (!ts) return;
      const stack = undoStack.current.get(ts) ?? [];
      stack.push(current.map((b) => ({ ...b, xyxy: [...b.xyxy] as Box["xyxy"] })));
      if (stack.length > 200) stack.shift();
      undoStack.current.set(ts, stack);
      redoStack.current.delete(ts); // a new action invalidates redo
    },
    [ts],
  );

  // Escape hatch: an audited frame seeds from its audit record, so a bad
  // audit (e.g. one saved while the labels fetch was down) buries the
  // teacher suggestions with no way back. This re-seeds the EDITOR from the
  // teacher labels (undoable; the audit log itself is untouched until the
  // next verdict is saved, which then supersedes the bad record).
  const reseedFromTeacher = useCallback(() => {
    if (!ts) return;
    pushUndo(boxes);
    const seed = teacherSeed(labels[ts]?.fisheye_bboxes ?? []);
    setBoxes(seed.map((b) => ({ ...b, xyxy: [...b.xyxy] as Box["xyxy"] })));
    setMultiSel([]);
    setSelected(null);
    setDraft(null);
  }, [ts, boxes, pushUndo, labels, teacherSeed]);

  const appendAudit = useCallback(
    async (v: AuditRecord["verdict"], recordBoxes: Box[], advance = true) => {
      if (!meta || !ts) return;
      const rec: AuditRecord = {
        id: crypto.randomUUID(),
        clip_key: clipKey,
        frame_ts: ts,
        // frame.chunk attributes concatenated-activity frames to their
        // actual capture chunk so exports round-trip to the repo files
        frame_id: frameIdFor(
          meta.scene,
          frame?.chunk ?? meta.triplet_ts,
          ts,
        ),
        verdict: v,
        boxes: recordBoxes,
        source: labels[ts]?.source ?? "manual",
        created_at: new Date().toISOString(),
      };
      setAudits((a) => [...a, rec]); // optimistic
      try {
        await backend.append(rec);
      } catch (e) {
        setError(`audit write failed: ${String(e)}`);
      }
      // In plan mode gotoPlan owns navigation; this late advance (it runs
      // after the awaited write) used to bump one frame PAST the plan target.
      if (advance) setIndex((i) => Math.min(i + 1, frames.length - 1));
    },
    [meta, ts, frame, clipKey, labels, backend, frames.length],
  );

  // pointer helpers -----------------------------------------------------
  const toNorm = useCallback((e: { clientX: number; clientY: number }) => {
    const el = stageRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return [
      (e.clientX - rect.left) / rect.width,
      (e.clientY - rect.top) / rect.height,
    ] as [number, number];
  }, []);

  function onPointerDown(e: React.PointerEvent) {
    const pt = toNorm(e);
    if (!pt) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const [x, y] = pt;
    if (centroidMode && selected != null) {
      const b = boxes[selected];
      if (b) {
        const [bx0, by0, bx1, by1] = normaliseBox(b.xyxy);
        // clamp the click into the selected box: a centroid lives inside
        const cx = Math.min(Math.max(x, bx0), bx1);
        const cy = Math.min(Math.max(y, by0), by1);
        pushUndo(boxes);
        setBoxes((bs) => bs.map((bb, i) =>
          i === selected ? { ...bb, centroid: [cx, cy] as [number, number] } : bb));
      }
      setCentroidMode(false);
      return;
    }
    if (drawMode) {
      setDrag({ kind: "draw", startX: x, startY: y });
      setDraft({ cls, xyxy: [x, y, x, y] });
      return;
    }
    // boxes of the other image space are parked off-stage so they can be
    // neither hit nor selected while indices stay aligned with `boxes`
    const hit = hitTestPrioritised(
      boxes.map((b) => (inSpace(b) ? b : { ...b, xyxy: [-9, -9, -9, -9] as Box["xyxy"] })),
      selected, x, y, 0.02);
    if (!hit) {
      setSelected(null);
      setMultiSel([]);
      setDrag({ kind: "marquee", startX: x, startY: y });
      setMarquee([x, y, x, y]);
      return;
    }
    setMultiSel([]);
    setSelected(hit.index);
    pushUndo(boxes);
    setDrag(
      hit.corner
        ? { kind: "resize", index: hit.index, corner: hit.corner }
        : { kind: "move", index: hit.index, lastX: x, lastY: y },
    );
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!drag) return;
    const pt = toNorm(e);
    if (!pt) return;
    const [x, y] = pt;
    if (drag.kind === "draw") {
      setDraft({ cls, xyxy: [drag.startX, drag.startY, x, y] });
    } else if (drag.kind === "marquee") {
      setMarquee([drag.startX, drag.startY, x, y]);
    } else if (drag.kind === "move") {
      setBoxes((bs) =>
        bs.map((b, i) =>
          i === drag.index
            ? { ...b, xyxy: moveBox(b.xyxy, x - drag.lastX, y - drag.lastY) }
            : b,
        ),
      );
      setDrag({ ...drag, lastX: x, lastY: y });
    } else {
      setBoxes((bs) =>
        bs.map((b, i) =>
          i === drag.index
            ? { ...b, xyxy: resizeBox(b.xyxy, drag.corner, x, y) }
            : b,
        ),
      );
    }
  }

  function onPointerUp() {
    if (drag?.kind === "marquee" && marquee) {
      const [mx0, my0, mx1, my1] = marquee;
      // a tiny drag is just a deselect click, not a marquee
      if (Math.abs(mx1 - mx0) > 0.01 && Math.abs(my1 - my0) > 0.01) {
        setMultiSel(boxesInMarquee(
          boxes.map((b) => (inSpace(b) ? b : { ...b, xyxy: [-9, -9, -9, -9] as Box["xyxy"] })), marquee));
      }
      setMarquee(null);
      setDrag(null);
      return;
    }
    if (drag?.kind === "draw" && draft && meta) {
      const [w, h] = meta.image_size;
      const xyxy = normaliseBox(draft.xyxy);
      // 2 native px minimum: distant ducks/buoys are genuinely 2-4 px, and a
      // drag below that is indistinguishable from a click. (Was 5 px, which
      // silently swallowed deliberate small boxes.)
      if (isValidBox(xyxy, 2 / w, 2 / h)) {
        pushUndo(boxes);
        setBoxes((bs) => [...bs, { cls: draft.cls, xyxy, source: "dashboard-web",
          ...(curSpace === "thermal" ? { space: "thermal" as const } : {}) }]);
        // draw mode is sticky: stay in it for the next box, and do NOT
        // auto-select the committed box — class keys then set the class for
        // the NEXT boxes instead of silently retyping the last one.
      }
      setDraft(null);
    }
    setDrag(null);
  }

  // keyboard ------------------------------------------------------------
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
      const undo = () => {
        const stack = undoStack.current.get(ts);
        const prev = stack?.pop();
        if (prev) {
          const r = redoStack.current.get(ts) ?? [];
          r.push(boxes.map((b) => ({ ...b, xyxy: [...b.xyxy] as Box["xyxy"] })));
          redoStack.current.set(ts, r);
          setBoxes(prev);
        }
      };
      if (mod && (e.key === "z" || e.key === "Z")) {
        // Cmd/Ctrl+Z undo, +Shift redo — parity with the Tkinter dashboard
        // (which bound <Command-z>/<Control-z>), plus the redo it lacked.
        e.preventDefault();
        if (e.shiftKey) {
          const r = redoStack.current.get(ts);
          const next = r?.pop();
          if (next) {
            const stack = undoStack.current.get(ts) ?? [];
            stack.push(boxes.map((b) => ({ ...b, xyxy: [...b.xyxy] as Box["xyxy"] })));
            undoStack.current.set(ts, stack);
            setBoxes(next);
          }
        } else {
          undo();
        }
      } else if (mod && (e.key === "a" || e.key === "A")) {
        // cmd/ctrl+A: select every box on the frame in the current image space
        e.preventDefault();
        setSelected(null);
        setMultiSel(boxes.map((b, i) => (inSpace(b) ? i : -1)).filter((i) => i >= 0));
      } else if (e.key === "X") {
        // shift+X: delete every box on the frame in the current image space (one undo step)
        if (boxes.some(inSpace)) {
          pushUndo(boxes);
          setBoxes((bs) => bs.filter((b) => !inSpace(b)));
          setMultiSel([]);
          setSelected(null);
        }
      } else if (e.key === "ArrowLeft" && e.shiftKey) {
        e.preventDefault(); // ±5 s clip jump, as the old dashboard (15 frames @3fps)
        setIndex((i) => Math.max(i - 15, 0));
      } else if (e.key === "ArrowRight" && e.shiftKey) {
        e.preventDefault();
        setIndex((i) => Math.min(i + 15, frames.length - 1));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        // Plan mode: arrows walk the PLAN (the sprint flow); Alt+arrow
        // steps single clip frames for context.
        if (plan && planActive && !e.altKey) gotoPlan(planCursor - 1);
        else setIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "ArrowRight" || e.key === "n") {
        e.preventDefault();
        if (plan && planActive && !e.altKey) gotoPlan(planCursor + 1);
        else setIndex((i) => Math.min(i + 1, frames.length - 1));
      } else if (e.key === " ") {
        e.preventDefault(); // space play/pause, as the old dashboard
        setPlaying((pl) => !pl);
      } else if (e.key === "Escape") {
        // cancel in-progress draw, else deselect (old tool used Escape to
        // leave; on the web, backing out is the browser's job)
        if (draft || drawMode) {
          setDraft(null);
          setDrawMode(false);
        } else {
          setSelected(null);
          setMultiSel([]);
        }
      } else if (e.key === "a") {
        void appendAudit("edit", boxes, !(plan && planActive));
        if (plan && planActive) gotoPlan(planCursor + 1); // accept advances
      } else if (e.key === "r") {
        void appendAudit("reject", [], !(plan && planActive));
        if (plan && planActive) gotoPlan(planCursor + 1);
      } else if (e.key === "h") {
        setNightBoost((v) => !v);         // h = "high gain": plain key, no browser conflict (b is the boat class)
      } else if (e.key === "t") {
        // plain t: cmd+T is the browser's new-tab shortcut
        setThermalMode((m) => (m === "off" ? "assist" : m === "assist" ? "frame" : "off"));
      } else if (e.key === "c") {
        // centroid placement for the selected box (mask-annotation seed)
        if (selected != null) setCentroidMode((m) => !m);
      } else if (e.key === "x" || e.key === "Delete" || e.key === "Backspace") {
        if (multiSel.length > 0) {
          pushUndo(boxes);
          setBoxes((bs) => bs.filter((_, i) => !multiSel.includes(i)));
          setMultiSel([]);
          setSelected(null);
        } else if (selected != null) {
          pushUndo(boxes);
          setBoxes((bs) => bs.filter((_, i) => i !== selected));
          setSelected(null);
        }
      } else if (e.key === "u") {
        undo();
      } else if (e.key === "e" || e.key === "d") {
        setDrawMode((d) => !d);
      } else if (e.key === "]" && plan) {
        gotoPlan(planCursor + 1);
      } else if (e.key === "[" && plan) {
        gotoPlan(planCursor - 1);
      } else if (e.key === "Escape" && centroidMode) {
        setCentroidMode(false);
      } else if (e.key === "s") {
        // `s` = structure while labelling, as the old dashboard
        setCls("structure");
        if (multiSel.length > 0) {
          pushUndo(boxes);
          setBoxes((bs) => bs.map((b, i) => (multiSel.includes(i) ? { ...b, cls: "structure" } : b)));
        } else if (selected != null) {
          pushUndo(boxes);
          setBoxes((bs) =>
            bs.map((b, i) => (i === selected ? { ...b, cls: "structure" } : b)),
          );
        }
      } else if (e.key in CLASS_KEYS || e.key === "D") {
        const next = e.key === "D" ? CLASS_KEYS["d"]! : CLASS_KEYS[e.key]!;
        setCls(next);
        if (multiSel.length > 0) {
          // marquee group retype: the whole cluster takes the class
          pushUndo(boxes);
          setBoxes((bs) => bs.map((b, i) => (multiSel.includes(i) ? { ...b, cls: next } : b)));
        } else if (selected != null) {
          pushUndo(boxes);
          setBoxes((bs) =>
            bs.map((b, i) => (i === selected ? { ...b, cls: next } : b)),
          );
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [boxes, selected, multiSel, ts, frames.length, appendAudit, pushUndo, draft, inSpace,
      drawMode, plan, planActive, planCursor, gotoPlan, centroidMode]);

  async function exportJsonl() {
    if (!meta) return;
    const [w, h] = meta.native_size;
    let jsonl: string;
    let name: string;
    if (plan) {
      // Plan session: gather the audits of EVERY clip the plan touches,
      // so one file carries the whole session regardless of where the
      // export button was pressed. Named after the plan, not the clip.
      const keys = [...new Set(plan.frames.map((f) => f.clipKey))];
      const parts: string[] = [];
      for (const key of keys) {
        const clipAudits = key === clipKey ? audits : await backend.list(key);
        const part = auditsToJsonl(clipAudits, { width: w, height: h });
        if (part.trim()) parts.push(part.trim());
      }
      jsonl = parts.join("\n") + "\n";
      name = `audits_plan_${plan.createdAt.slice(0, 10)}_${plan.frames.length}frames.jsonl`;
    } else {
      jsonl = auditsToJsonl(audits, { width: w, height: h });
      name = `audits_${clipKey}.jsonl`;
    }
    const blob = new Blob([jsonl], { type: "application/jsonl" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (error) {
    return (
      <div className="p-8 font-mono text-sm text-status-serious">{error}</div>
    );
  }
  if (!meta || !frame) {
    return <div className="p-8 font-mono text-sm text-subtle">loading…</div>;
  }

  const audited = folded.size;
  const all = draft ? [...boxes, draft] : boxes;

  return (
    <div className="flex h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border bg-surface-1 px-4 py-2">
          <Link
            href={`/clips/${clipKey}`}
            className="flex items-center gap-1 text-xs text-muted hover:text-foreground"
          >
            <ArrowLeft size={14} aria-hidden /> viewer
          </Link>
          <div className="h-4 w-px bg-border" aria-hidden />
          <h1 className="truncate text-sm font-semibold tracking-tight">
            Annotate · {meta.title}
          </h1>
          {fromPack && packStale && (
            <span
              data-pack-stale
              className="rounded-sm border border-status-serious bg-status-serious/10 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-status-serious"
              title="You are online but the network labels failed to load, so this clip is showing an OLDER offline pack — teacher boxes/masks may be missing or outdated. Reload to retry; rebuild the pack on /offline to refresh it."
            >
              stale pack — reload
            </span>
          )}
          {fromPack && (
            <span
              data-from-pack
              className="rounded-sm border border-status-warn px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-status-warn"
              title="Reading this clip from an offline pack: only packed frames are shown; audits queue locally until synced."
            >
              offline pack · {frames.length} frames
            </span>
          )}
          {!fromPack && packed && packed.size > 0 && (
            <span
              data-packed-count
              className="rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-subtle"
              title="Frames of this clip held in an offline pack in this browser"
            >
              {packed.size} packed
            </span>
          )}
          {plan && (
            <button
              type="button"
              onClick={() => {
                setPlanActive(true);
                gotoPlan(planCursor);
              }}
              aria-pressed={planActive}
              title="Navigate the plan (arrows walk the planned frames)"
              className={`flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[11px] tabular-nums ${planActive ? "border-seeblau-100/60 bg-seeblau-100/10 text-seeblau-deep" : "border-border text-subtle hover:text-foreground"}`}
            >
              plan {planCursor + 1}/{plan.frames.length}
              <span aria-label="previous plan frame" role="button"
                onPointerDown={(e) => { e.stopPropagation(); gotoPlan(planCursor - 1); }}
                className="hover:text-foreground">[</span>
              <span aria-label="next plan frame" role="button"
                onPointerDown={(e) => { e.stopPropagation(); gotoPlan(planCursor + 1); }}
                className="hover:text-foreground">]</span>
            </button>
          )}
          {verdict && (
            <span
              data-audited-check
              title={`This frame has an audit record (latest verdict: ${verdict})`}
              className={`rounded-sm border px-1.5 py-0.5 font-mono text-[11px] ${verdict === "reject" ? "border-status-serious text-status-serious" : "border-status-good text-status-good"}`}
            >
              ✓
            </span>
          )}
          <input
            data-goto-frame
            type="text"
            inputMode="numeric"
            placeholder={plan && planActive ? `#/${plan.frames.length}` : `#/${frames.length}`}
            title="Type a frame number and press Enter to jump (plan frames in plan mode, clip frames otherwise)"
            className="w-16 rounded-sm border border-border bg-surface-2 px-1.5 py-0.5 text-center font-mono text-[11px] tabular-nums placeholder:text-subtle focus:border-seeblau-100 focus:outline-none"
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              const el = e.currentTarget;
              const n = parseInt(el.value, 10);
              if (!Number.isFinite(n)) return;
              if (plan && planActive) gotoPlan(n - 1);
              else setIndex(Math.max(0, Math.min(n - 1, frames.length - 1)));
              el.value = "";
              el.blur();
            }}
          />
          <span className="ml-auto flex items-center gap-1" title="Which teacher's suggestions seed unaudited frames (union label sets carry per-box source)">
            {(["union", "grounding-dino", "dart-sam3", "both"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTeacher(t)}
                aria-pressed={teacher === t}
                className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] ${teacher === t ? "border-seeblau-100/60 bg-seeblau-100/10 text-seeblau-deep" : "border-border text-subtle hover:text-foreground"}`}
              >
                {t === "union" ? "all" : t === "grounding-dino" ? "DINO" : t === "dart-sam3" ? "DART" : "agreed"}
              </button>
            ))}
          </span>
          <button
            type="button"
            onClick={reseedFromTeacher}
            title="Replace the boxes in the editor with the teacher suggestions for this frame (use when an earlier audit hid them; undoable with Cmd+Z; save a verdict to make it stick)"
            className="rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] text-subtle hover:text-foreground"
          >
            reseed
          </button>
          <button
            type="button"
            onClick={() => setShowMasks((m) => !m)}
            aria-pressed={showMasks}
            title="Show/hide the teacher's SAM3 mask outlines under the suggestion boxes"
            className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] ${showMasks ? "border-seeblau-100/60 bg-seeblau-100/10 text-seeblau-deep" : "border-border text-subtle hover:text-foreground"}`}
          >
            masks
          </button>
          <button
            type="button"
            onClick={() => setPlanActive(false)}
            aria-pressed={plan ? !planActive : true}
            title="Navigate this clip's full frame set (arrows step frames)"
            className={`rounded-sm border px-2 py-0.5 font-mono text-xs tabular-nums ${plan && !planActive ? "border-seeblau-100/60 bg-seeblau-100/10 text-seeblau-deep" : plan ? "border-border text-subtle hover:text-foreground" : "border-transparent text-muted"}`}
          >
            audited {audited}/{frames.length}
          </button>
          <button
            type="button"
            onClick={() => void exportJsonl()}
            className="flex items-center gap-1.5 rounded-sm border border-border px-2 py-1 text-xs text-muted hover:border-border-strong hover:text-foreground"
          >
            <Download size={13} aria-hidden />
            {plan ? "export plan JSONL" : "export JSONL"}
          </button>
          <span className="group relative">
            <button
              type="button"
              aria-label="About the exported JSONL"
              className="flex h-6 w-6 items-center justify-center rounded-full border border-border text-subtle hover:border-border-strong hover:text-foreground"
            >
              <Info size={13} aria-hidden />
            </button>
            <span className="pointer-events-none absolute right-0 top-full z-20 mt-1 hidden w-80 rounded-sm border border-border bg-surface-1 p-3 text-left text-[11px] leading-snug text-muted shadow-sm group-hover:block group-focus-within:block">
              <b className="text-foreground">The exported file is the repo&apos;s
              training-frames JSONL:</b> one line per audited frame with your
              verdict and corrected boxes.
              <br /><br />
              <b className="text-foreground">What to do with it:</b> drop it
              into the repo as{" "}
              <code className="bg-surface-3 px-1">labels/training_frames.jsonl</code>{" "}
              (or merge if one exists). It then drives the audited scorer
              retrain and the fine-tune export.
              <br /><br />
              Exporting is a snapshot convenience: every audit is already
              saved to the shared database as you work, so nothing is lost if
              you never export.{plan ? " In a plan session the file gathers every clip the plan touches." : ""}
            </span>
          </span>
        </div>

        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-surface-2 p-4">
          <div
            ref={stageRef}
            className="relative touch-none select-none"
            style={{ aspectRatio: `${meta.image_size[0]} / ${meta.image_size[1]}`, width: "min(100%, 864px)" }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            {thermalMode === "frame" ? (
              <img
                src={provider.frameUrl(clipKey, thermalHdFailed.has(ts) ? "thermal" : "thermal_hd", ts)}
                onError={() => setThermalHdFailed((f) => (f.has(ts) ? f : new Set(f).add(ts)))}
                alt={`Thermal frame ${ts}`}
                className="h-full w-full"
                style={{ imageRendering: "pixelated" }}
                draggable={false}
              />
            ) : (
              <img
                src={provider.frameUrl(clipKey, hdFailed.has(ts) ? "frames" : "frames_hd", ts)}
                onError={() => setHdFailed((f) => (f.has(ts) ? f : new Set(f).add(ts)))}
                alt={`Frame ${ts}`}
                className="h-full w-full"
                style={nightBoost ? { filter: "brightness(3.2) contrast(1.35) saturate(0.8)" } : undefined}
                draggable={false}
              />
            )}
            {thermalMode === "frame" && (
              <span className="pointer-events-none absolute left-2 top-2 rounded bg-black/70 px-2 py-0.5 font-mono text-[11px] text-white">
                THERMAL FRAME · boxes saved in thermal space
              </span>
            )}
            {thermalAssist && (
              <canvas
                ref={thermalCanvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
                aria-label="thermal frame warped onto the fisheye by bearing"
              />
            )}
            <svg
              className="absolute inset-0 h-full w-full"
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
            >
              {all.map((box, i) => {
                if (!inSpace(box)) return null;   // other image space: hidden, index kept
                const [x0, y0, x1, y1] = normaliseBox(box.xyxy);
                const colour = colourForClass(box.cls);
                const isSel = i === selected || multiSel.includes(i);
                return (
                  <g key={i}>
                    {showMasks && box.polygon && box.polygon.length >= 6 && (
                      <polygon
                        points={Array.from(
                          { length: Math.floor(box.polygon.length / 2) },
                          (_, k) => `${box.polygon![2 * k]},${box.polygon![2 * k + 1]}`,
                        ).join(" ")}
                        fill={colour}
                        fillOpacity={0.08}
                        stroke={colour}
                        strokeOpacity={0.55}
                        strokeWidth={1}
                        strokeDasharray="4 3"
                        vectorEffect="non-scaling-stroke"
                        pointerEvents="none"
                      />
                    )}
                    <rect
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill={isSel ? colour : "none"}
                      fillOpacity={isSel ? 0.12 : 0}
                      stroke={colour}
                      strokeWidth={isSel ? 3 : 2}
                      vectorEffect="non-scaling-stroke"
                    />
                    {box.centroid && (
                      <g pointerEvents="none">
                        <circle cx={box.centroid[0]} cy={box.centroid[1]} r={0.006}
                                fill={colour} stroke="#fff" strokeWidth={1}
                                vectorEffect="non-scaling-stroke" />
                        <line x1={box.centroid[0] - 0.012} y1={box.centroid[1]}
                              x2={box.centroid[0] + 0.012} y2={box.centroid[1]}
                              stroke={colour} strokeWidth={1} vectorEffect="non-scaling-stroke" />
                        <line x1={box.centroid[0]} y1={box.centroid[1] - 0.012}
                              x2={box.centroid[0]} y2={box.centroid[1] + 0.012}
                              stroke={colour} strokeWidth={1} vectorEffect="non-scaling-stroke" />
                      </g>
                    )}
                    {isSel &&
                      (
                        [
                          [x0, y0],
                          [x1, y0],
                          [x0, y1],
                          [x1, y1],
                        ] as const
                      ).map(([cx, cy], k) => (
                        <rect
                          key={k}
                          x={cx - 0.01}
                          y={cy - 0.01}
                          width={0.02}
                          height={0.02}
                          fill={colour}
                        />
                      ))}
                  </g>
                );
              })}
            </svg>
            {/* class tags rendered in HTML so text keeps aspect */}
            {all.map((box, i) => {
              const [x0, y0] = normaliseBox(box.xyxy);
              return (
                <span
                  key={i}
                  role="button"
                  tabIndex={0}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    setSelected(i);
                  }}
                  className="absolute cursor-pointer rounded-sm px-1 font-mono text-[10px] text-white"
                  style={{
                    left: `${x0 * 100}%`,
                    top: `calc(${y0 * 100}% - 16px)`,
                    background: "rgba(10,42,58,0.8)",
                  }}
                >
                  {box.cls}
                </span>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 border-t border-border bg-surface-1 px-3 py-2">
          <button
            type="button"
            onClick={() => {
              void appendAudit("edit", boxes, !(plan && planActive));
              if (plan && planActive) gotoPlan(planCursor + 1);
            }}
            className="flex items-center gap-1.5 rounded-sm bg-status-good px-3 py-1.5 text-xs font-medium text-white hover:opacity-90"
          >
            <Check size={14} aria-hidden /> accept (a)
          </button>
          <button
            type="button"
            onClick={() => {
              void appendAudit("reject", [], !(plan && planActive));
              if (plan && planActive) gotoPlan(planCursor + 1);
            }}
            className="flex items-center gap-1.5 rounded-sm border border-status-serious px-3 py-1.5 text-xs font-medium text-status-serious hover:bg-status-serious/10"
          >
            <X size={14} aria-hidden /> reject (r)
          </button>
          <button
            type="button"
            onClick={() => setDrawMode((d) => !d)}
            aria-pressed={drawMode}
            className={
              drawMode
                ? "flex items-center gap-1.5 rounded-sm border border-accent bg-surface-2 px-3 py-1.5 text-xs text-accent-strong"
                : "flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-border-strong"
            }
          >
            <Pencil size={14} aria-hidden /> draw (d)
          </button>
          <button
            type="button"
            disabled={selected == null}
            onClick={() => {
              if (selected != null) {
                pushUndo(boxes);
                setBoxes((bs) => bs.filter((_, i) => i !== selected));
                setSelected(null);
              }
            }}
            className="flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-border-strong disabled:opacity-40"
          >
            <Trash2 size={14} aria-hidden /> delete (x)
          </button>

          <div className="ml-auto flex items-center gap-1">
            {LABEL_CLASSES.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => {
                  setCls(c);
                  if (selected != null) {
                    setBoxes((bs) =>
                      bs.map((b, i) => (i === selected ? { ...b, cls: c } : b)),
                    );
                  }
                }}
                aria-pressed={cls === c}
                className={
                  cls === c
                    ? "rounded-sm border px-2 py-1 font-mono text-[11px]"
                    : "rounded-sm border border-transparent px-2 py-1 font-mono text-[11px] text-muted hover:border-border"
                }
                style={
                  cls === c
                    ? { borderColor: colourForClass(c), color: colourForClass(c) }
                    : undefined
                }
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3 border-t border-border bg-surface-1 px-3 py-2">
          <input
            type="range"
            aria-label="Frame"
            min={0}
            max={Math.max(frames.length - 1, 0)}
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
            className="min-w-0 flex-1 accent-(--seeblau-100)"
          />
          <span className="font-mono text-xs tabular-nums text-muted">
            {index + 1}/{frames.length}
          </span>
          <span className="w-24 text-right font-mono text-xs tabular-nums">
            {ts}
          </span>
          {verdict && (
            <span
              className={
                verdict === "reject"
                  ? "rounded-sm border border-status-serious px-1.5 py-0.5 font-mono text-[10px] uppercase text-status-serious"
                  : "rounded-sm border border-status-good px-1.5 py-0.5 font-mono text-[10px] uppercase text-status-good"
              }
            >
              {verdict}
            </span>
          )}
        </div>
        {plan && (
          <div
            data-plan-progress
            className="flex items-center gap-2 border-t border-border bg-surface-1 px-4 py-1"
            title="Plan completion: planned frames with an audit record, across every clip in the plan"
          >
            <div className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full bg-status-good transition-[width]"
                style={{ width: `${Math.round((100 * planDone) / Math.max(1, plan.frames.length))}%` }}
              />
            </div>
            <span className="font-mono text-[10px] tabular-nums text-subtle">
              {planDone}/{plan.frames.length} audited
            </span>
          </div>
        )}
      </div>

      <aside className="w-72 shrink-0 space-y-4 overflow-auto border-l border-border bg-surface-1 p-3 text-xs">
        <section>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
            Boxes on this frame
          </h2>
          <ul className="space-y-0.5 font-mono text-[11px]">
            {boxes.length === 0 && (
              <li className="text-subtle">none — draw with d</li>
            )}
            {boxes.map((b, i) => (
              <li key={i}>
                <button
                  type="button"
                  onClick={() => setSelected(i)}
                  className={
                    i === selected
                      ? "flex w-full items-center gap-2 rounded-sm bg-surface-3 px-1.5 py-1 text-left"
                      : "flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left hover:bg-surface-2"
                  }
                >
                  <span
                    aria-hidden
                    className="h-2 w-2 rounded-full"
                    style={{ background: colourForClass(b.cls) }}
                  />
                  {b.cls}
                  {b.confidence != null && (
                    <span className="ml-auto tabular-nums text-subtle">
                      {b.confidence.toFixed(2)}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
            Keys
          </h2>
          <dl className="grid grid-cols-[2rem_1fr] gap-y-1 text-muted">
            <dt className="font-mono">a</dt>
            <dd>accept frame (boxes as shown)</dd>
            <dt className="font-mono">r</dt>
            <dd>reject frame (no obstacles)</dd>
            <dt className="font-mono">d</dt>
            <dd>draw a new box</dd>
            <dt className="font-mono">⌘A / ⇧X</dt>
            <dd>select all boxes on the frame / delete all boxes on the frame (current image space; one undo step)</dd>
            <dt className="font-mono">x</dt>
            <dd>delete selected box</dd>
            <dt className="font-mono">u / ⌘Z</dt>
            <dd>undo (this frame) · ⇧⌘Z redo</dd>
            <dt className="font-mono">h</dt>
            <dd>night boost (display-only brightness for dark frames)</dd>
            <dt className="font-mono">t</dt>
            <dd>thermal, cycles: assist (the Lepton frame warped onto the fisheye by bearing, as a band at the horizon) → frame (pure-thermal annotation: the Lepton frame is the stage, boxes are saved in thermal space and hidden from the fisheye view) → off</dd>
            <dt className="font-mono">c</dt>
            <dd>
              place a pseudo-centroid: with a box selected, press c then
              click where the object actually sits inside the box — a
              crosshair marks it and it exports with the box as a
              point-prompt seed for instance masks
            </dd>
            <dt className="font-mono">s</dt>
            <dd>class: structure (while labelling)</dd>
            <dt className="font-mono">b D B p m o</dt>
            <dd>class: boat duck buoy person structure other</dd>
            <dt className="font-mono">←/→</dt>
            <dd>previous / next frame (walks the plan in a plan session; ⌥←→ steps frames, ⇧←→ jumps ±5 s)</dd>
            <dt className="font-mono">space</dt>
            <dd>play / pause at capture speed</dd>
          </dl>
          <p className="mt-3 leading-relaxed text-subtle">
            {backend.mode === "local"
              ? "Audit log is stored in this browser (no Supabase configured). Export JSONL to keep it."
              : "Audit log appends to Supabase (sail_audits)."}
          </p>
        </section>
      </aside>
    </div>
  );
}
