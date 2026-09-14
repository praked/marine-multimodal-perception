"use client";

import { selectTrail } from "@/lib/viewer/trail";
import { BinTable } from "@/components/viewer/BinTable";
import { CutsPanel } from "@/components/viewer/CutsPanel";
import { HealthBar } from "@/components/viewer/HealthBar";
import {
  fisheyeHealth,
  radarHealth,
  runsFromBooleans,
  runsFromStates,
  thermalHealth,
} from "@/lib/health";
import { sectorEdges, sectorView } from "@/lib/sectors";
import { FisheyePanel } from "@/components/viewer/FisheyePanel";
import { HeadingPanel } from "@/components/viewer/HeadingPanel";
import { LayerPanel } from "@/components/viewer/LayerPanel";
import { RadarPanel } from "@/components/viewer/RadarPanel";
import { ScorerPanel } from "@/components/viewer/ScorerPanel";
import { TargetChips } from "@/components/viewer/TargetChips";
import { ThermalPanel } from "@/components/viewer/ThermalPanel";
import { Transport, type CutControls } from "@/components/viewer/Transport";
import {
  cutRuns,
  cutStats,
  daysRemaining,
  getCurationBackend,
  inCut,
  isDeleted,
  loadCurationSafe,
  nextKeptIndex,
  normaliseCuts,
  type Cut,
  type CurationRecord,
} from "@/lib/curation";
import { getProvider } from "@/lib/data";
import { formatActivityRange } from "@/lib/format";
import { useViewerStore } from "@/lib/stores/viewer";
import { FrameAssetLoader, type FrameAssets } from "@/lib/viewer/frameAssets";
import type {
  Box,
  ClipMeta,
  Instance,
  LabelRecord,
  RadarFrames,
  SectorRecord,
} from "@/lib/types";
import { ArrowLeft, RotateCcw, Scissors, SquarePen } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const BASE_INTERVAL_MS = 333; // footage is ~3 fps at capture
const TRAIL_GROUPS = 8;

interface ClipData {
  meta: ClipMeta;
  sectors: Record<string, SectorRecord>;
  radar: RadarFrames;
  radarKeys: string[]; // time-ordered radar timestamps
  boxes: Record<string, Box[]>;
  instances: Record<string, Instance[]>;
  labels: Record<string, LabelRecord>;
  curation: CurationRecord | null;
}

function Panel({
  title,
  meta,
  footer,
  children,
}: {
  title: string;
  meta?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="flex min-h-0 flex-col bg-surface-1">
      <header className="flex items-baseline justify-between gap-2 border-b border-border px-3 py-1.5">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
          {title}
        </h2>
        {meta && (
          <span className="font-mono text-[10px] tabular-nums text-subtle">
            {meta}
          </span>
        )}
      </header>
      <div className="min-h-0 flex-1 p-1.5">{children}</div>
      {footer}
    </section>
  );
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return `${String(d.getDate()).padStart(2, "0")}-${String(d.getMonth() + 1).padStart(2, "0")}-${d.getFullYear()}`;
}

export function ClipViewer({ clipKey }: { clipKey: string }) {
  const [data, setData] = useState<ClipData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const layers = useViewerStore((s) => s.layers);
  const threshold = useViewerStore((s) => s.threshold);
  const scoreSource = useViewerStore((s) => s.scoreSource);
  const speed = useViewerStore((s) => s.speed);
  const showCutFrames = useViewerStore((s) => s.showCutFrames);
  const setShowCutFrames = useViewerStore((s) => s.setShowCutFrames);
  const provider = getProvider();

  // Curation (cuts + deleted state) for this set. `cuts` is the live,
  // editable copy; the record is what the store last confirmed.
  const [record, setRecord] = useState<CurationRecord | null>(null);
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [inIndex, setInIndex] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [curationError, setCurationError] = useState<string | null>(null);
  const [loadedAt, setLoadedAt] = useState(0); // wall clock for "days until prune"

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const meta = await provider.getMeta(clipKey);
        const [sectors, radar, boxes, instances, labels, records] = await Promise.all([
          provider.getSectors(clipKey),
          provider.getRadar(clipKey),
          provider.getBoxes(clipKey),
          provider.getInstances(clipKey),
          provider.getLabels(clipKey),
          loadCurationSafe(),
        ]);
        if (cancelled) return;
        const radarKeys = Object.keys(radar).sort();
        const rec = records.get(clipKey) ?? null;
        setData({ meta, sectors, radar, radarKeys, boxes, instances, labels, curation: rec });
        setRecord(rec);
        setCuts(rec?.cuts ?? []);
        setLoadedAt(Date.now());
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [clipKey, provider]);

  const total = data?.meta.n_frames ?? 0;
  const frames = useMemo(() => data?.meta.frames ?? [], [data]);
  const skipCuts = cuts.length > 0 && !showCutFrames;

  /** Seek honouring the cut skip: lands on the nearest kept frame in `dir`
      (falls back to the other direction, then stays put). */
  const seekKept = useCallback(
    (i: number, dir: 1 | -1 = 1): number => {
      const n = frames.length;
      if (n === 0) return 0;
      const c = Math.min(Math.max(i, 0), n - 1);
      if (!skipCuts) return c;
      const k = nextKeptIndex(frames, cuts, c, dir);
      if (k >= 0) return k;
      const back = nextKeptIndex(frames, cuts, c, dir === 1 ? -1 : 1);
      return back >= 0 ? back : c;
    },
    [frames, cuts, skipCuts],
  );

  // Landing on a cut frame with skipping on (first load, a new cut around
  // the playhead, toggling the switch off) moves to the nearest kept one.
  // Render-time adjustment (house pattern): no effect, no extra paint.
  if (data && skipCuts && frames[index] && inCut(cuts, frames[index]!.ts)) {
    const k = seekKept(index, 1);
    if (k !== index) setIndex(k);
  }

  const frame = data?.meta.frames[index] ?? null;
  const ts = frame?.ts ?? "";

  // Shared per-clip asset loader: fisheye + thermal + mask fetched and
  // decoded together, committed to BOTH panels atomically (no drift).
  const loader = useMemo(() => {
    if (!data) return null;
    return new FrameAssetLoader(
      (kind, t) => provider.resolveAssetUrl(clipKey, kind, t),
      data.meta.image_size,
    );
  }, [data, clipKey, provider]);
  const [displayed, setDisplayed] = useState<{
    index: number;
    assets: FrameAssets;
  } | null>(null);
  const reqSeq = useRef(0);
  const shownSeq = useRef(0);

  useEffect(() => {
    if (!data || !loader || !frame) return;
    const seq = ++reqSeq.current;
    const at = index;
    loader.load(frame).then((assets) => {
      // monotonic commit: late loads still paint unless something newer has
      if (seq < shownSeq.current) return;
      shownSeq.current = seq;
      setDisplayed({ index: at, assets });
    });
    loader.prefetch(data.meta.frames, index, 4);
  }, [index, data, loader, frame]);

  // Playback: advance only when the NEXT frame's assets are decoded, so the
  // panels step in lockstep at whatever rate the network sustains (frame-
  // hold, never freeze-and-drift). Cut frames are skipped unless shown.
  const [buffering, setBuffering] = useState(false);
  useEffect(() => {
    if (!playing || total === 0 || !data || !loader) return;
    const id = setInterval(() => {
      setIndex((i) => {
        if (i + 1 >= total) {
          // end of clip: stop on the last frame (no wrap — the playhead used
          // to jump back to 0, which read as "playing past the end")
          setPlaying(false);
          setBuffering(false);
          return i;
        }
        let next = i + 1;
        if (skipCuts) {
          // wrap=false: at the last kept frame, playback STOPS (a wrap here
          // played "past the end of the bar" whenever the clip had an end cut)
          const k = nextKeptIndex(frames, cuts, next, 1, false);
          if (k < 0) {
            setPlaying(false);
            setBuffering(false);
            return i;
          }
          next = k;
        }
        const entry = data.meta.frames[next];
        if (!entry) return i;
        if (loader.isReady(entry.ts)) {
          setBuffering(false);
          return next;
        }
        void loader.load(entry); // not ready: keep loading, hold this frame
        setBuffering(true);
        return i;
      });
    }, BASE_INTERVAL_MS / speed);
    return () => {
      clearInterval(id);
      setBuffering(false);
    };
  }, [playing, speed, total, data, loader, skipCuts, frames, cuts]);

  const onSeek = useCallback(
    (i: number) => setIndex((cur) => seekKept(i, i >= cur ? 1 : -1)),
    [seekKept],
  );
  const onTogglePlay = useCallback(() => setPlaying((p) => !p), []);

  // ---- curation edits ----------------------------------------------------
  const persistCuts = useCallback(
    async (next: Cut[]) => {
      const norm = normaliseCuts(next);
      setCuts(norm);
      setSaving(true);
      setCurationError(null);
      try {
        const rec = await getCurationBackend().setCuts(clipKey, norm);
        setRecord(rec);
        setCuts(rec.cuts);
      } catch (e) {
        setCurationError(`cut save failed: ${String(e)}`);
      } finally {
        setSaving(false);
      }
    },
    [clipKey],
  );

  const markIn = useCallback(() => setInIndex(index), [index]);
  const markOut = useCallback(() => {
    if (!frames.length) return;
    const a = inIndex ?? index;
    const [s, e] = a <= index ? [a, index] : [index, a];
    const cut: Cut = { start_ts: frames[s]!.ts, end_ts: frames[e]!.ts };
    setInIndex(null);
    void persistCuts([...cuts, cut]);
  }, [frames, inIndex, index, cuts, persistCuts]);
  const clearIn = useCallback(() => setInIndex(null), []);

  const dragged = useRef<Cut[] | null>(null);
  const onDragEdge = useCallback(
    (cut: number, edge: "start" | "end", frameIndex: number) => {
      const f = frames[Math.min(Math.max(frameIndex, 0), frames.length - 1)];
      if (!f) return;
      setCuts((cs) => {
        const next = cs.map((c, i) =>
          i === cut ? { ...c, [edge === "start" ? "start_ts" : "end_ts"]: f.ts } : c,
        );
        dragged.current = next;
        return next;
      });
    },
    [frames],
  );
  const onDragEnd = useCallback(() => {
    if (dragged.current) {
      const next = dragged.current;
      dragged.current = null;
      void persistCuts(next);
    }
  }, [persistCuts]);

  const restore = useCallback(async () => {
    setSaving(true);
    try {
      const rec = await getCurationBackend().setDeleted(clipKey, false);
      setRecord(rec);
    } catch (e) {
      setCurationError(`restore failed: ${String(e)}`);
    } finally {
      setSaving(false);
    }
  }, [clipKey]);

  // keyboard transport
  const totalRef = useRef(total);
  useEffect(() => {
    totalRef.current = total;
  }, [total]);
  const seekRef = useRef(seekKept);
  useEffect(() => {
    seekRef.current = seekKept;
  }, [seekKept]);
  const markRef = useRef({ markIn, markOut });
  useEffect(() => {
    markRef.current = { markIn, markOut };
  }, [markIn, markOut]);
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
      if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIndex((i) => seekRef.current(i - (e.shiftKey ? 5 : 1), -1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setIndex((i) => seekRef.current(i + (e.shiftKey ? 5 : 1), 1));
      } else if (e.key === "Home") {
        setIndex(seekRef.current(0, 1));
      } else if (e.key === "End") {
        setIndex(seekRef.current(totalRef.current - 1, -1));
      } else if (e.key === "i") {
        markRef.current.markIn();
      } else if (e.key === "o") {
        markRef.current.markOut();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const record_: SectorRecord | null = (data && ts && data.sectors[ts]) || null;
  const sectorWidthDeg = useViewerStore((s) => s.sectorWidthDeg);
  const sv = useMemo(
    () => (record_ ? sectorView(record_, sectorWidthDeg) : null),
    [record_, sectorWidthDeg],
  );
  const spokeEdges = useMemo(
    () => (record_ ? sectorEdges(record_, sectorWidthDeg) : undefined),
    [record_, sectorWidthDeg],
  );

  // per-sensor health timelines (pastel bars under each feed)
  const health = useMemo(() => {
    if (!data) return null;
    const frames = data.meta.frames;
    return {
      fisheye: runsFromBooleans(fisheyeHealth(frames)),
      thermal: data.meta.streams.thermal
        ? runsFromStates(thermalHealth(frames))
        : [],
      radar: data.meta.streams.radar
        ? runsFromBooleans(radarHealth(frames, data.radar))
        : [],
    };
  }, [data]);
  const playhead = total > 1 ? index / (total - 1) : 0;

  // cut regions + stats for the transport / health bars / panel
  const runs = useMemo(() => cutRuns(frames, cuts), [frames, cuts]);
  const stats = useMemo(() => cutStats(frames, cuts), [frames, cuts]);
  const edges = useMemo(
    () =>
      cuts.map((c, cut) => {
        let startIdx = -1;
        let endIdx = -1;
        frames.forEach((f, i) => {
          if (inCut([c], f.ts)) {
            if (startIdx < 0) startIdx = i;
            endIdx = i;
          }
        });
        return { cut, startIdx: Math.max(startIdx, 0), endIdx: Math.max(endIdx, 0) };
      }),
    [frames, cuts],
  );

  // radar trail: current group + previous TRAIL_GROUPS-1 in radar time,
  // age-gated so a dead radar shows as silent, not as a frozen last trail.
  const { trail, radarSilentSince } = useMemo(() => {
    if (!data || !ts) return { trail: [] as number[][][], radarSilentSince: null as string | null };
    const sel = selectTrail(data.radarKeys, ts, TRAIL_GROUPS);
    return {
      trail: sel.groups.map((k) => data.radar[k] ?? []),
      radarSilentSince: sel.groups.length === 0 ? sel.lastKey ?? "start" : null,
    };
  }, [data, ts]);

  if (error) {
    return (
      <div className="p-8 font-mono text-sm text-status-serious">
        Failed to load clip {clipKey}: {error}
      </div>
    );
  }
  if (!data || !frame) {
    return (
      <div className="p-8 font-mono text-sm text-subtle">loading clip…</div>
    );
  }

  const { meta } = data;
  const attitude = record_?.attitude;
  const nReturns = trail.length
    ? trail[trail.length - 1]!.filter(
        (p) => (p[1] ?? 0) >= (layers.clutterFilter ? 0.5 : 0),
      ).length
    : 0;
  const hasScorer = Boolean(record_?.p_obstacle);
  const effectiveSource = hasScorer ? scoreSource : "scores";
  const deleted = isDeleted(record);

  const cutControls: CutControls = {
    runs,
    edges,
    inIndex,
    isCurrentCut: inCut(cuts, ts),
    nCuts: cuts.length,
    showCutFrames,
    onMarkIn: markIn,
    onMarkOut: markOut,
    onClearIn: clearIn,
    onToggleShowCutFrames: () => setShowCutFrames(!showCutFrames),
    onDragEdge,
    onDragEnd,
  };

  return (
    <div className="flex h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border bg-surface-1 px-4 py-2">
          <Link
            href="/clips"
            className="flex items-center gap-1 text-xs text-muted hover:text-foreground"
          >
            <ArrowLeft size={14} aria-hidden /> clips
          </Link>
          <div className="h-4 w-px bg-border" aria-hidden />
          <h1 className="truncate text-sm font-semibold tracking-tight">
            {meta.title}
          </h1>
          <span className="truncate font-mono text-[11px] tabular-nums text-subtle">
            {formatActivityRange(meta.triplet_ts, meta.end_ts)}
            {(meta.chunks?.length ?? 1) > 1 && ` · ${meta.chunks!.length} chunks`}
          </span>
          {cuts.length > 0 && (
            <span
              data-cut-count
              className="flex shrink-0 items-center gap-1 rounded-sm border border-status-warn/60 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-status-warn"
              title={`${stats.cut} of ${frames.length} frames cut`}
            >
              <Scissors size={11} aria-hidden /> {cuts.length} cut{cuts.length > 1 ? "s" : ""}
            </span>
          )}
          <Link
            href={`/annotate/${clipKey}`}
            className="ml-auto flex items-center gap-1.5 rounded-sm border border-border px-2 py-1 text-xs text-muted hover:border-border-strong hover:text-foreground"
          >
            <SquarePen size={13} aria-hidden /> annotate
          </Link>
        </div>

        {deleted && record?.deleted_at && (
          <div
            role="status"
            data-deleted-banner
            className="flex flex-wrap items-center gap-3 border-b border-status-serious/40 bg-status-serious/10 px-4 py-2 font-mono text-xs text-status-serious"
          >
            <span>
              Deleted on {fmtDate(record.deleted_at)}
              {record.purged_at
                ? " — bundle purged; this set is gone from the corpus."
                : ` — hidden from the catalogue, planner, data links and coverage map; ${daysRemaining(record.deleted_at, loadedAt)} days until prune.`}
            </span>
            {!record.purged_at && (
              <button
                type="button"
                onClick={() => void restore()}
                disabled={saving}
                className="flex items-center gap-1 rounded-sm border border-status-good px-2 py-0.5 uppercase tracking-wider text-status-good hover:bg-status-good/10 disabled:opacity-50"
              >
                <RotateCcw size={11} aria-hidden /> restore
              </button>
            )}
          </div>
        )}

        <div className="grid min-h-0 flex-1 grid-cols-2 grid-rows-2 gap-px overflow-auto bg-border">
          <Panel
            title="Fisheye · navigable water + edge"
            meta={
              attitude
                ? `roll ${attitude.roll_deg.toFixed(1)}° · pitch ${attitude.pitch_deg.toFixed(1)}° · yaw ${Math.round(attitude.yaw_deg)}°`
                : undefined
            }
            footer={
              health && (
                <HealthBar runs={health.fisheye} playhead={playhead} cuts={runs}
                           label="fisheye" note="red = loop-rate drop or chunk roll" />
              )
            }
          >
            <FisheyePanel
              assets={displayed?.assets ?? null}
              boxes={data.boxes[displayed?.assets.ts ?? ts] ?? []}
              instances={data.instances[displayed?.assets.ts ?? ts] ?? []}
              label={data.labels[displayed?.assets.ts ?? ts] ?? null}
              layers={layers}
              size={meta.image_size}
              radarPoints={trail.length ? trail[trail.length - 1] : []}
            />
          </Panel>
          <Panel
            title={`Thermal camera${meta.thermal_size ? ` (${meta.thermal_size[0]}×${meta.thermal_size[1]})` : ""}`}
            footer={
              health &&
              health.thermal.length > 0 && (
                <HealthBar runs={health.thermal} playhead={playhead} cuts={runs}
                           label="thermal" note="red = sensor down" />
              )
            }
          >
            <ThermalPanel
              img={displayed?.assets.thermal ?? null}
              down={
                !meta.streams.thermal ||
                (displayed !== null && displayed.assets.thermal === null)
              }
              downReason={
                meta.streams.thermal
                  ? "sensor was down at this point in the mission"
                  : "no thermal stream in this clip"
              }
            />
          </Panel>
          <Panel
            title="Radar bird's-eye"
            meta={meta.streams.radar ? `${nReturns} returns` : undefined}
            footer={
              <>
                {layers.targetTracks && record_?.targets && (
                  <TargetChips targets={record_.targets} />
                )}
                {health && health.radar.length > 0 && (
                  <HealthBar runs={health.radar} playhead={playhead} cuts={runs}
                             label="radar"
                             note="red = no returns/heartbeat near this frame" />
                )}
              </>
            }
          >
            <RadarPanel
              trail={trail}
              silentSince={radarSilentSince}
              down={!meta.streams.radar}
              yMin={layers.clutterFilter ? 0.5 : 0}
              spokeEdges={spokeEdges}
              targets={record_?.targets ?? null}
              showTargets={layers.targetTracks}
            />
          </Panel>
          <Panel
            title={
              effectiveSource === "scores"
                ? "Fused score per sector"
                : effectiveSource === "p_obstacle"
                  ? "Learned scorer · p(obstacle) per sector"
                  : "Learned scorer · threat per sector"
            }
          >
            <ScorerPanel
              view={sv}
              scoreSource={effectiveSource}
              showConfirmed={layers.confirmedMarkers}
            />
          </Panel>
        </div>

        <Transport
          index={index}
          total={total}
          ts={ts}
          playing={playing}
          buffering={buffering}
          onSeek={onSeek}
          onTogglePlay={onTogglePlay}
          cuts={cutControls}
        />
      </div>

      <aside className="w-80 shrink-0 space-y-4 overflow-auto border-l border-border bg-surface-1 p-3">
        <section>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
            Layers
          </h2>
          <LayerPanel streams={meta.streams} hasScorer={hasScorer} />
        </section>
        <section>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
            Cuts
          </h2>
          <CutsPanel
            cuts={cuts}
            stats={stats}
            saving={saving}
            error={curationError}
            onChange={(next) => void persistCuts(next)}
            onSeekTs={(t) => {
              // land on the last kept frame before the cut (or the cut's
              // first frame when cut frames are shown)
              const i = frames.findIndex((f) => f.ts === t);
              if (i >= 0) setIndex(seekKept(i, -1));
            }}
          />
        </section>
        <section>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
            Fusion bins
          </h2>
          <BinTable view={sv} threshold={threshold} scoreSource={effectiveSource} />
        </section>
        <section>
          <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
            Heading
          </h2>
          <HeadingPanel record={record_} />
        </section>
      </aside>
    </div>
  );
}
