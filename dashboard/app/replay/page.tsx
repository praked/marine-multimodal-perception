"use client";

import type { ReplayHandle } from "@/components/replay/ReplayScene";
import { getProvider, listCuratedClips } from "@/lib/data";
import { enriched } from "@/lib/datamap";
import { formatActivityRange } from "@/lib/format";
import type { ClipSummary, RadarFrames } from "@/lib/types";
import { Pause, Play } from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const ReplayScene = dynamic(
  () =>
    import("@/components/replay/ReplayScene").then((m) => ({
      default: m.ReplayScene,
    })),
  { ssr: false, loading: () => <div className="p-8 font-mono text-sm text-subtle">loading scene…</div> },
);

/* 3D mission replay: pick an activity, scrub its timeline — the sun and
   moon ride their real ephemeris, the light matches the capture instant,
   and the boat's radar returns float around the hull on the actual
   satellite sector. */

export default function ReplayPage() {
  const provider = useMemo(() => getProvider(), []);
  const [clips, setClips] = useState<(ClipSummary & { enrichment: NonNullable<ClipSummary["enrichment"]> })[]>([]);
  const [clipKeySel, setClipKeySel] = useState<string | null>(null);
  const [radar, setRadar] = useState<RadarFrames>({});
  const [radarKeys, setRadarKeys] = useState<string[]>([]);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const handleRef = useRef<ReplayHandle | null>(null);

  useEffect(() => {
    listCuratedClips().then(({ active: all }) => {
      const e = enriched(all).sort((a, b) =>
        b.triplet_ts.localeCompare(a.triplet_ts),
      );
      setClips(e);
      if (e.length) setClipKeySel(e[0]!.clip_id.replace("/", "__"));
    });
  }, [provider]);

  const clip = clips.find((c) => c.clip_id.replace("/", "__") === clipKeySel);
  const samples = useMemo(() => clip?.enrichment.sun_samples ?? [], [clip]);

  // render-time reset when the selected clip changes (not an effect)
  const [loadedClipKey, setLoadedClipKey] = useState<string | null>(null);
  if (clipKeySel !== loadedClipKey) {
    setLoadedClipKey(clipKeySel);
    setRadar({});
    setRadarKeys([]);
    setIdx(0);
  }

  useEffect(() => {
    if (!clipKeySel) return;
    let cancelled = false;
    provider.getRadar(clipKeySel).then((r) => {
      if (cancelled) return;
      setRadar(r);
      setRadarKeys(Object.keys(r).sort());
    });
    return () => {
      cancelled = true;
    };
  }, [clipKeySel, provider]);

  const pushFrame = useCallback(
    (i: number) => {
      if (!clip || !handleRef.current || samples.length === 0) return;
      const sample = samples[Math.min(i, samples.length - 1)]!;
      const [ts] = sample;
      const date = new Date(
        `${clip.triplet_ts.slice(0, 10)}T${ts.split(".")[0]}+02:00`, // CEST
      );
      // nearest radar group at/before ts
      let group: number[][] = [];
      for (let k = radarKeys.length - 1; k >= 0; k--) {
        if (radarKeys[k]! <= ts) {
          group = radar[radarKeys[k]!] ?? [];
          break;
        }
      }
      handleRef.current.setFrame({ date, radar: group, yawDeg: 0 });
    },
    [clip, samples, radar, radarKeys],
  );

  useEffect(() => pushFrame(idx), [idx, pushFrame]);

  useEffect(() => {
    if (!playing || samples.length === 0) return;
    const id = setInterval(
      () => setIdx((i) => (i + 1 >= samples.length ? 0 : i + 1)),
      120, // one sun-sample (~60 s of mission) every 120 ms
    );
    return () => clearInterval(id);
  }, [playing, samples.length]);

  const onReady = useCallback(
    (h: ReplayHandle) => {
      handleRef.current = h;
      pushFrame(0);
    },
    [pushFrame],
  );

  const sample = samples[Math.min(idx, Math.max(samples.length - 1, 0))];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-border bg-surface-1 px-4 py-2">
        <h1 className="text-sm font-semibold tracking-tight">Mission replay</h1>
        <select
          value={clipKeySel ?? ""}
          onChange={(e) => setClipKeySel(e.target.value)}
          className="rounded-sm border border-border bg-surface-1 px-2 py-1 font-mono text-xs"
          aria-label="Activity"
        >
          {clips.map((c) => (
            <option key={c.clip_id} value={c.clip_id.replace("/", "__")}>
              {formatActivityRange(c.triplet_ts, c.end_ts)} · {c.title}
            </option>
          ))}
        </select>
        <span className="ml-auto font-mono text-[10px] text-subtle">
          sun + moon on real ephemeris · sun + moon models NASA
        </span>
      </div>

      <div className="relative min-h-0 flex-1">
        {clip && (
          <ReplayScene
            lat={clip.enrichment.gps_init.lat}
            lon={clip.enrichment.gps_init.lon}
            onReady={onReady}
          />
        )}
      </div>

      <div className="flex items-center gap-3 border-t border-border bg-surface-1 px-4 py-2">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause" : "Play"}
          className="flex h-8 w-8 items-center justify-center rounded-sm bg-seeblau-100 text-inverse hover:bg-seeblau-deep"
        >
          {playing ? <Pause size={16} aria-hidden /> : <Play size={16} aria-hidden />}
        </button>
        <input
          type="range"
          aria-label="Mission time"
          min={0}
          max={Math.max(samples.length - 1, 0)}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
          className="min-w-0 flex-1 accent-(--seeblau-100)"
        />
        <span className="w-24 text-right font-mono text-xs tabular-nums">
          {sample ? sample[0] : "–"}
        </span>
        <span className="font-mono text-[10px] tabular-nums text-subtle">
          sun {sample ? `${sample[1]}°` : "–"}
        </span>
      </div>
    </div>
  );
}
