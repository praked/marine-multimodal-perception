"use client";

import { loadPlan, type StoredPlan } from "@/lib/annotate/planner";
import { getAuditBackend } from "@/lib/audit/backend";
import { keptFrames, type CurationRecord } from "@/lib/curation";
import { getProvider, listCuratedClips } from "@/lib/data";
import {
  deletePack,
  downloadPack,
  estimateBytes,
  listPacks,
  packablePlanFrames,
  storageEstimate,
  subscribePacks,
  type DownloadProgress,
  type PackFrame,
  type PackManifest,
} from "@/lib/offline/packs";
import { queuedCount, subscribeQueue } from "@/lib/offline/queue";
import { clipKeyFromId, type ClipMeta, type ClipSummary } from "@/lib/types";
import { HardDriveDownload, Trash2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

/* Offline packs: download the frames (+ thermal + JSON side files) an
   annotation session needs into the browser, so /annotate works with the
   network off. A pack is a saved plan or a whole set; deleted sets and cut
   frames are never packed. Packs live in this browser profile. */

function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  return `${Math.round(b / 1e3)} KB`;
}

export default function OfflinePage() {
  const [plan, setPlan] = useState<StoredPlan | null>(null);
  const [planFrames, setPlanFrames] = useState<{ frames: PackFrame[]; dropped: number } | null>(null);
  const [clips, setClips] = useState<ClipSummary[] | null>(null);
  const [records, setRecords] = useState<Map<string, CurationRecord>>(new Map());
  const [packs, setPacks] = useState<PackManifest[]>([]);
  const [progress, setProgress] = useState<{ name: string; p: DownloadProgress } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [storage, setStorage] = useState<{ usage: number; quota: number; persisted: boolean } | null>(null);
  const [pending, setPending] = useState(0);
  const [online, setOnline] = useState(true);
  const [busy, setBusy] = useState(false);

  const refreshPacks = () => {
    void listPacks().then(setPacks);
    void storageEstimate().then(setStorage);
    void queuedCount().then(setPending);
  };

  useEffect(() => {
    // browser stores (localStorage / IndexedDB / navigator): read post-render
    void Promise.resolve().then(() => {
      setPlan(loadPlan());
      setOnline(navigator.onLine);
      refreshPacks();
    });
    const unsubPacks = subscribePacks(refreshPacks);
    const unsubQueue = subscribeQueue(() => void queuedCount().then(setPending));
    const on = () => setOnline(true);
    const off = () => setOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    listCuratedClips()
      .then(({ active, records }) => {
        setClips([...active].sort((a, b) => b.triplet_ts.localeCompare(a.triplet_ts)));
        setRecords(records);
      })
      .catch((e: unknown) => setError(String(e)));
    return () => {
      unsubPacks();
      unsubQueue();
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);

  // Size the saved plan against the catalogue (needs the metas of the clips
  // it touches, for thermal presence + curation).
  useEffect(() => {
    if (!plan || plan.frames.length === 0) return;
    let cancelled = false;
    (async () => {
      const provider = getProvider();
      const keys = [...new Set(plan.frames.map((f) => f.clipKey))];
      const metas = new Map<string, ClipMeta>();
      for (const k of keys) {
        try {
          metas.set(k, await provider.getMeta(k));
        } catch {
          /* clip gone: its frames count as dropped */
        }
      }
      if (cancelled) return;
      setPlanFrames(packablePlanFrames(plan.frames, metas, records));
    })();
    return () => {
      cancelled = true;
    };
  }, [plan, records]);

  async function download(spec: { kind: PackManifest["kind"]; name: string; frames: PackFrame[] }) {
    if (busy || spec.frames.length === 0) return;
    setBusy(true);
    setError(null);
    setProgress({ name: spec.name, p: { done: 0, failed: 0, total: spec.frames.length, bytes: 0 } });
    try {
      const backend = getAuditBackend();
      await downloadPack(
        getProvider(),
        { id: crypto.randomUUID(), kind: spec.kind, name: spec.name, frames: spec.frames },
        (k) => backend.list(k),
        (p) => setProgress({ name: spec.name, p }),
      );
    } catch (e) {
      setError(`download failed: ${String(e)}`);
    } finally {
      setBusy(false);
      setProgress(null);
      refreshPacks();
    }
  }

  async function downloadSet(clip: ClipSummary) {
    const key = clipKeyFromId(clip.clip_id);
    const meta = await getProvider().getMeta(key);
    const frames: PackFrame[] = keptFrames(meta.frames.filter((f) => f.fisheye), records.get(key)).map((f) => ({
      clipKey: key,
      ts: f.ts,
      thermal: f.thermal,
    }));
    await download({ kind: "set", name: clip.title, frames });
  }

  const packedKeys = new Set(packs.flatMap((p) => Object.keys(p.clips)));

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        Offline annotation · packs
      </div>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">Offline packs</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Download the frames an annotation session needs into this browser. With a
        pack present, <Link href="/annotate" className="text-seeblau-deep underline">annotate</Link> works
        with the network off: audits queue locally and sync when you are back
        online. Packs never include deleted sets or cut frames, and they live in
        this browser profile (not per user).
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-3 font-mono text-[11px] text-subtle" data-offline-status>
        <span className={online ? "text-status-good" : "text-status-warn"}>
          {online ? "online" : "offline"}
        </span>
        {storage && (
          <span>
            storage {fmtBytes(storage.usage)} of {fmtBytes(storage.quota)} ·{" "}
            {storage.persisted ? "persisted" : "not persisted (browser may evict; granted on first download)"}
          </span>
        )}
        {pending > 0 && <span className="text-status-warn">{pending} audit{pending > 1 ? "s" : ""} pending sync</span>}
      </div>

      {error && <p className="mt-3 font-mono text-xs text-status-serious">{error}</p>}

      {progress && (
        <div className="mt-4 rounded-sm border border-seeblau-100/60 bg-surface-1 p-3" data-download-progress>
          <div className="flex justify-between font-mono text-[11px] tabular-nums">
            <span>downloading {progress.name}</span>
            <span>
              {progress.p.done + progress.p.failed}/{progress.p.total} · {fmtBytes(progress.p.bytes)}
              {progress.p.failed > 0 && ` · ${progress.p.failed} failed`}
            </span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full bg-seeblau-100"
              style={{ width: `${((progress.p.done + progress.p.failed) / Math.max(progress.p.total, 1)) * 100}%` }}
            />
          </div>
        </div>
      )}

      <section className="mt-6 rounded-sm border border-border bg-surface-1 p-4">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">Saved annotation plan</h2>
        {!plan || plan.frames.length === 0 ? (
          <p className="mt-2 text-xs text-subtle">
            No saved plan. Build one on <Link href="/annotate" className="underline">/annotate</Link> first.
          </p>
        ) : (
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
            <span className="font-mono tabular-nums">
              {plan.frames.length} frames · built {plan.createdAt.slice(0, 16).replace("T", " ")}
              {planFrames && (
                <>
                  {" "}· {planFrames.frames.length} packable ≈ {fmtBytes(estimateBytes(planFrames.frames))}
                  {planFrames.dropped > 0 && (
                    <span className="text-status-warn"> · {planFrames.dropped} dropped (deleted/cut)</span>
                  )}
                </>
              )}
            </span>
            <button
              type="button"
              disabled={busy || !planFrames || planFrames.frames.length === 0 || !online}
              onClick={() =>
                planFrames &&
                void download({
                  kind: "plan",
                  name: `plan ${plan.createdAt.slice(0, 10)} · ${planFrames.frames.length} frames`,
                  frames: planFrames.frames,
                })
              }
              data-download-plan
              className="flex items-center gap-1 rounded-sm bg-seeblau-100 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-inverse hover:bg-seeblau-deep disabled:opacity-50"
            >
              <HardDriveDownload size={12} aria-hidden /> download plan
            </button>
          </div>
        )}
      </section>

      <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">Packs in this browser</h2>
        {packs.length === 0 ? (
          <p className="mt-2 text-xs text-subtle">none yet</p>
        ) : (
          <ul className="mt-2 divide-y divide-border">
            {packs.map((p) => (
              <li key={p.id} data-pack={p.id} className="flex flex-wrap items-center gap-3 py-2 text-xs">
                <span className="font-medium">{p.name}</span>
                <span className="font-mono text-[11px] text-subtle">{p.kind}</span>
                <span className="font-mono text-[11px] tabular-nums text-muted">
                  {p.done}/{p.frames.length} frames · {fmtBytes(p.bytes)} · {Object.keys(p.clips).length} set{Object.keys(p.clips).length > 1 ? "s" : ""}
                </span>
                <span
                  data-pack-status={p.status}
                  className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase ${
                    p.status === "ready"
                      ? "border-status-good text-status-good"
                      : p.status === "downloading"
                        ? "border-seeblau-100 text-seeblau-deep"
                        : "border-status-warn text-status-warn"
                  }`}
                  title={p.error}
                >
                  {p.status}
                </span>
                <span className="ml-auto flex items-center gap-2">
                  {Object.keys(p.clips).length === 1 && (
                    <Link
                      href={`/annotate/${Object.keys(p.clips)[0]}${p.kind === "plan" ? "?plan=1" : ""}`}
                      className="font-mono text-[11px] uppercase tracking-wider text-seeblau-deep hover:underline"
                    >
                      annotate
                    </Link>
                  )}
                  {p.kind === "plan" && Object.keys(p.clips).length > 1 && (
                    <Link href="/annotate" className="font-mono text-[11px] uppercase tracking-wider text-seeblau-deep hover:underline">
                      continue plan
                    </Link>
                  )}
                  <button
                    type="button"
                    onClick={() => void deletePack(p.id)}
                    aria-label={`Delete pack ${p.name}`}
                    className="flex items-center gap-1 rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] uppercase text-subtle hover:border-status-serious hover:text-status-serious"
                  >
                    <Trash2 size={11} aria-hidden /> delete
                  </button>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">Download a whole set</h2>
        {!clips ? (
          <p className="mt-2 text-xs text-subtle">loading…</p>
        ) : (
          <ul className="mt-2 divide-y divide-border">
            {clips.map((clip) => {
              const key = clipKeyFromId(clip.clip_id);
              const est = clip.n_frames * 75_000 + (clip.streams.thermal ? clip.n_frames * 6_000 : 0);
              return (
                <li key={clip.clip_id} data-set={key} className="flex flex-wrap items-center gap-3 py-2 text-xs">
                  <span className="font-medium">{clip.title}</span>
                  <span className="font-mono text-[11px] tabular-nums text-subtle">
                    {clip.n_frames} frames ≈ {fmtBytes(est)}
                    {records.get(key)?.cuts.length ? ` · ${records.get(key)!.cuts.length} cut(s) excluded` : ""}
                  </span>
                  {packedKeys.has(key) && (
                    <span className="rounded-sm border border-status-good px-1.5 py-0.5 font-mono text-[10px] uppercase text-status-good">packed</span>
                  )}
                  <button
                    type="button"
                    disabled={busy || !online}
                    onClick={() => void downloadSet(clip).catch((e) => setError(String(e)))}
                    data-download-set={key}
                    className="ml-auto flex items-center gap-1 rounded-sm border border-border px-2 py-1 font-mono text-[11px] uppercase tracking-wider text-muted hover:border-seeblau-100 hover:text-seeblau-deep disabled:opacity-50"
                  >
                    <HardDriveDownload size={12} aria-hidden /> download
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}
