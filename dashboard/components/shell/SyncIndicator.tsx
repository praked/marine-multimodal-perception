"use client";

import { syncPendingAudits } from "@/lib/audit/backend";
import { queuedCount, subscribeQueue, type SyncReport } from "@/lib/offline/queue";
import { CloudOff, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

/* "N audits pending sync" pill in the top strip: shows the offline audit
   queue, syncs on click and automatically when the browser comes back
   online. Hidden while the queue is empty and the browser is online. */
export function SyncIndicator() {
  const [count, setCount] = useState(0);
  const [online, setOnline] = useState(true);
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<SyncReport | null>(null);

  const refresh = useCallback(() => {
    void queuedCount().then(setCount);
  }, []);

  const sync = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    try {
      setLast(await syncPendingAudits());
    } finally {
      setBusy(false);
      refresh();
    }
  }, [busy, refresh]);

  useEffect(() => {
    // IndexedDB + navigator.onLine are browser stores: read after render
    void Promise.resolve().then(() => {
      refresh();
      setOnline(navigator.onLine);
    });
    const unsub = subscribeQueue(refresh);
    const goOnline = () => {
      setOnline(true);
      void sync();
    };
    const goOffline = () => setOnline(false);
    window.addEventListener("online", goOnline);
    window.addEventListener("offline", goOffline);
    return () => {
      unsub();
      window.removeEventListener("online", goOnline);
      window.removeEventListener("offline", goOffline);
    };
  }, [refresh, sync]);

  if (count === 0 && online) return null;
  return (
    <span
      data-sync-indicator
      data-pending={count}
      className={`flex items-center gap-2 rounded-sm border px-2 py-0.5 font-mono text-[11px] tabular-nums ${
        online ? "border-status-warn text-status-warn" : "border-border-strong text-muted"
      }`}
      title={
        online
          ? "Audits saved while offline are replayed into the shared log; a retried record can never duplicate (client ids)."
          : "Offline: audits are saved locally and synced when the connection returns."
      }
    >
      {!online && <CloudOff size={12} aria-hidden />}
      {count > 0
        ? `${count} audit${count > 1 ? "s" : ""} pending sync`
        : "offline"}
      {count > 0 && (
        <button
          type="button"
          onClick={() => void sync()}
          disabled={busy || !online}
          className="flex items-center gap-1 rounded-sm border border-current px-1.5 py-0 uppercase tracking-wider hover:bg-surface-3 disabled:opacity-50"
          aria-label="Sync pending audits now"
        >
          <RefreshCw size={11} className={busy ? "animate-spin" : ""} aria-hidden /> sync
        </button>
      )}
      {last && last.rejected > 0 && (
        <span className="text-status-serious" title={`${last.rejected} record(s) were rejected by the server and stay queued`}>
          {last.rejected} rejected
        </span>
      )}
    </span>
  );
}
