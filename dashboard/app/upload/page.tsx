"use client";

import { checkSession, type SessionCheck } from "@/lib/upload";
import { isSupabaseConfigured } from "@/lib/data/provider";
import { createClient } from "@/lib/supabase/client";
import { CheckCircle2, CircleAlert, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState } from "react";

/* Upload a capture session (all its chunks) straight from the browser:
   structural validation first (chunk pairing, capture naming), then
   direct-to-bucket uploads via short-lived signed PUTs, then a tracking
   row the workstation processor picks up (validate -> bake -> enrich ->
   publish). The list below shows every session's live status. */

interface IngestRow {
  session: string;
  status: string;
  report: { summary?: string } | null;
  updated_at: string;
}

type FileState = "pending" | "uploading" | "done" | "failed";

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [session, setSession] = useState("");
  const [check, setCheck] = useState<SessionCheck | null>(null);
  const [states, setStates] = useState<Record<string, FileState>>({});
  const [phase, setPhase] = useState<"pick" | "uploading" | "done" | "error">("pick");
  const [message, setMessage] = useState<string | null>(null);
  const [rows, setRows] = useState<IngestRow[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  async function fetchRows(): Promise<IngestRow[] | null> {
    if (!isSupabaseConfigured()) return null;
    const { data } = await createClient()
      .from("sail_ingests")
      .select("session,status,report,updated_at")
      .order("updated_at", { ascending: false })
      .limit(20);
    return (data as IngestRow[] | null) ?? null;
  }
  function refreshRows() {
    fetchRows().then((data) => {
      if (data) setRows(data);
    }).catch(() => undefined);
  }
  useEffect(() => {
    let cancelled = false;
    const tick = () =>
      fetchRows().then((data) => {
        if (data && !cancelled) setRows(data);
      }).catch(() => undefined);
    void tick();
    const id = window.setInterval(() => void tick(), 10000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  function onPick(list: FileList | null) {
    if (!list || list.length === 0) return;
    const fs = [...list];
    const dir = fs[0]!.webkitRelativePath.split("/")[0] ?? "";
    setSession(dir.replace(/[^A-Za-z0-9_-]/g, "-"));
    setFiles(fs);
    setCheck(checkSession(fs.map((f) => f.name)));
    setStates({});
    setPhase("pick");
    setMessage(null);
  }

  async function uploadAll() {
    if (!check || files.length === 0) return;
    setPhase("uploading");
    const valid = files.filter((f) => !check.rejected.includes(f.name));
    const queue = [...valid];
    let failed = 0;
    const workers = Array.from({ length: 4 }, async () => {
      for (;;) {
        const f = queue.shift();
        if (!f) return;
        setStates((s) => ({ ...s, [f.name]: "uploading" }));
        try {
          const presign = await fetch("/api/upload", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ session, file: f.name, size: f.size }),
          });
          if (!presign.ok) throw new Error((await presign.json()).error);
          const { url } = await presign.json();
          const put = await fetch(url, { method: "PUT", body: f });
          if (!put.ok) throw new Error(`PUT ${put.status}`);
          setStates((s) => ({ ...s, [f.name]: "done" }));
        } catch {
          failed += 1;
          setStates((s) => ({ ...s, [f.name]: "failed" }));
        }
      }
    });
    await Promise.all(workers);
    if (failed > 0) {
      setPhase("error");
      setMessage(`${failed} file(s) failed — fix connectivity and press upload again (done files are skipped by re-upload)`);
      return;
    }
    // tracking row for the workstation processor
    if (isSupabaseConfigured()) {
      const { error } = await createClient().from("sail_ingests").upsert({
        session,
        status: "uploaded",
        manifest: {
          files: valid.map((f) => ({ name: f.name, size: f.size })),
          chunks: check.chunks.length,
        },
        updated_at: new Date().toISOString(),
      }, { onConflict: "session" });
      if (error) {
        setPhase("error");
        setMessage(`uploaded, but the tracking row failed: ${error.message}`);
        return;
      }
    }
    setPhase("done");
    setMessage(null);
    refreshRows();
  }

  const totalBytes = files.reduce((a, f) => a + f.size, 0);

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        Capture ingest
      </div>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">Upload a capture session</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Pick the session folder (all chunks). The files are checked for
        capture-format pairing, uploaded straight to the private bucket, and
        queued for the processing pipeline: validation, bake, enrichment and
        publish. The status list below updates as the workstation processes it.
      </p>

      <section className="mt-6 rounded-sm border border-border bg-surface-1 p-4">
        <input
          ref={inputRef}
          type="file"
          // @ts-expect-error non-standard directory attributes
          webkitdirectory=""
          directory=""
          multiple
          className="hidden"
          onChange={(e) => onPick(e.target.files)}
        />
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex items-center gap-2 rounded-sm border border-border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-muted hover:border-border-strong hover:text-foreground"
          >
            <UploadCloud size={14} aria-hidden /> choose session folder
          </button>
          {files.length > 0 && (
            <span className="font-mono text-xs tabular-nums text-muted">
              {session} · {files.length} files · {(totalBytes / 1e9).toFixed(2)} GB
            </span>
          )}
          {check && (
            <button
              type="button"
              onClick={() => void uploadAll()}
              disabled={!check.ok || phase === "uploading"}
              className="rounded-sm bg-seeblau-100 px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-inverse hover:bg-seeblau-deep disabled:opacity-50"
            >
              {phase === "uploading" ? "uploading…" : "upload session"}
            </button>
          )}
        </div>

        {check && (
          <div className="mt-4 border-t border-border pt-3">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-subtle">
              {check.chunks.length} chunks ·{" "}
              {check.ok ? "structure OK" : "structure problems"}
            </div>
            {check.problems.length > 0 && (
              <ul className="mb-2 space-y-0.5 text-[11px]">
                {check.problems.slice(0, 12).map((p, i) => (
                  <li key={i} className={p.includes("allowed") ? "text-subtle" : "text-status-warn"}>
                    {p}
                  </li>
                ))}
              </ul>
            )}
            {check.rejected.length > 0 && (
              <p className="mb-2 text-[11px] text-status-serious">
                ignored (not capture files): {check.rejected.slice(0, 6).join(", ")}
                {check.rejected.length > 6 ? ` +${check.rejected.length - 6}` : ""}
              </p>
            )}
            <div className="max-h-56 overflow-y-auto">
              <table className="w-full font-mono text-[11px] tabular-nums">
                <tbody>
                  {files.filter((f) => !check.rejected.includes(f.name)).map((f) => (
                    <tr key={f.name} className="border-b border-border/50">
                      <td className="py-0.5 pr-2">{f.name}</td>
                      <td className="pr-2 text-right text-subtle">{(f.size / 1e6).toFixed(1)} MB</td>
                      <td className="w-20 text-right">
                        {states[f.name] === "done" && <CheckCircle2 size={12} className="ml-auto text-status-good" aria-label="uploaded" />}
                        {states[f.name] === "failed" && <CircleAlert size={12} className="ml-auto text-status-serious" aria-label="failed" />}
                        {states[f.name] === "uploading" && <span className="text-subtle">…</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {phase === "done" && (
          <p className="mt-3 text-sm text-status-good">
            Uploaded and queued. The workstation processor picks it up next
            (`pnpm process:incoming` there); this page tracks its progress below.
          </p>
        )}
        {message && <p className="mt-3 text-sm text-status-serious">{message}</p>}
      </section>

      <section className="mt-6">
        <h2 className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
          Sessions
        </h2>
        <ul className="space-y-1.5">
          {rows.map((r) => (
            <li key={r.session} className="flex items-center gap-3 rounded-sm border border-border bg-surface-1 px-3 py-2 font-mono text-xs">
              <span
                className={`h-2 w-2 rounded-full ${r.status === "ready" ? "bg-status-good" : r.status === "failed" ? "bg-status-serious" : "bg-viz-water-edge"}`}
                aria-hidden
              />
              <span className="font-medium">{r.session}</span>
              <span className="text-subtle">{r.status}</span>
              {r.report?.summary && <span className="truncate text-subtle">{r.report.summary}</span>}
              <span className="ml-auto text-subtle">{r.updated_at.slice(0, 16).replace("T", " ")}</span>
            </li>
          ))}
          {rows.length === 0 && <li className="text-xs text-subtle">no uploaded sessions yet</li>}
        </ul>
      </section>
    </div>
  );
}
