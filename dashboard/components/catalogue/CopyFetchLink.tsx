"use client";

import { mintAndCopyFetchLink, mintPending, type MintResult } from "@/lib/exportLink";
import { Check, Link2 } from "lucide-react";
import { useState } from "react";

/* Per-clip "copy fetch link": mints a 7-day bundle-fetch command via
   /api/export and puts it on the clipboard. Paste it into a shell on any
   machine (a GPU box) and the whole clip bundle downloads — no dashboard
   login needed there. Lives inside the card's <Link>, so it must swallow
   the click; the page-level toast (onResult) reports the outcome. */

export function CopyFetchLink({
  clipKey,
  onResult,
}: {
  clipKey: string;
  onResult: (r: MintResult) => void;
}) {
  const [state, setState] = useState<"idle" | "busy" | "done">("idle");

  async function mint() {
    if (state === "busy") return;
    setState("busy");
    onResult(mintPending(1));
    const res = await mintAndCopyFetchLink([clipKey]);
    onResult(res);
    setState(res.kind === "ok" ? "done" : "idle");
    if (res.kind === "ok") window.setTimeout(() => setState("idle"), 2500);
  }

  return (
    <button
      type="button"
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        void mint();
      }}
      title="Copy a fetch command for this clip's data (runs anywhere, link valid 7 days)"
      aria-label={`Copy fetch link for ${clipKey}`}
      className="shrink-0 rounded-sm border border-border p-1 text-subtle hover:border-border-strong hover:text-foreground"
    >
      {state === "done"
        ? <Check size={13} className="text-status-good" aria-hidden />
        : <Link2 size={13} className={state === "busy" ? "animate-pulse" : ""} aria-hidden />}
    </button>
  );
}
