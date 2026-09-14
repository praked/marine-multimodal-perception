"use client";

/* Client helper for the bundle-fetch links (/api/export): mint a 7-day
   fetch command for one or more clips and put it on the clipboard.
   Returns a human-readable outcome for the page's toast. */

export interface MintResult {
  kind: "pending" | "ok" | "error";
  message: string;
}

/** The pending toast callers show the moment the mint starts — presigning
    a large clip's thousands of objects can take ~15 s server-side. */
export function mintPending(nClips: number): MintResult {
  return {
    kind: "pending",
    message:
      `Minting data link for ${nClips} clip${nClips > 1 ? "s" : ""}… ` +
      "large clips can take 1-2 min.",
  };
}

export async function mintAndCopyFetchLink(clips: string[]): Promise<MintResult> {
  try {
    const r = await fetch("/api/export", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ clips }),
    });
    if (!r.ok) {
      const { error } = await r.json().catch(() => ({ error: r.status }));
      return { kind: "error", message: `Link failed: ${error}` };
    }
    const { command, files, bytes } = await r.json();
    await navigator.clipboard.writeText(command);
    const size = bytes > 1e9
      ? `${(bytes / 1e9).toFixed(2)} GB`
      : `${(bytes / 1e6).toFixed(1)} MB`;
    return {
      kind: "ok",
      message:
        `Data link copied: ${clips.length} clip${clips.length > 1 ? "s" : ""}, ` +
        `${files} files, ${size}. Paste into a shell; valid 7 days.`,
    };
  } catch (e) {
    return { kind: "error", message: `Link failed: ${String(e)}` };
  }
}
