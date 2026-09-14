import type { FrameKind } from "@/lib/data/provider";
import {
  decodeSeg,
  edgeSegments,
  smoothEdge,
  tintOverlay,
  waterEdgeColumns,
} from "@/lib/viewer/segTint";
import type { FrameEntry } from "@/lib/types";

/* Synchronised frame-asset loader.

   The panels used to fetch their own assets and swap whenever their bytes
   arrived: fisheye (frame + mask, canvas) and thermal (<img>) drifted apart
   and slow loads froze the canvas. This loader fetches EVERYTHING a frame
   needs together, decodes it off the hot path, and hands the viewer one
   atomic FrameAssets object — both panels swap in the same commit. An LRU
   keeps recent frames; prefetch keeps playback ahead of the network. */

export interface SegLayer {
  tint: ImageData;
  segments: [number, number][][];
}

export interface FrameAssets {
  ts: string;
  fisheye: HTMLImageElement | null;
  thermal: HTMLImageElement | null;
  seg: SegLayer | null;
}

async function loadDecodedImage(
  url: string,
  cors = false,
): Promise<HTMLImageElement> {
  const img = new Image();
  // crossOrigin ONLY where we read pixels back (the seg mask's getImageData).
  // Fisheye/thermal are draw-only — canvas taint doesn't block drawing — and
  // no-cors loads are immune to missing-ACAO/cache-mode issues (Arc bug).
  if (cors) img.crossOrigin = "anonymous";
  img.src = url;
  try {
    await img.decode(); // pixels ready — no paint-time decode jank
  } catch {
    // Safari rejects decode() spuriously on cross-origin/redirected images.
    // Fall back to the load event; only a genuinely pixel-less image fails.
    if (!img.complete || img.naturalWidth === 0) {
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () =>
          reject(new Error(`image failed: ${url.slice(0, 120)}`));
        if (img.complete && img.naturalWidth > 0) resolve();
      });
    }
  }
  if (img.naturalWidth === 0) {
    throw new Error(`image empty: ${url.slice(0, 120)}`);
  }
  return img;
}

export function decodeSegImage(
  img: HTMLImageElement,
  w: number,
  h: number,
): SegLayer {
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const ctx = off.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, w, h);
  const cls = decodeSeg(ctx.getImageData(0, 0, w, h).data, w, h);
  const tint = new ImageData(tintOverlay(cls, w, h), w, h);
  const edge = smoothEdge(waterEdgeColumns(cls, w, h));
  // maxJump 40 was tuned at 648 px native height; scale to bundle resolution
  const segments = edgeSegments(edge, Math.max(8, (40 * h) / 648));
  return { tint, segments };
}

export class FrameAssetLoader {
  private cache = new Map<string, Promise<FrameAssets>>();
  private ready = new Set<string>();
  private readonly capacity = 40;
  // Concurrency gate: fast scrubbing can otherwise fire hundreds of image
  // loads at once and browsers fail them en masse (ERR_INSUFFICIENT_RESOURCES).
  private inFlight = 0;
  private waiters: (() => void)[] = [];
  private readonly maxConcurrent = 12;

  private async acquire(): Promise<void> {
    if (this.inFlight < this.maxConcurrent) {
      this.inFlight++;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
    this.inFlight++;
  }

  private releaseSlot(): void {
    this.inFlight--;
    this.waiters.shift()?.();
  }

  /** True once a frame's assets are decoded and swappable (playback holds
      the current frame until the next one is ready, so the pair of camera
      panels advance in lockstep at whatever rate the network sustains). */
  isReady(ts: string): boolean {
    return this.ready.has(ts);
  }

  constructor(
    private resolveUrl: (kind: FrameKind, ts: string) => Promise<string>,
    private size: [number, number],
  ) {}

  load(entry: FrameEntry): Promise<FrameAssets> {
    const cached = this.cache.get(entry.ts);
    if (cached) return cached;
    const [w, h] = this.size;
    const promise: Promise<FrameAssets> = (async () => {
      const warn = (kind: string) => (e: unknown) => {
        console.warn(`[frameAssets] ${kind} load failed for ts=${entry.ts}:`, e);
        return null;
      };
      const loadKind = async (kind: FrameKind) => {
        const cors = kind === "seg";
        await this.acquire();
        try {
          return await this.resolveUrl(kind, entry.ts).then((u) =>
            loadDecodedImage(u, cors),
          );
        } catch {
          // transient blips (throttled tab, cold start, network hiccup)
          // deserve one retry before we report a hole
          await new Promise((r) => setTimeout(r, 300));
          return await this.resolveUrl(kind, entry.ts).then((u) =>
            loadDecodedImage(u, cors),
          );
        } finally {
          this.releaseSlot();
        }
      };
      const [fisheye, thermal, segImg] = await Promise.all([
        entry.fisheye ? loadKind("frames").catch(warn("fisheye")) : null,
        entry.thermal ? loadKind("thermal").catch(warn("thermal")) : null,
        entry.seg ? loadKind("seg").catch(warn("seg")) : null,
      ]);
      // Never cache a frame whose EXPECTED core assets failed: cached
      // failures replay as a permanent-looking DOWN on every revisit.
      if ((entry.fisheye && !fisheye) || (entry.thermal && !thermal)) {
        this.cache.delete(entry.ts);
        this.ready.delete(entry.ts);
      }
      let seg: SegLayer | null = null;
      if (segImg) {
        try {
          seg = decodeSegImage(segImg, w, h);
        } catch {
          seg = null;
        }
      }
      return { ts: entry.ts, fisheye, thermal, seg };
    })();
    promise.then(() => this.ready.add(entry.ts)).catch(() => {});
    this.cache.set(entry.ts, promise);
    if (this.cache.size > this.capacity) {
      const oldest = this.cache.keys().next().value;
      if (oldest) {
        this.cache.delete(oldest);
        this.ready.delete(oldest);
      }
    }
    return promise;
  }

  /** Kick off loads for the next few frames without awaiting them. */
  prefetch(frames: FrameEntry[], from: number, ahead = 4): void {
    for (let i = 1; i <= ahead; i++) {
      const entry = frames[from + i];
      if (entry) void this.load(entry);
    }
  }
}
