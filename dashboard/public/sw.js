/* ASVProject dashboard service worker.

   Two jobs:
   1. App shell offline: pages, RSC payloads and Next's hashed static chunks
      are cached as they are fetched (network-first for pages/RSC so a deploy
      wins when online; cache-first for immutable /_next/static). Offline, a
      previously visited page renders from cache; an unvisited one falls back
      to the cached /offline page.
   2. Offline packs: image assets that the /offline page stored in Cache
      Storage under their page-facing URL (/api/assets/... or /demo/...) are
      served from there — cache-first — so <img src> in the annotate stage
      works with no network. The pack caches are written by the page
      (lib/offline/packs.ts), never by this worker.

   The password-gate cookie is set by /login; cached responses were fetched
   with it, and offline requests never reach the middleware, so cached pages
   serve. Nothing cross-origin (Supabase REST, R2) is ever cached here. */

const SHELL_CACHE = "asvproject-shell-v1";
const STATIC_CACHE = "asvproject-static-v1";
const PACK_PREFIX = "asvproject-pack:";
const OFFLINE_FALLBACK = "/offline";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((c) => c.add(OFFLINE_FALLBACK).catch(() => {})),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keep = new Set([SHELL_CACHE, STATIC_CACHE]);
      for (const name of await caches.keys()) {
        if (!keep.has(name) && !name.startsWith(PACK_PREFIX)) await caches.delete(name);
      }
      await self.clients.claim();
    })(),
  );
});

function isAssetPath(url) {
  return url.pathname.startsWith("/api/assets/") || url.pathname.startsWith("/demo/");
}

async function fromPacks(request) {
  // caches.match searches every cache; pack caches are the only ones that
  // hold asset URLs, so a hit is always a packed frame.
  return caches.match(request, { ignoreSearch: false, ignoreVary: true });
}

async function networkFirst(request, cacheName, key) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(request);
    if (res && res.ok && res.type === "basic") {
      cache.put(key || request, res.clone()).catch(() => {});
    }
    return res;
  } catch (err) {
    const hit = await cache.match(key || request);
    if (hit) return hit;
    throw err;
  }
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res && res.ok) cache.put(request, res.clone()).catch(() => {});
  return res;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return; // Supabase / R2: passthrough

  // Offline packs: page-facing asset URLs (no ?json — that is the signing
  // call, which only makes sense online).
  if (isAssetPath(url) && !url.searchParams.has("json")) {
    event.respondWith(
      (async () => {
        const packed = await fromPacks(request);
        if (packed) return packed;
        return fetch(request);
      })(),
    );
    return;
  }

  if (url.pathname.startsWith("/_next/static/")) {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
    return;
  }

  if (url.pathname.startsWith("/api/")) return; // live-only endpoints

  if (request.mode === "navigate") {
    event.respondWith(
      (async () => {
        try {
          return await networkFirst(request, SHELL_CACHE, url.pathname);
        } catch {
          const cache = await caches.open(SHELL_CACHE);
          return (
            (await cache.match(OFFLINE_FALLBACK)) ||
            new Response("Offline and this page is not cached.", {
              status: 503,
              headers: { "content-type": "text/plain" },
            })
          );
        }
      })(),
    );
    return;
  }

  // RSC payloads, fonts, manifest, icons, demo JSON: network-first with a
  // cache fallback keyed by the full URL.
  event.respondWith(
    networkFirst(request, SHELL_CACHE).catch(
      () => new Response("", { status: 503 }),
    ),
  );
});

self.addEventListener("message", (event) => {
  if (event.data === "skipWaiting") self.skipWaiting();
});
