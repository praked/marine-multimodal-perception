"use client";

import { useEffect } from "react";

/* Registers the service worker (public/sw.js) once per page load. The
   worker caches the app shell as pages are visited and serves offline
   packs; nothing else changes for online use. Registration is skipped in
   non-secure contexts (plain http on a LAN box) where browsers refuse it. */
export function PwaRegister() {
  useEffect(() => {
    if (typeof window === "undefined" || !("serviceWorker" in navigator)) return;
    if (!window.isSecureContext) return;
    navigator.serviceWorker
      .register("/sw.js", { scope: "/" })
      .catch((e) => console.warn("[pwa] service worker registration failed:", e));
  }, []);
  return null;
}
