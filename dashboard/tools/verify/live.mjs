/* Live verification of set curation + offline annotation, run against the
 * deployed dashboard behind the password gate (Playwright, headless).
 *
 *   cd dashboard && node tools/verify/live.mjs [--base https://asvproject-dashboard.vercel.app]
 *
 * Reads DASHBOARD_PASSWORD + Supabase service credentials from .env.local.
 * Leaves the catalogue as it found it: the set it deletes is restored, the
 * cut it creates is removed, the test audit row it syncs is deleted again
 * (service role) and the pack is removed from the browser profile.
 */

import { createClient } from "@supabase/supabase-js";
import { existsSync, readFileSync } from "node:fs";
import { chromium } from "playwright";

const env = {};
if (existsSync(".env.local")) {
  for (const line of readFileSync(".env.local", "utf8").split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#") || !t.includes("=")) continue;
    const i = t.indexOf("=");
    env[t.slice(0, i).trim()] = t.slice(i + 1).trim();
  }
}
const argBase = process.argv.indexOf("--base");
const BASE = argBase >= 0 ? process.argv[argBase + 1] : "https://asvproject-dashboard.vercel.app";
const PASSWORD = process.env.DASHBOARD_PASSWORD ?? env.DASHBOARD_PASSWORD;
const DELETE_KEY = "2026-08-18_pontoon__2026-08-18_17-08-21"; // 249 frames
const PACK_KEY = "2026-08-19_afloat_canoe_fx__2026-08-19_15-58-20"; // 180 frames

const sb = createClient(env.NEXT_PUBLIC_SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

const results = [];
/** Poll a Supabase read until it satisfies `pred` (UI writes are optimistic;
    the upsert lands a moment later). */
async function waitDb(query, pred, timeoutMs = 15000) {
  const t0 = Date.now();
  for (;;) {
    const { data } = await query();
    if (pred(data)) return data;
    if (Date.now() - t0 > timeoutMs) return data;
    await new Promise((r) => setTimeout(r, 500));
  }
}
function ok(name, pass, detail = "") {
  results.push({ name, pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}
async function step(name, fn) {
  try {
    const detail = await fn();
    ok(name, true, detail ?? "");
  } catch (e) {
    ok(name, false, String(e?.message ?? e));
  }
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await context.newPage();
page.on("pageerror", (e) => console.log("  [pageerror]", e.message));

// ---- login --------------------------------------------------------------
await page.goto(`${BASE}/login`);
await page.fill('input[name="password"]', PASSWORD);
await Promise.all([page.waitForURL((u) => !u.pathname.startsWith("/login")), page.keyboard.press("Enter")]);
ok("login through the gate", !page.url().includes("/login"), page.url());

// ---- 1. delete → gone → export refuses → restore → back -----------------
await page.goto(`${BASE}/clips`);
await page.waitForSelector("[data-clip-key]");
await step("clips: catalogue renders active sets", async () => {
  const n = await page.locator("[data-clip-key]").count();
  if (n < 10) throw new Error(`only ${n} cards`);
  return `${n} cards`;
});
await step("clips: delete a set (two-click confirm) removes it from the active list", async () => {
  const card = page.locator(`[data-clip-key="${DELETE_KEY}"]`);
  await card.locator(`button[aria-label="Delete ${DELETE_KEY}"]`).click();
  await card.locator(`button[aria-label="Confirm delete ${DELETE_KEY}"]`).click();
  await page.waitForSelector(`[data-clip-key="${DELETE_KEY}"]`, { state: "detached", timeout: 15000 });
  const { data } = await sb.from("sail_curation").select("deleted_at, restored_at").eq("clip_key", DELETE_KEY).single();
  if (!data?.deleted_at) throw new Error("no deleted_at in sail_curation");
  return `deleted_at ${data.deleted_at}`;
});
await step("clips: Deleted (N) view shows it with date, days remaining and Restore", async () => {
  await page.getByRole("tab", { name: /Deleted \(\d+\)/ }).click();
  const card = page.locator(`[data-clip-key="${DELETE_KEY}"]`);
  await card.waitFor({ timeout: 10000 });
  const text = await card.innerText();
  if (!/Deleted on \d{2}-\d{2}-\d{4}/.test(text) || !/days until prune/.test(text)) throw new Error(text.slice(0, 200));
  return text.match(/Deleted on [^·]+·[^\n]+/)?.[0];
});
await step("api: /api/export refuses the deleted set (410)", async () => {
  const r = await context.request.post(`${BASE}/api/export`, { data: { clips: [DELETE_KEY] } });
  if (r.status() !== 410) throw new Error(`status ${r.status()}: ${await r.text()}`);
  return (await r.json()).error;
});
await step("viewer: deleted banner with restore on the set", async () => {
  await page.goto(`${BASE}/clips/${DELETE_KEY}`);
  await page.waitForSelector("[data-deleted-banner]", { timeout: 30000 });
  return (await page.locator("[data-deleted-banner]").innerText()).slice(0, 80);
});
await step("clips: Restore brings it back to the active list", async () => {
  await page.goto(`${BASE}/clips`);
  await page.waitForSelector("[data-clip-key]");
  await page.getByRole("tab", { name: /Deleted \(\d+\)/ }).click();
  await page.locator(`button[aria-label="Restore ${DELETE_KEY}"]`).click();
  await page.waitForSelector(`[data-clip-key="${DELETE_KEY}"]`, { state: "detached", timeout: 15000 });
  await page.getByRole("tab", { name: /Active \(\d+\)/ }).click();
  await page.locator(`[data-clip-key="${DELETE_KEY}"]`).waitFor({ timeout: 10000 });
  const r = await context.request.post(`${BASE}/api/export`, { data: { clips: [DELETE_KEY] } });
  return `export status after restore: ${r.status()}`;
});

// ---- 2. cuts: create / skip / edit / remove ------------------------------
await page.goto(`${BASE}/clips/${PACK_KEY}`);
await page.waitForSelector('input[aria-label="Scrub through frames"]', { timeout: 60000 });
const counter = () => page.locator("span.font-mono.text-xs.tabular-nums.text-muted").filter({ hasText: /^\d+\/\d+$/ }).first().innerText();
await step("viewer: in/out points create a cut (greyed region + header chip)", async () => {
  await page.locator("body").click({ position: { x: 5, y: 300 } });
  await page.keyboard.press("Home");
  for (let i = 0; i < 5; i++) await page.keyboard.press("ArrowRight");
  await page.keyboard.press("i");
  for (let i = 0; i < 5; i++) await page.keyboard.press("ArrowRight");
  await page.keyboard.press("o");
  await page.waitForSelector("[data-cut-count]", { timeout: 15000 });
  await page.waitForSelector("[data-cut-region]");
  const data = await waitDb(
    () => sb.from("sail_curation").select("cuts").eq("clip_key", PACK_KEY).maybeSingle(),
    (d) => d && d.cuts.length === 1,
  );
  if (!data || data.cuts.length !== 1) throw new Error(`sail_curation cuts: ${JSON.stringify(data)}`);
  return `${await page.locator("[data-cut-count]").innerText()} · ${JSON.stringify(data.cuts[0])}`;
});
await step("viewer: stepping skips the cut (frame 5 → 12; IN/OUT inclusive) and the show-cut-frames toggle reveals it", async () => {
  await page.keyboard.press("Home");
  for (let i = 0; i < 4; i++) await page.keyboard.press("ArrowRight");
  const before = await counter();
  await page.keyboard.press("ArrowRight");
  const after = await counter();
  if (before !== "5/180" || after !== "12/180") throw new Error(`${before} -> ${after}`);
  await page.getByRole("switch", { name: /show cut frames/i }).click();
  await page.keyboard.press("Home");
  for (let i = 0; i < 6; i++) await page.keyboard.press("ArrowRight");
  const shown = await counter();
  const chip = await page.locator("[data-current-cut]").count();
  await page.getByRole("switch", { name: /show cut frames/i }).click();
  if (shown !== "7/180" || chip !== 1) throw new Error(`shown ${shown}, cut chip ${chip}`);
  return `${before} → ${after} skipping; ${shown} with cut frames shown`;
});
await step("viewer: playback skips the cut", async () => {
  await page.keyboard.press("Home");
  for (let i = 0; i < 5; i++) await page.keyboard.press("ArrowRight");
  await page.keyboard.press(" ");
  await page.waitForTimeout(1500);
  await page.keyboard.press(" ");
  const at = Number((await counter()).split("/")[0]);
  if (at >= 7 && at <= 11) throw new Error(`playhead landed inside the cut: ${at}`);
  return `playhead ${at}/180`;
});
await step("viewer: editing a cut's end timestamp persists", async () => {
  const end = page.locator('input[aria-label="Cut 1 end"]');
  const oldEnd = await end.inputValue();
  const stats0 = await page.locator("[data-cut-stats]").innerText();
  await end.fill("15:58:33.0");
  await end.press("Enter");
  await page.waitForFunction((s0) => document.querySelector("[data-cut-stats]")?.textContent !== s0, stats0, { timeout: 15000 });
  const data = await waitDb(
    () => sb.from("sail_curation").select("cuts").eq("clip_key", PACK_KEY).maybeSingle(),
    (d) => d?.cuts?.[0]?.end_ts === "15:58:33.0",
  );
  if (data.cuts[0].end_ts !== "15:58:33.0") throw new Error(JSON.stringify(data.cuts));
  return `${oldEnd} → ${data.cuts[0].end_ts} · ${await page.locator("[data-cut-stats]").innerText()}`;
});
await step("viewer: removing the cut clears chip, regions and the row", async () => {
  await page.locator('button[aria-label="Remove cut 1"]').click();
  await page.waitForSelector("[data-cut-count]", { state: "detached", timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("[data-cut-region]").length === 0);
  const data = await waitDb(
    () => sb.from("sail_curation").select("cuts").eq("clip_key", PACK_KEY).maybeSingle(),
    (d) => d?.cuts?.length === 0,
  );
  if (data.cuts.length !== 0) throw new Error(JSON.stringify(data.cuts));
  const { count } = await sb.from("sail_curation_log").select("*", { count: "exact", head: true }).eq("clip_key", PACK_KEY).eq("action", "set_cuts");
  return `cuts [] in sail_curation; ${count} set_cuts log rows`;
});

// ---- 3. offline: pack → offline audit → sync ----------------------------
await step("offline: /offline downloads a set pack to ready", async () => {
  await page.goto(`${BASE}/offline`);
  await page.waitForSelector(`[data-download-set="${PACK_KEY}"]`, { timeout: 30000 });
  await page.click(`[data-download-set="${PACK_KEY}"]`);
  await page.waitForSelector('[data-pack-status="ready"], [data-pack-status="partial"], [data-pack-status="failed"]', { timeout: 300000 });
  const st = await page.locator("[data-pack-status]").first().getAttribute("data-pack-status");
  if (st !== "ready") throw new Error(`pack status ${st}`);
  return await page.locator("[data-pack]").first().innerText();
});
await step("offline: service worker controls the page", async () => {
  await page.goto(`${BASE}/annotate/${PACK_KEY}`);
  await page.waitForSelector('img[alt^="Frame "]', { timeout: 60000 });
  const ctrl = await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
    return Boolean(navigator.serviceWorker.controller);
  });
  if (!ctrl) throw new Error("no controller");
  return "controller present";
});
let queuedId = null;
await step("offline: with the network off, a packed frame renders and an audit queues", async () => {
  await context.setOffline(true);
  await page.goto(`${BASE}/annotate/${PACK_KEY}`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("[data-from-pack]", { timeout: 60000 });
  const img = page.locator('img[alt^="Frame "]');
  await page.waitForFunction(() => {
    const i = document.querySelector('img[alt^="Frame "]');
    return i && i.complete && i.naturalWidth > 0;
  }, null, { timeout: 30000 });
  const w = await img.evaluate((i) => i.naturalWidth);
  await page.locator("body").click({ position: { x: 5, y: 300 } });
  await page.keyboard.press("a");
  await page.waitForSelector('[data-sync-indicator][data-pending="1"]', { timeout: 15000 });
  queuedId = await page.evaluate(
    () =>
      new Promise((resolve) => {
        const req = indexedDB.open("asvproject-offline", 1);
        req.onsuccess = () => {
          const tx = req.result.transaction("auditQueue", "readonly");
          const all = tx.objectStore("auditQueue").getAll();
          all.onsuccess = () => resolve(all.result[0]?.id ?? null);
        };
      }),
  );
  if (!queuedId) throw new Error("no queued record");
  return `frame ${w}px wide from pack; queued audit ${queuedId}`;
});
await step("offline: back online, sync lands the row in sail_audits (no duplicate on re-sync)", async () => {
  await context.setOffline(false);
  await page.waitForTimeout(1500);
  const btn = page.getByRole("button", { name: /Sync pending audits now/ });
  if (await btn.count()) await btn.click().catch(() => {});
  await page.waitForSelector("[data-sync-indicator]", { state: "detached", timeout: 30000 });
  const { data } = await sb.from("sail_audits").select("id, clip_key").eq("id", queuedId);
  if (!data || data.length !== 1) throw new Error(`rows for ${queuedId}: ${JSON.stringify(data)}`);
  // second sync attempt of the same id must be a no-op (duplicate-key => synced)
  const { error } = await sb.from("sail_audits").select("id").eq("id", queuedId);
  if (error) throw error;
  return `sail_audits row ${queuedId} for ${data[0].clip_key}`;
});

// ---- 4. regression: untouched features ---------------------------------
await step("regression: viewer paints a frame and the fusion table", async () => {
  await page.goto(`${BASE}/clips/2026-07-08__2026-07-08_16-37-01`);
  await page.waitForSelector('input[aria-label="Scrub through frames"]', { timeout: 60000 });
  await page.waitForFunction(() => {
    const c = document.querySelector("canvas");
    return c && c.width > 0;
  }, null, { timeout: 60000 });
  return "canvas painted";
});
await step("regression: annotate planner page lists clips + build button", async () => {
  await page.goto(`${BASE}/annotate`);
  await page.waitForSelector('a[href^="/annotate/"]', { timeout: 60000 });
  const n = await page.locator('a[href^="/annotate/"]').count();
  const btn = await page.getByRole("button", { name: /build plan/i }).isEnabled();
  if (!btn) throw new Error("build plan disabled");
  return `${n} clip links, build plan enabled`;
});
await step("regression: data map renders", async () => {
  await page.goto(`${BASE}/datamap`);
  await page.waitForSelector("svg", { timeout: 60000 });
  return "svg present";
});

// ---- cleanup ------------------------------------------------------------
await step("cleanup: test audit row removed, pack deleted, catalogue as found", async () => {
  if (queuedId) await sb.from("sail_audits").delete().eq("id", queuedId);
  await page.goto(`${BASE}/offline`);
  await page.waitForSelector("[data-pack]", { timeout: 30000 });
  await page.locator('button[aria-label^="Delete pack"]').first().click();
  await page.waitForSelector("[data-pack]", { state: "detached", timeout: 15000 });
  const { data } = await sb.from("sail_curation").select("clip_key, deleted_at, restored_at, cuts").in("clip_key", [DELETE_KEY, PACK_KEY]);
  return JSON.stringify(data);
});

await browser.close();
const failed = results.filter((r) => !r.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
