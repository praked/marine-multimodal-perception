/* Live verification of the /crops swarm self-recognition review page,
 * run against the deployed dashboard behind the password gate.
 *
 *   cd dashboard && node tools/verify/crops.mjs [--base https://...]
 *
 * SAFE AGAINST A CONCURRENT LABELLING SESSION: all write steps happen at a
 * far review-order position (~#60000) that a human frontier won't reach,
 * so the script never labels — or cleans up — crops near where AuthorTwo is
 * working. (A 2026-09-03 run raced his live session at the frontier and
 * its cleanup deleted one of his labels, since restored.)
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
const FAR_POS = 60000; // review-order position for all write steps

const MANIFEST_PATH =
  process.env.CROPS_MANIFEST ??
  "/Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/manifest.jsonl";
const manifestRows = existsSync(MANIFEST_PATH)
  ? readFileSync(MANIFEST_PATH, "utf8")
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l))
  : null;
const byId = manifestRows ? new Map(manifestRows.map((r) => [r.crop_id, r])) : null;
const PREDICTIONS_PATH =
  process.env.CROPS_PREDICTIONS ??
  "/Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/predictions.json";
const predictions = existsSync(PREDICTIONS_PATH)
  ? JSON.parse(readFileSync(PREDICTIONS_PATH, "utf8"))
  : null;
// replicate lib/crops orderForReview: stable partition, giants (>55%) last
let reviewOrder = null;
if (manifestRows) {
  const giants = [];
  const normal = [];
  for (const r of manifestRows) {
    const [x0, y0, x1, y1] = r.window;
    (((x1 - x0) * (y1 - y0)) / (864 * 648) > 0.55 ? giants : normal).push(r);
  }
  reviewOrder = [...normal, ...giants];
}

const sb = createClient(env.NEXT_PUBLIC_SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

const results = [];
function ok(name, pass, detail = "") {
  results.push({ name, pass });
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
async function waitDb(query, pred, timeoutMs = 15000) {
  const t0 = Date.now();
  for (;;) {
    const { data } = await query();
    if (pred(data)) return data;
    if (Date.now() - t0 > timeoutMs) return data;
    await new Promise((r) => setTimeout(r, 500));
  }
}
const currentCropId = (page) =>
  page.locator("[data-crop-stage]").getAttribute("data-crop-id");
async function jumpTo(page, n) {
  await page.fill("[data-goto-frame]", String(n));
  await page.press("[data-goto-frame]", "Enter");
  await page.waitForFunction(
    (want) => document.querySelector("[data-index]")?.textContent?.startsWith(`${want}/`),
    n,
    { timeout: 10000 },
  );
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await context.newPage();
page.on("pageerror", (e) => console.log("  [pageerror]", e.message));

// ---- login ----------------------------------------------------------------
await page.goto(`${BASE}/login`);
await page.fill('input[name="password"]', PASSWORD);
await Promise.all([
  page.waitForURL((u) => !u.pathname.startsWith("/login")),
  page.keyboard.press("Enter"),
]);
ok("login through the gate", !page.url().includes("/login"), page.url());

// ---- read-only checks at the frontier -------------------------------------
await step("/crops loads on the first unlabelled crop", async () => {
  await page.goto(`${BASE}/crops`);
  await page.waitForSelector("[data-crop-stage]", { timeout: 60000 });
  const id = await currentCropId(page);
  const progress = await page.locator("[data-progress]").innerText();
  return `crop ${id.slice(0, 8)}…, ${progress}`;
});

await step("first crop is NOT a giant (near-frame boxes queued last)", async () => {
  if (!byId) throw new Error("local manifest needed");
  const row = byId.get(await currentCropId(page));
  if (!row) throw new Error("current crop not in local manifest");
  const [x0, y0, x1, y1] = row.window;
  const frac = ((x1 - x0) * (y1 - y0)) / (864 * 648);
  if (frac > 0.55) throw new Error(`window covers ${(frac * 100).toFixed(0)}% of the frame`);
  return `window ${x1 - x0}x${y1 - y0} = ${(frac * 100).toFixed(1)}% of the frame`;
});

await step("full-width progress bar with 'k/N labelled' text", async () => {
  const bar = page.locator("[data-progress-bar]");
  await bar.waitFor({ timeout: 15000 });
  const text = await bar.innerText();
  if (!/\d+\/\d+ labelled/.test(text)) throw new Error(`bar text: ${text}`);
  return text.trim();
});

// ---- model assist ---------------------------------------------------------
if (predictions) {
  await step("review-queue default puts a model-accept first", async () => {
    const pressed = await page
      .locator('[data-order-toggle] button[aria-pressed="true"]')
      .innerText();
    if (pressed.trim() !== "review queue") throw new Error(`default mode: ${pressed}`);
    const id = await currentCropId(page);
    const pr = predictions[id];
    if (!pr) throw new Error(`no local prediction for ${id}`);
    if (pr.label !== "accept") throw new Error(`first crop predicted ${pr.label}`);
    return `first crop ${id.slice(0, 8)}… predicted accept p=${pr.p}`;
  });

  await step("model chip renders from predictions", async () => {
    const chip = page.locator("[data-model-chip]");
    await chip.waitFor({ timeout: 15000 });
    const text = (await chip.innerText()).trim();
    if (!/^model: (OURS|OTHER|BAD BOX) \d\.\d\d$/.test(text)) {
      throw new Error(`chip text: ${text}`);
    }
    return text;
  });

  await step("'model agrees k/n' header stat present", async () => {
    const stat = page.locator("[data-model-agree]");
    await stat.waitFor({ timeout: 15000 });
    const text = (await stat.innerText()).trim();
    if (!/^model agrees \d+\/\d+$/.test(text)) throw new Error(text);
    return text;
  });

  await step("ordering toggle persists across reload", async () => {
    await page.locator('[data-order-toggle] button', { hasText: "chronological" }).click();
    await page.goto(`${BASE}/crops`);
    await page.waitForSelector("[data-crop-stage]", { timeout: 60000 });
    const pressed = await page
      .locator('[data-order-toggle] button[aria-pressed="true"]')
      .innerText();
    if (pressed.trim() !== "chronological") throw new Error(`after reload: ${pressed}`);
    // stay in chronological: the jump + write steps below address the base
    // review order (this context's localStorage is ephemeral, not AuthorTwo's)
    return "chronological survived reload";
  });
} else {
  console.log("SKIP  model-assist steps (no local predictions.json yet)");
}

// ---- type-to-jump ---------------------------------------------------------
await step(`type-to-jump lands on review-order position ${FAR_POS}`, async () => {
  if (!reviewOrder) throw new Error("local manifest needed");
  // NB: runs in chronological mode (the model steps above end there), so
  // positions address the giant-demoted base review order replicated here
  await jumpTo(page, FAR_POS);
  const id = await currentCropId(page);
  const want = reviewOrder[FAR_POS - 1].crop_id;
  if (id !== want) throw new Error(`position ${FAR_POS}: got ${id}, expected ${want}`);
  return `#${FAR_POS} -> ${id.slice(0, 8)}… (matches local review order)`;
});

await step("global hotkeys are inert while the jump input is focused", async () => {
  const id = await currentCropId(page);
  await page.focus("[data-goto-frame]");
  await page.keyboard.press("a"); // must NOT label while typing
  await page.keyboard.press("Escape");
  await page.locator("[data-crop-stage]").click({ position: { x: 5, y: 5 } });
  await new Promise((r) => setTimeout(r, 1500));
  const { data } = await sb.from("sail_crop_labels").select("id").eq("crop_id", id);
  if ((data ?? []).length > 0) throw new Error("typing in the input labelled the crop");
  const still = await currentCropId(page);
  if (still !== id) throw new Error("focused input still advanced the crop");
  return "a-in-input wrote nothing and did not advance";
});

// ---- crop rendering + overlay at the far position -------------------------
await step("crop image renders (signed R2 asset, natural size > 0)", async () => {
  await page.waitForFunction(
    () => {
      const el = document.querySelector("[data-crop-stage] img");
      return el && el.complete && el.naturalWidth > 0;
    },
    { timeout: 30000 },
  );
  return await page
    .locator("[data-crop-stage] img")
    .evaluate((el) => `${el.naturalWidth}x${el.naturalHeight}`);
});

await step("detection-box overlay matches the manifest window math", async () => {
  const id = await currentCropId(page);
  const rect = page.locator("[data-overlay-box]");
  await rect.waitFor({ timeout: 15000 });
  const got = {
    x: Number(await rect.getAttribute("x")),
    y: Number(await rect.getAttribute("y")),
  };
  if (!byId) return `rendered at ${JSON.stringify(got)} (no local manifest)`;
  const row = byId.get(id);
  const [wx0, wy0, wx1, wy1] = row.window;
  // the client reads the COMPACT manifest, whose xyxy is rounded to 4 dp
  // (~0.04 frame px) — compare against the same rounding
  const rx = (v) => Math.round(v * 1e4) / 1e4;
  const exp = {
    x: Math.max(0, (rx(row.xyxy[0]) * 864 - wx0) / (wx1 - wx0)),
    y: Math.max(0, (rx(row.xyxy[1]) * 648 - wy0) / (wy1 - wy0)),
  };
  for (const k of ["x", "y"]) {
    if (Math.abs(got[k] - exp[k]) > 1e-3) {
      throw new Error(`${k}: got ${got[k]}, expected ${exp[k]}`);
    }
  }
  return `overlay at (${got.x.toFixed(3)},${got.y.toFixed(3)}) — matches manifest`;
});

await step("context thumbnail shows the full frame with the box", async () => {
  const thumb = page.locator("[data-context-thumb] img");
  await thumb.waitFor({ timeout: 20000 });
  await page.waitForFunction(
    () => {
      const el = document.querySelector("[data-context-thumb] img");
      return el && el.complete && el.naturalWidth > 0;
    },
    { timeout: 20000 },
  );
  return `frame thumbnail ${await thumb.evaluate((el) => `${el.naturalWidth}x${el.naturalHeight}`)}`;
});

await step("screenshots of sample crops for eyeballing", async () => {
  const dir = process.env.CROPS_SHOT_DIR ?? "/tmp";
  const shots = [];
  for (let k = 0; k < 3; k++) {
    await page.screenshot({ path: `${dir}/crops_verify_${k}.png` });
    shots.push(`${dir}/crops_verify_${k}.png`);
    for (let j = 0; j < 3; j++) await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(800);
  }
  return shots.join(" ");
});

// ---- write steps, far from any human frontier -----------------------------
await jumpTo(page, FAR_POS);
const testCropIds = new Set();

let farCropId = await currentCropId(page);
await step("`a` appends an accept row and advances", async () => {
  testCropIds.add(farCropId);
  await page.keyboard.press("a");
  const nextId = await currentCropId(page);
  if (nextId === farCropId) throw new Error("did not advance");
  const rows = await waitDb(
    () => sb.from("sail_crop_labels").select("label").eq("crop_id", farCropId).eq("label", "accept"),
    (d) => (d ?? []).length > 0,
  );
  if (!rows?.length) throw new Error("no accept row landed");
  return `accept row for ${farCropId.slice(0, 8)}…`;
});

await step("header shows a green tick on the just-labelled crop (via undo)", async () => {
  await page.keyboard.press("u");
  const chip = page.locator("[data-verdict-check]");
  await chip.waitFor({ timeout: 10000 });
  const text = (await chip.innerText()).trim();
  if (text !== "✓") throw new Error(`chip shows "${text}", expected ✓`);
  return "✓ chip on the accepted crop";
});

await step("`x` records deny; undo returns and shows the verdict word", async () => {
  await jumpTo(page, FAR_POS + 1);
  const id = await currentCropId(page);
  testCropIds.add(id);
  await page.keyboard.press("x");
  await page.keyboard.press("u");
  if ((await currentCropId(page)) !== id) throw new Error("undo landed elsewhere");
  const rows = await waitDb(
    () => sb.from("sail_crop_labels").select("label").eq("crop_id", id).eq("label", "deny"),
    (d) => (d ?? []).length > 0,
  );
  if (!rows?.length) throw new Error("deny row missing");
  const chip = (await page.locator("[data-verdict-check]").innerText()).trim();
  if (chip !== "other boat") throw new Error(`chip shows "${chip}"`);
  return `deny row for ${id.slice(0, 8)}…, chip "other boat"`;
});

await step("`f` records a junk (bad box) label", async () => {
  await jumpTo(page, FAR_POS + 2);
  const id = await currentCropId(page);
  testCropIds.add(id);
  await page.keyboard.press("f");
  const rows = await waitDb(
    () => sb.from("sail_crop_labels").select("label").eq("crop_id", id).eq("label", "junk"),
    (d) => (d ?? []).length > 0,
  );
  if (!rows?.length) throw new Error("junk row missing");
  return `junk row for ${id.slice(0, 8)}…`;
});

await step("refresh resumes at the first unlabelled crop (not a test crop)", async () => {
  await page.goto(`${BASE}/crops`);
  await page.waitForSelector("[data-crop-stage]", { timeout: 60000 });
  const id = await currentCropId(page);
  if (testCropIds.has(id)) throw new Error("resumed on a labelled test crop");
  return `landed on ${id.slice(0, 8)}…`;
});

// ---- cleanup (far-position crops only — never a human's) ------------------
for (const id of testCropIds) {
  if (id) await sb.from("sail_crop_labels").delete().eq("crop_id", id);
}
console.log(`cleanup: ${testCropIds.size} far-position test crops cleared`);

await browser.close();
const failed = results.filter((r) => !r.pass);
console.log(failed.length === 0 ? `\nALL ${results.length} PASS` : `\n${failed.length} FAILED`);
process.exit(failed.length === 0 ? 0 : 1);
