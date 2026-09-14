// Verify the union label set + teacher fields serve from production.
import { readFileSync } from "node:fs";
import { chromium } from "playwright";
const env = Object.fromEntries(readFileSync(".env.local", "utf8").split("\n").filter(l => l.includes("=") && !l.startsWith("#")).map(l => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]));
const BASE = "https://asvproject-dashboard.vercel.app";
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`${BASE}/login`);
await page.fill("input[type=password]", env.DASHBOARD_PASSWORD);
await Promise.all([page.waitForURL(u => !u.pathname.startsWith("/login")), page.keyboard.press("Enter")]);
console.log("login:", !page.url().includes("/login"));
// fetch one 07-08 activity's labels.json through the app's asset API
for (const clip of ["2026-07-08__2026-07-08_16-37-01", "2026-08-26_afloat__2026-08-26_19-22-29"]) {
  const labels = await page.evaluate(async (c) => {
    const r = await fetch(`/api/assets/${c}/labels.json`);
    return r.ok ? r.json() : { __status: r.status };
  }, clip);
  if (labels.__status) { console.log(clip, "labels fetch status", labels.__status); continue; }
  const recs = Object.values(labels);
  const srcs = {}; let polys = 0, prompts = 0;
  for (const rec of recs) for (const b of rec.fisheye_bboxes ?? []) {
    srcs[b.source ?? "?"] = (srcs[b.source ?? "?"] ?? 0) + 1;
    if (b.polygon?.length >= 6) polys++;
    if (b.prompt) prompts++;
  }
  console.log(clip, "records", recs.length, "union?", recs[0]?.source, "box sources", JSON.stringify(srcs), "polygons", polys, "prompts", prompts);
}
// annotate page renders with the teacher chips
await page.goto(`${BASE}/annotate/2026-07-08__2026-07-08_16-37-01`);
await page.waitForTimeout(4000);
const chips = await page.evaluate(() => [...document.querySelectorAll("button")].map(b => b.textContent?.trim()).filter(t => ["all","DINO","DART","agreed"].includes(t ?? "")));
console.log("teacher chips on annotate:", JSON.stringify(chips));
await browser.close();
