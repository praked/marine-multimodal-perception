/* Apply supabase/migrations/*.sql in filename order over SUPABASE_DB_URL.
 * Idempotent by construction (the migrations use IF NOT EXISTS / drop-then-
 * create policies). Usage:
 *   SUPABASE_DB_URL=postgresql://... node tools/db/apply-migrations.mjs
 */

import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

const here = path.dirname(fileURLToPath(import.meta.url));
const migrationsDir = path.join(here, "..", "..", "supabase", "migrations");

const dbUrl = process.env.SUPABASE_DB_URL;
if (!dbUrl) {
  console.error("SUPABASE_DB_URL is required");
  process.exit(1);
}

const client = new pg.Client({
  connectionString: dbUrl,
  ssl: { rejectUnauthorized: false },
});
await client.connect();

const files = (await readdir(migrationsDir)).filter((f) => f.endsWith(".sql")).sort();
for (const file of files) {
  const sql = await readFile(path.join(migrationsDir, file), "utf8");
  console.log(`applying ${file}…`);
  await client.query(sql);
}
await client.end();
console.log(`done: ${files.length} migration(s) applied`);
