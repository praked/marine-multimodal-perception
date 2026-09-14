/* Upload predictions.json (crop_id -> {label, p}) to R2 beside the manifest.
 *
 *   node --env-file=.env.local tools/swarm_crops/upload_predictions.mjs \
 *     [--file /Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/predictions.json] \
 *     [--set 2026-08-26]
 *
 * The /crops page picks the new file up on refresh (no deploy needed).
 */
import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { readFile } from "node:fs/promises";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const FILE = opt("file", "/Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/predictions.json");
const SET = opt("set", "2026-08-26");
for (const k of ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]) {
  if (!process.env[k]) {
    console.error(`${k} missing — run with node --env-file=.env.local`);
    process.exit(1);
  }
}
const body = await readFile(FILE);
JSON.parse(body); // refuse to upload a malformed file
const client = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});
await client.send(new PutObjectCommand({
  Bucket: process.env.R2_BUCKET,
  Key: `swarm_crops/${SET}/predictions.json`,
  Body: body,
  ContentType: "application/json",
}));
console.log(`uploaded ${FILE} (${(body.length / 1e6).toFixed(1)} MB) -> swarm_crops/${SET}/predictions.json`);
