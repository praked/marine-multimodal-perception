/* Shared-password gate. When DASHBOARD_PASSWORD is set, middleware requires
   the asvproject_gate cookie to hold sha256(password + SALT); the /login page
   sets it. When unset (CI, demo previews, local dev) the gate is off.

   This keeps the data hidden from the open internet without a user system —
   the team shares one password. It is NOT hardened auth: rotate the password
   by changing the env var (all cookies invalidate automatically). */

export const GATE_COOKIE = "asvproject_gate";
const SALT = "asvproject-gate-v1";

/** Hex sha256 of the password + salt — works in Node and the edge runtime. */
export async function gateToken(password: string): Promise<string> {
  const data = new TextEncoder().encode(password + SALT);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
