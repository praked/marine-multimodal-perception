/* Shared annotation plans (sail_plans): plans built in ANY browser/session
   appear in every auditor's planner. The shared row is the plan DEFINITION
   (name + frames); per-auditor progress cursors stay in localStorage
   (lib/annotate/planner.ts library). Demo mode: shared store is silently
   absent and plans stay browser-local. */
import { isSupabaseConfigured } from "@/lib/data";
import { createClient } from "@/lib/supabase/client";
import type { StoredPlan } from "@/lib/annotate/planner";

export interface SharedPlan {
  id: string;
  name: string;
  requested: number;
  frames: { clipKey: string; ts: string }[];
  stats?: StoredPlan["stats"];
  clamped: boolean;
  created_at: string;
}

export async function listSharedPlans(): Promise<SharedPlan[]> {
  if (!isSupabaseConfigured()) return [];
  const { data, error } = await createClient()
    .from("sail_plans")
    .select("id,name,requested,frames,stats,clamped,created_at")
    .order("created_at", { ascending: true });
  if (error) throw error;
  return (data ?? []) as SharedPlan[];
}

/** Insert; returns the shared id (adopted as the local plan id). */
export async function pushSharedPlan(p: StoredPlan): Promise<string | null> {
  if (!isSupabaseConfigured()) return null;
  const { data, error } = await createClient()
    .from("sail_plans")
    .insert({
      name: p.name ?? "plan",
      requested: p.requested,
      frames: p.frames,
      stats: p.stats ?? null,
      clamped: p.clamped ?? false,
    })
    .select("id")
    .single();
  if (error) throw error;
  return (data as { id: string }).id;
}

export async function renameSharedPlan(id: string, name: string): Promise<void> {
  if (!isSupabaseConfigured()) return;
  const { error } = await createClient().from("sail_plans").update({ name }).eq("id", id);
  if (error) throw error;
}

export async function deleteSharedPlan(id: string): Promise<void> {
  if (!isSupabaseConfigured()) return;
  const { error } = await createClient().from("sail_plans").delete().eq("id", id);
  if (error) throw error;
}
