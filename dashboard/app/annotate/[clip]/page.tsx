import { AnnotateStage } from "@/components/annotate/AnnotateStage";
import { Suspense } from "react";

export default async function AnnotateClipPage({
  params,
}: {
  params: Promise<{ clip: string }>;
}) {
  const { clip } = await params;
  return (
    <Suspense fallback={<div className="p-8 font-mono text-sm text-subtle">loading…</div>}>
      <AnnotateStage clipKey={clip} />
    </Suspense>
  );
}
