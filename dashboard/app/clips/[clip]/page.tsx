import { ClipViewer } from "@/components/viewer/ClipViewer";

export default async function ClipPage({
  params,
}: {
  params: Promise<{ clip: string }>;
}) {
  const { clip } = await params;
  return <ClipViewer clipKey={clip} />;
}
