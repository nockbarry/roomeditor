import { useEffect } from "react";
import { useSplatScene } from "../../hooks/useSplatScene.ts";
import { useEditorStore } from "../../stores/editorStore.ts";
import { useProjectStore } from "../../stores/projectStore.ts";
import { useSegmentStore } from "../../stores/segmentStore.ts";
import { useSplatHighlight } from "../../hooks/useSplatHighlight.ts";

interface ViewportProps {
  projectId: string;
}

function formatGaussianCount(n: number | null): string {
  if (!n) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

export function Viewport({ projectId }: ViewportProps) {
  const { containerRef, loading, error, numSplats, loadPly, splatMeshRef, onPickRef } =
    useSplatScene();

  const cameraMode = useEditorStore((s) => s.cameraMode);
  const gaussianCount = useProjectStore((s) => s.currentProject?.gaussian_count ?? null);

  const segmentIndexMap = useSegmentStore((s) => s.segmentIndexMap);
  const segments = useSegmentStore((s) => s.segments);
  const selectSegment = useSegmentStore((s) => s.selectSegment);

  // Wire pick callback → segment selection
  useEffect(() => {
    onPickRef.current = (splatIdx, _point) => {
      if (splatIdx === null || !segmentIndexMap) {
        selectSegment(null);
        return;
      }
      const segValue = segmentIndexMap[splatIdx]; // 0=unassigned, 1+=segment
      if (segValue === 0) {
        selectSegment(null);
        return;
      }
      const seg = segments[segValue - 1];
      selectSegment(seg?.id ?? null);
    };
  }, [segmentIndexMap, segments, selectSegment, onPickRef]);

  // Highlight selected segments
  useSplatHighlight(splatMeshRef);

  useEffect(() => {
    loadPly(`/data/${projectId}/scene.ply?t=${Date.now()}`);
  }, [projectId, loadPly]);

  return (
    <div
      ref={containerRef}
      className="flex-1 relative"
      style={{ minHeight: "400px" }}
    >
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-10 pointer-events-none">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-sm text-gray-400">
              {numSplats > 0
                ? `Loading... ${formatGaussianCount(numSplats)} splats`
                : "Loading 3D scene..."}
            </p>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute top-2 left-2 right-2 bg-red-900/80 text-red-200 text-xs p-2 rounded z-10">
          Failed to load scene: {error}
        </div>
      )}
      <div className="absolute bottom-2 left-2 flex items-center gap-2 z-10">
        <span className="text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded">
          {cameraMode === "orbit" ? "Orbit" : "FPS"} mode
        </span>
        {(numSplats > 0 || gaussianCount) && (
          <span className="text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded">
            {formatGaussianCount(numSplats || gaussianCount)} Gaussians
          </span>
        )}
      </div>
      <div className="absolute bottom-2 right-2 text-[10px] text-gray-600 bg-gray-900/80 px-2 py-1 rounded z-10">
        LMB drag to orbit &middot; Scroll to zoom &middot; RMB drag to pan
      </div>
    </div>
  );
}
