import { useEffect } from "react";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import { useSegmentStore } from "../../stores/segmentStore.ts";
import { useSplatScene } from "../../hooks/useSplatScene.ts";
import { useSplatHighlight } from "../../hooks/useSplatHighlight.ts";
import { Box, Loader2 } from "lucide-react";

interface SplatViewerProps {
  projectId: string;
}

function formatGaussianCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

export function SplatViewer({ projectId }: SplatViewerProps) {
  const { containerRef, loading, error, numSplats, loadPly, splatMeshRef, onPickRef, onHoverRef } =
    useSplatScene();

  const plyUrl = useAnySplatStore((s) => s.plyUrl);
  const plyVersion = useAnySplatStore((s) => s.plyVersion);
  const isRunning = useAnySplatStore((s) => s.isRunning);
  const lastRun = useAnySplatStore((s) => s.lastRun);

  const segmentIndexMap = useSegmentStore((s) => s.segmentIndexMap);
  const segments = useSegmentStore((s) => s.segments);
  const selectSegment = useSegmentStore((s) => s.selectSegment);
  const setHoveredSegment = useSegmentStore((s) => s.setHoveredSegment);
  const hoveredSegmentId = useSegmentStore((s) => s.hoveredSegmentId);

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

  // Wire hover callback → segment hover highlight
  useEffect(() => {
    onHoverRef.current = (splatIdx) => {
      if (splatIdx === null || !segmentIndexMap) {
        setHoveredSegment(null);
        return;
      }
      const segValue = segmentIndexMap[splatIdx];
      if (segValue === 0) {
        setHoveredSegment(null);
        return;
      }
      const seg = segments[segValue - 1];
      setHoveredSegment(seg?.id ?? null);
    };
  }, [segmentIndexMap, segments, setHoveredSegment, onHoverRef]);

  // Highlight selected segments
  useSplatHighlight(splatMeshRef);

  useEffect(() => {
    if (!plyUrl) return;
    loadPly(`${plyUrl}?t=${Date.now()}`);
  }, [plyUrl, plyVersion, loadPly]);

  return (
    <div className={`flex-1 min-h-0 relative rounded-lg overflow-hidden bg-gray-900 border border-gray-800 ${hoveredSegmentId !== null ? "cursor-pointer" : ""}`}>
      {/* Canvas container — always rendered so useSplatScene can initialize */}
      <div ref={containerRef} className="absolute inset-0" />

      {/* Placeholder overlay — shown when no PLY and not running */}
      {!plyUrl && !isRunning && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 z-10">
          <div className="text-center">
            <Box className="w-10 h-10 mx-auto mb-3 text-gray-700" />
            <p className="text-sm text-gray-500">No result yet</p>
            <p className="text-xs text-gray-600 mt-1">Select frames and click Rebuild</p>
          </div>
        </div>
      )}

      {/* Loading/running overlay */}
      {(loading || isRunning) && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10 pointer-events-none">
          <div className="text-center">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-emerald-400" />
            <p className="text-xs text-gray-400">
              {isRunning
                ? "Running AnySplat..."
                : numSplats > 0
                  ? `Loading... ${formatGaussianCount(numSplats)} splats`
                  : "Loading 3D scene..."}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute top-2 left-2 right-2 bg-red-900/80 text-red-200 text-xs p-2 rounded z-10">
          Failed to load: {error}
        </div>
      )}

      {lastRun && !loading && (
        <div className="absolute bottom-2 left-2 flex items-center gap-2 z-10">
          <span className="text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded">
            {formatGaussianCount(numSplats || lastRun.n_gaussians)} Gaussians
          </span>
        </div>
      )}

      <div className="absolute bottom-2 right-2 text-[10px] text-gray-600 bg-gray-900/80 px-2 py-1 rounded z-10">
        LMB drag to orbit &middot; Scroll to zoom &middot; RMB drag to pan
      </div>
    </div>
  );
}
