import { useEffect, useRef } from "react";
import { useSplatScene } from "../../hooks/useSplatScene.ts";
import type { TrainingSnapshot } from "../../types/api.ts";

interface ProgressiveViewerProps {
  snapshot: TrainingSnapshot;
}

export function ProgressiveViewer({ snapshot }: ProgressiveViewerProps) {
  const { containerRef, loading, loadPly } = useSplatScene();
  const loadedUrlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!snapshot.snapshot_url) return;
    // Skip if we already loaded this snapshot
    if (loadedUrlRef.current === snapshot.snapshot_url) return;
    loadedUrlRef.current = snapshot.snapshot_url;
    loadPly(`${snapshot.snapshot_url}?t=${Date.now()}`);
  }, [snapshot.snapshot_url, loadPly]);

  const pct = Math.round((snapshot.step / snapshot.total_steps) * 100);

  return (
    <div className="w-full">
      <div
        ref={containerRef}
        className="w-full rounded-lg overflow-hidden bg-gray-900 border border-gray-800 relative"
        style={{ height: "360px" }}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10 pointer-events-none">
            <div className="text-center">
              <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              <p className="text-xs text-gray-400">Loading 3D snapshot...</p>
            </div>
          </div>
        )}
      </div>
      <div className="flex justify-between items-center mt-2 text-xs text-gray-400">
        <span>
          Step {snapshot.step.toLocaleString()} / {snapshot.total_steps.toLocaleString()}
        </span>
        <span>Loss: {snapshot.loss.toFixed(4)}</span>
        <span>{snapshot.n_gaussians.toLocaleString()} Gaussians</span>
      </div>
      <div className="mt-2">
        <div className="w-full bg-gray-800 rounded-full h-1.5">
          <div
            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
