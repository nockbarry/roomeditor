import { useCallback, useMemo, useRef, useState } from "react";
import { useAnySplatStore, computeUsedFrameIndices } from "../../stores/anysplatStore.ts";
import { CheckSquare, Square, Loader2, Film, ImageIcon, ChevronDown, ChevronRight, RefreshCw } from "lucide-react";
import type { FrameInfo } from "../../types/api.ts";

interface FrameGalleryProps {
  projectId: string;
}

interface FrameGroup {
  source: string;
  sourceType: string;
  frames: { frame: FrameInfo; globalIndex: number }[];
}

function groupFramesBySource(frames: FrameInfo[]): FrameGroup[] {
  if (frames.length === 0) return [];

  const groups: FrameGroup[] = [];
  let currentSource = frames[0].source_file || "";
  let currentType = frames[0].source_type || "";
  let currentGroup: FrameGroup = {
    source: currentSource || "Unknown",
    sourceType: currentType,
    frames: [],
  };

  for (let i = 0; i < frames.length; i++) {
    const src = frames[i].source_file || "";
    if (src !== currentSource) {
      if (currentGroup.frames.length > 0) groups.push(currentGroup);
      currentSource = src;
      currentType = frames[i].source_type || "";
      currentGroup = {
        source: src || "Unknown",
        sourceType: currentType,
        frames: [],
      };
    }
    currentGroup.frames.push({ frame: frames[i], globalIndex: i });
  }
  if (currentGroup.frames.length > 0) groups.push(currentGroup);

  return groups;
}

export function FrameGallery({ projectId }: FrameGalleryProps) {
  const frames = useAnySplatStore((s) => s.frames);
  const selectedCount = useAnySplatStore((s) => s.selectedCount);
  const framesLoading = useAnySplatStore((s) => s.framesLoading);
  const maxViews = useAnySplatStore((s) => s.maxViews);
  const chunked = useAnySplatStore((s) => s.chunked);
  const toggleFrame = useAnySplatStore((s) => s.toggleFrame);
  const toggleAllFrames = useAnySplatStore((s) => s.toggleAllFrames);
  const extractFrames = useAnySplatStore((s) => s.extractFrames);
  const fps = useAnySplatStore((s) => s.fps);
  const lastClickedRef = useRef<number | null>(null);

  const groups = useMemo(() => groupFramesBySource(frames), [frames]);
  const hasMultipleGroups = groups.length > 1;

  // Track collapsed groups by source name
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggleCollapsed = useCallback((source: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(source)) next.delete(source);
      else next.add(source);
      return next;
    });
  }, []);

  // Compute which frames will actually be used in the next run
  const usedIndices = useMemo(
    () => computeUsedFrameIndices(frames, maxViews, chunked),
    [frames, maxViews, chunked]
  );
  const usedCount = usedIndices.size;
  const isSubsampled = !chunked && selectedCount > maxViews;

  const handleClick = useCallback(
    (globalIndex: number, e: React.MouseEvent) => {
      const frame = frames[globalIndex];

      if (e.shiftKey && lastClickedRef.current !== null) {
        const start = Math.min(lastClickedRef.current, globalIndex);
        const end = Math.max(lastClickedRef.current, globalIndex);
        const updates: Record<string, boolean> = {};
        for (let i = start; i <= end; i++) {
          updates[frames[i].filename] = !frame.selected;
        }
        import("../../api/anysplat.ts").then(({ updateFrameSelection }) => {
          updateFrameSelection(projectId, updates).then((manifest) => {
            useAnySplatStore.setState({
              frames: manifest.frames,
              selectedCount: manifest.selected_count,
            });
          });
        });
      } else {
        toggleFrame(projectId, frame.filename, !frame.selected);
      }

      lastClickedRef.current = globalIndex;
    },
    [frames, projectId, toggleFrame]
  );

  const handleGroupToggle = useCallback(
    (group: FrameGroup, selectAll: boolean) => {
      const updates: Record<string, boolean> = {};
      for (const { frame } of group.frames) {
        updates[frame.filename] = selectAll;
      }
      import("../../api/anysplat.ts").then(({ updateFrameSelection }) => {
        updateFrameSelection(projectId, updates).then((manifest) => {
          useAnySplatStore.setState({
            frames: manifest.frames,
            selectedCount: manifest.selected_count,
          });
        });
      });
    },
    [projectId]
  );

  if (framesLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-5 h-5 animate-spin mx-auto mb-2 text-gray-500" />
          <p className="text-xs text-gray-500">Extracting frames...</p>
        </div>
      </div>
    );
  }

  if (frames.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-xs text-gray-600">No frames yet</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Global header */}
      <div className="px-3 py-2 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">
            {selectedCount} of {frames.length} selected
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => toggleAllFrames(projectId, true)}
              className="text-[10px] text-gray-500 hover:text-white px-1"
              title="Select all"
            >
              <CheckSquare className="w-3 h-3" />
            </button>
            <button
              onClick={() => toggleAllFrames(projectId, false)}
              className="text-[10px] text-gray-500 hover:text-white px-1"
              title="Deselect all"
            >
              <Square className="w-3 h-3" />
            </button>
          </div>
        </div>
        {isSubsampled && (
          <div className="mt-1.5 text-[10px] text-amber-400/80 bg-amber-900/10 rounded px-2 py-1.5">
            <strong>{usedCount}</strong> of {selectedCount} selected frames will be evenly sampled for reconstruction.
            Deselecting frames may shift which ones are used.
            {!chunked && " Switch to Chunked mode to use all selected frames."}
          </div>
        )}
      </div>

      {/* Frame groups */}
      <div className="flex-1 overflow-y-auto">
        {groups.map((group) => {
          const groupSelectedCount = group.frames.filter(
            ({ frame }) => frame.selected
          ).length;
          const groupUsedCount = group.frames.filter(
            ({ globalIndex }) => usedIndices.has(globalIndex)
          ).length;
          const isCollapsed = collapsed.has(group.source);

          return (
            <div key={group.source}>
              {/* Group header */}
              {hasMultipleGroups && (
                <div className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-900/70 border-b border-gray-800/50 sticky top-0 z-10">
                  <button
                    onClick={() => toggleCollapsed(group.source)}
                    className="text-gray-500 hover:text-white"
                  >
                    {isCollapsed ? (
                      <ChevronRight className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </button>
                  {/* Source type badge */}
                  <span className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-medium ${
                    group.sourceType === "video"
                      ? "bg-blue-900/40 text-blue-400"
                      : "bg-green-900/40 text-green-400"
                  }`}>
                    {group.sourceType || "file"}
                  </span>
                  <span className="text-[11px] text-gray-300 truncate flex-1">
                    {group.source}
                  </span>
                  <span className="text-[10px] text-gray-500 shrink-0">
                    {isSubsampled
                      ? `${groupUsedCount} used / ${groupSelectedCount} sel / ${group.frames.length}`
                      : `${groupSelectedCount}/${group.frames.length}`
                    }
                  </span>
                  {/* Re-extract button for video sources */}
                  {group.sourceType === "video" && (
                    <button
                      onClick={() => extractFrames(projectId, fps)}
                      className="text-gray-600 hover:text-white p-0.5"
                      title={`Re-extract frames from ${group.source}`}
                    >
                      <RefreshCw className="w-2.5 h-2.5" />
                    </button>
                  )}
                  <button
                    onClick={() => handleGroupToggle(group, true)}
                    className="text-gray-600 hover:text-white p-0.5"
                    title={`Select all from ${group.source}`}
                  >
                    <CheckSquare className="w-2.5 h-2.5" />
                  </button>
                  <button
                    onClick={() => handleGroupToggle(group, false)}
                    className="text-gray-600 hover:text-white p-0.5"
                    title={`Deselect all from ${group.source}`}
                  >
                    <Square className="w-2.5 h-2.5" />
                  </button>
                </div>
              )}

              {/* Thumbnail grid */}
              {!isCollapsed && (
                <div className="p-2">
                  <div className="grid grid-cols-3 gap-1">
                    {group.frames.map(({ frame, globalIndex }) => {
                      const isUsed = usedIndices.has(globalIndex);
                      const isSelected = frame.selected;
                      const isSkipped = isSelected && !isUsed;

                      return (
                        <button
                          key={frame.filename}
                          onClick={(e) => handleClick(globalIndex, e)}
                          className={`relative aspect-square rounded overflow-hidden border transition-colors ${
                            isSelected
                              ? "border-emerald-500/50"
                              : "border-gray-800 opacity-40"
                          }`}
                          title={`${frame.filename} — sharpness: ${frame.sharpness.toFixed(1)}${isSkipped ? " (will be skipped)" : ""}`}
                        >
                          <img
                            src={`/data/${projectId}/frames/${frame.filename}`}
                            alt={frame.filename}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                          {frame.sharpness > 0 && frame.sharpness < 50 && (
                            <div className="absolute inset-0 border-2 border-red-500/60 rounded pointer-events-none" />
                          )}
                          {/* Corner icon: check for selected, square for unselected */}
                          <div className="absolute top-0.5 right-0.5">
                            {isSelected ? (
                              <CheckSquare className="w-3 h-3 text-emerald-400" />
                            ) : (
                              <Square className="w-3 h-3 text-gray-600" />
                            )}
                          </div>
                          {/* Small "used" dot when subsampled — only indicator that shifts */}
                          {isSubsampled && isUsed && (
                            <div className="absolute bottom-0.5 left-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          )}
                          {isSubsampled && isSkipped && (
                            <div className="absolute bottom-0.5 left-0.5 w-1.5 h-1.5 rounded-full bg-amber-400/60" />
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
