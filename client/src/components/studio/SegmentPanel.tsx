import { useSegmentStore } from "../../stores/segmentStore.ts";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import {
  Scan,
  Loader2,
  Trash2,
  Link,
  Eye,
  EyeOff,
  ChevronRight,
  Undo2,
  RotateCcw,
  Copy,
  Layers,
} from "lucide-react";
import { useState, useEffect, useCallback } from "react";

interface SegmentPanelProps {
  projectId: string;
}

const STEP_OPTIONS = [0.01, 0.05, 0.1, 0.5];

export function SegmentPanel({ projectId }: SegmentPanelProps) {
  const segments = useSegmentStore((s) => s.segments);
  const selectedSegmentIds = useSegmentStore((s) => s.selectedSegmentIds);
  const hoveredSegmentId = useSegmentStore((s) => s.hoveredSegmentId);
  const isSegmenting = useSegmentStore((s) => s.isSegmenting);
  const isAssigning = useSegmentStore((s) => s.isAssigning);
  const totalGaussians = useSegmentStore((s) => s.totalGaussians);
  const unassignedGaussians = useSegmentStore((s) => s.unassignedGaussians);
  const undoCount = useSegmentStore((s) => s.undoCount);
  const autoSegment = useSegmentStore((s) => s.autoSegment);
  const assignGaussians = useSegmentStore((s) => s.assignGaussians);
  const selectSegment = useSegmentStore((s) => s.selectSegment);
  const toggleSelectSegment = useSegmentStore((s) => s.toggleSelectSegment);
  const rangeSelectSegment = useSegmentStore((s) => s.rangeSelectSegment);
  const deleteSegment = useSegmentStore((s) => s.deleteSegment);
  const batchTransform = useSegmentStore((s) => s.batchTransform);
  const toggleVisibility = useSegmentStore((s) => s.toggleVisibility);
  const duplicateSegment = useSegmentStore((s) => s.duplicateSegment);
  const renameSegment = useSegmentStore((s) => s.renameSegment);
  const createBackground = useSegmentStore((s) => s.createBackground);
  const undo = useSegmentStore((s) => s.undo);
  const restoreOriginal = useSegmentStore((s) => s.restoreOriginal);
  const fetchUndoCount = useSegmentStore((s) => s.fetchUndoCount);
  const setHoveredSegment = useSegmentStore((s) => s.setHoveredSegment);

  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [editingLabelId, setEditingLabelId] = useState<number | null>(null);
  const [editLabel, setEditLabel] = useState("");
  const [step, setStep] = useState(0.1);

  useEffect(() => {
    fetchUndoCount(projectId);
  }, [projectId, fetchUndoCount]);

  const bumpPly = useCallback(() => {
    useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
  }, []);

  const handleSegmentClick = (segId: number, e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      toggleSelectSegment(segId);
    } else if (e.shiftKey) {
      rangeSelectSegment(segId);
    } else {
      selectSegment(segId);
      setExpandedId(expandedId === segId ? null : segId);
    }
  };

  const handleDelete = async (segId: number) => {
    await deleteSegment(projectId, segId);
    bumpPly();
  };

  const handleTransform = async (type: "translation" | "rotation" | "scale", values: number[]) => {
    await batchTransform(projectId, { [type]: values });
    bumpPly();
  };

  const handleVisibility = async (segId: number, visible: boolean) => {
    await toggleVisibility(projectId, segId, visible);
    bumpPly();
  };

  const handleDuplicate = async (segId: number) => {
    await duplicateSegment(projectId, segId);
    bumpPly();
  };

  const handleRename = async (segId: number) => {
    if (editLabel.trim()) {
      await renameSegment(projectId, segId, editLabel.trim());
    }
    setEditingLabelId(null);
  };

  const handleUndo = async () => {
    await undo(projectId);
    bumpPly();
  };

  const handleRestoreOriginal = async () => {
    await restoreOriginal(projectId);
    bumpPly();
  };

  const handleCreateBackground = async () => {
    await createBackground(projectId);
  };

  const selectedSegments = segments.filter((s) => selectedSegmentIds.includes(s.id));
  const hasGaussians = selectedSegments.some((s) => s.n_gaussians > 0);

  return (
    <div className="flex flex-col h-full">
      {/* Header with undo/restore */}
      <div className="px-3 py-2 border-b border-gray-800 flex items-center gap-1">
        <button
          onClick={handleUndo}
          disabled={undoCount === 0}
          title={undoCount > 0 ? `Undo (${undoCount} available)` : "Nothing to undo"}
          className="p-1.5 rounded hover:bg-gray-800 disabled:opacity-30 transition-colors"
        >
          <Undo2 className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={handleRestoreOriginal}
          title="Restore original"
          className="p-1.5 rounded hover:bg-gray-800 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
        {undoCount > 0 && (
          <span className="text-[10px] text-gray-600 ml-1">{undoCount} undo</span>
        )}
        <div className="flex-1" />
        <label className="text-[10px] text-gray-600 mr-1">Step</label>
        <select
          value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          className="bg-gray-900 border border-gray-800 rounded px-1 py-0.5 text-[10px] text-gray-400 focus:outline-none"
        >
          {STEP_OPTIONS.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* Segment action buttons */}
      <div className="px-3 py-3 border-b border-gray-800 space-y-2">
        <button
          onClick={() => autoSegment(projectId)}
          disabled={isSegmenting}
          className="w-full bg-violet-600 hover:bg-violet-500 disabled:opacity-50 px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2"
        >
          {isSegmenting ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Detecting objects...
            </>
          ) : (
            <>
              <Scan className="w-3.5 h-3.5" />
              Detect Objects (SAM2)
            </>
          )}
        </button>

        {segments.length > 0 && (
          <button
            onClick={() => assignGaussians(projectId)}
            disabled={isAssigning}
            className="w-full bg-gray-800 hover:bg-gray-700 disabled:opacity-50 px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2"
          >
            {isAssigning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Assigning to 3D...
              </>
            ) : (
              <>
                <Link className="w-3.5 h-3.5" />
                Assign to Gaussians
              </>
            )}
          </button>
        )}

        {unassignedGaussians > 0 && (
          <button
            onClick={handleCreateBackground}
            className="w-full bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
          >
            <Layers className="w-3.5 h-3.5" />
            Create Background ({unassignedGaussians.toLocaleString()} gs)
          </button>
        )}
      </div>

      {/* Transform controls for selection */}
      {selectedSegmentIds.length > 0 && hasGaussians && (
        <div className="px-3 py-2 border-b border-gray-800 space-y-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">
            Transform ({selectedSegmentIds.length} selected)
          </div>

          {/* Translate */}
          <div>
            <div className="text-[10px] text-gray-600 mb-1">Translate</div>
            <div className="grid grid-cols-3 gap-1">
              {(["X", "Y", "Z"] as const).map((axis, i) => (
                <div key={axis} className="flex gap-0.5">
                  <button
                    onClick={() => {
                      const t = [0, 0, 0]; t[i] = -step;
                      handleTransform("translation", t);
                    }}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                  >
                    -{axis}
                  </button>
                  <button
                    onClick={() => {
                      const t = [0, 0, 0]; t[i] = step;
                      handleTransform("translation", t);
                    }}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                  >
                    +{axis}
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Rotate */}
          <div>
            <div className="text-[10px] text-gray-600 mb-1">Rotate (deg)</div>
            <div className="grid grid-cols-3 gap-1">
              {(["X", "Y", "Z"] as const).map((axis, i) => (
                <div key={axis} className="flex gap-0.5">
                  <button
                    onClick={() => {
                      const r = [0, 0, 0]; r[i] = -15;
                      handleTransform("rotation", r);
                    }}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                  >
                    -{axis}
                  </button>
                  <button
                    onClick={() => {
                      const r = [0, 0, 0]; r[i] = 15;
                      handleTransform("rotation", r);
                    }}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                  >
                    +{axis}
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Scale */}
          <div>
            <div className="text-[10px] text-gray-600 mb-1">Scale</div>
            <div className="grid grid-cols-3 gap-1">
              {(["X", "Y", "Z"] as const).map((axis, i) => (
                <div key={axis} className="flex gap-0.5">
                  <button
                    onClick={() => {
                      const s = [1, 1, 1]; s[i] = 1 / 1.1;
                      handleTransform("scale", s);
                    }}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                  >
                    -{axis}
                  </button>
                  <button
                    onClick={() => {
                      const s = [1, 1, 1]; s[i] = 1.1;
                      handleTransform("scale", s);
                    }}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                  >
                    +{axis}
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Segment list */}
      <div className="flex-1 overflow-y-auto">
        {segments.length === 0 && !isSegmenting && (
          <div className="p-3 text-center">
            <p className="text-xs text-gray-600">
              No objects detected yet.
            </p>
            <p className="text-[10px] text-gray-700 mt-1">
              Click "Detect Objects" to run SAM2
            </p>
          </div>
        )}

        {segments.map((seg) => {
          const isSelected = selectedSegmentIds.includes(seg.id);
          const isHovered = hoveredSegmentId === seg.id;
          return (
            <div
              key={seg.id}
              className={`border-b border-gray-800/50 ${
                isSelected ? "bg-violet-900/20" : isHovered ? "bg-violet-900/10" : ""
              }`}
              onMouseEnter={() => setHoveredSegment(seg.id)}
              onMouseLeave={() => setHoveredSegment(null)}
            >
              {/* Segment header */}
              <div className="flex items-center">
                <button
                  onClick={(e) => handleSegmentClick(seg.id, e)}
                  className="flex-1 px-3 py-2 flex items-center gap-2 hover:bg-gray-800/50 text-left min-w-0"
                >
                  <div
                    className="w-3 h-3 rounded-sm shrink-0"
                    style={{
                      backgroundColor: `rgb(${seg.color[0]}, ${seg.color[1]}, ${seg.color[2]})`,
                      opacity: seg.visible ? 1 : 0.3,
                    }}
                  />
                  {editingLabelId === seg.id ? (
                    <input
                      autoFocus
                      value={editLabel}
                      onChange={(e) => setEditLabel(e.target.value)}
                      onBlur={() => handleRename(seg.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleRename(seg.id);
                        if (e.key === "Escape") setEditingLabelId(null);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="text-xs text-gray-300 bg-gray-900 border border-gray-700 rounded px-1 py-0.5 flex-1 min-w-0 focus:outline-none focus:border-violet-500"
                    />
                  ) : (
                    <span
                      className={`text-xs flex-1 truncate ${
                        seg.visible ? "text-gray-300" : "text-gray-600 line-through"
                      }`}
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        setEditingLabelId(seg.id);
                        setEditLabel(seg.label);
                      }}
                    >
                      {seg.label}
                    </span>
                  )}
                  {seg.n_gaussians > 0 && (
                    <span className="text-[10px] text-gray-600 shrink-0">
                      {seg.n_gaussians.toLocaleString()}gs
                    </span>
                  )}
                  <ChevronRight
                    className={`w-3 h-3 text-gray-600 transition-transform shrink-0 ${
                      expandedId === seg.id ? "rotate-90" : ""
                    }`}
                  />
                </button>

                {/* Quick actions */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleVisibility(seg.id, !seg.visible);
                  }}
                  className="p-1.5 hover:bg-gray-800 rounded transition-colors mr-1"
                  title={seg.visible ? "Hide" : "Show"}
                >
                  {seg.visible ? (
                    <Eye className="w-3 h-3 text-gray-500" />
                  ) : (
                    <EyeOff className="w-3 h-3 text-gray-700" />
                  )}
                </button>
              </div>

              {/* Expanded actions */}
              {expandedId === seg.id && (
                <div className="px-3 pb-2 space-y-1.5">
                  <div className="flex gap-1 text-[10px] text-gray-500">
                    <span>Area: {seg.area.toLocaleString()}px</span>
                    <span>IoU: {(seg.confidence * 100).toFixed(0)}%</span>
                  </div>

                  <div className="flex gap-1">
                    <button
                      onClick={() => handleDuplicate(seg.id)}
                      disabled={!seg.n_gaussians}
                      className="flex-1 flex items-center justify-center gap-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-30 text-[10px] py-1.5 rounded"
                    >
                      <Copy className="w-3 h-3" />
                      Duplicate
                    </button>
                    <button
                      onClick={() => handleDelete(seg.id)}
                      className="flex-1 flex items-center justify-center gap-1 bg-red-900/30 hover:bg-red-900/50 text-red-400 text-[10px] py-1.5 rounded"
                    >
                      <Trash2 className="w-3 h-3" />
                      Delete
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer stats */}
      {totalGaussians > 0 && (
        <div className="px-3 py-2 border-t border-gray-800 text-[10px] text-gray-600">
          {totalGaussians.toLocaleString()} total gaussians
          {selectedSegmentIds.length > 1 && (
            <span className="ml-2 text-violet-400">
              {selectedSegmentIds.length} selected
            </span>
          )}
        </div>
      )}
    </div>
  );
}
