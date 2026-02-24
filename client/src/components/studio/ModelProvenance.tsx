import { useState, useRef, useEffect } from "react";
import { ChevronUp, X, Check, Minus } from "lucide-react";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import type { SceneFormatInfo } from "../../hooks/useSplatScene.ts";
import type { ModelStage } from "../../api/postprocess.ts";

interface ModelProvenanceProps {
  projectId: string;
  sceneFormat: SceneFormatInfo | null;
}

const STAGE_ORDER = ["anysplat", "cleanup", "prune", "refine"] as const;

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function StageRow({
  stage,
  isActive,
  isLast,
  onClick,
}: {
  stage: ModelStage;
  isActive: boolean;
  isLast: boolean;
  onClick: () => void;
}) {
  const gaussianCount = stage.files.ply?.gaussian_count;
  const plyMB = stage.files.ply?.size_mb;
  const spzMB = stage.files.spz?.size_mb;

  return (
    <button
      disabled={!stage.exists}
      onClick={onClick}
      className={`w-full text-left flex gap-2.5 ${stage.exists ? "cursor-pointer group" : "cursor-default"}`}
    >
      {/* Timeline column */}
      <div className="flex flex-col items-center w-5 shrink-0">
        {/* Dot */}
        <div className={`w-3.5 h-3.5 rounded-full flex items-center justify-center shrink-0 ${
          isActive
            ? "bg-emerald-500 ring-2 ring-emerald-400/40 ring-offset-1 ring-offset-gray-900"
            : stage.exists
              ? "bg-gray-600 group-hover:bg-gray-500"
              : "bg-gray-800 border border-gray-700"
        }`}>
          {stage.exists ? (
            <Check className={`w-2 h-2 ${isActive ? "text-white" : "text-gray-400"}`} />
          ) : (
            <Minus className="w-2 h-2 text-gray-700" />
          )}
        </div>
        {/* Connector */}
        {!isLast && (
          <div className={`w-px flex-1 min-h-[8px] ${stage.exists ? "bg-gray-600" : "bg-gray-800"}`} />
        )}
      </div>

      {/* Content */}
      <div className={`pb-2.5 min-w-0 flex-1 ${!isLast ? "" : ""}`}>
        <div className="flex items-center gap-1.5">
          <span className={`text-[11px] font-medium leading-none ${
            isActive ? "text-emerald-400" : stage.exists ? "text-gray-300 group-hover:text-white" : "text-gray-600"
          }`}>
            {stage.label}
          </span>
          {isActive && (
            <span className="text-[8px] uppercase tracking-wider text-emerald-500 font-semibold">viewing</span>
          )}
        </div>
        <div className={`text-[10px] leading-snug mt-0.5 ${stage.exists ? "text-gray-500" : "text-gray-700"}`}>
          {stage.description}
        </div>
        {/* Stats row */}
        {stage.exists && (gaussianCount || plyMB != null) && (
          <div className="flex items-center gap-2 mt-1 text-[9px]">
            {gaussianCount && (
              <span className="text-gray-400">{formatCount(gaussianCount)} gs</span>
            )}
            {spzMB != null && (
              <span className="text-emerald-400/80">SPZ {spzMB}MB</span>
            )}
            {plyMB != null && (
              <span className="text-gray-500">PLY {plyMB}MB</span>
            )}
          </div>
        )}
      </div>
    </button>
  );
}

export function ModelProvenance({ projectId, sceneFormat }: ModelProvenanceProps) {
  const [open, setOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  const modelInfo = useAnySplatStore((s) => s.modelInfo);
  const activeStageId = useAnySplatStore((s) => s.activeStageId);
  const activeFormat = useAnySplatStore((s) => s.activeFormat);
  const switchStage = useAnySplatStore((s) => s.switchStage);
  const setActiveFormat = useAnySplatStore((s) => s.setActiveFormat);

  // Close popover on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (!sceneFormat) return null;

  const isSpz = sceneFormat.format === "spz";

  // Find the active stage details
  const activeStage = modelInfo?.stages.find((s) => s.id === activeStageId);

  // Check if current stage has both formats
  const hasSpz = activeStage?.files.spz != null;
  const hasPly = activeStage?.files.ply != null;
  const hasBothFormats = hasSpz && hasPly;

  return (
    <div className="relative" ref={popoverRef}>
      {/* Badge */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded transition-colors ${
          isSpz
            ? "bg-emerald-900/80 text-emerald-300 border border-emerald-700/50 hover:bg-emerald-900"
            : "bg-gray-800/80 text-gray-400 border border-gray-700/50 hover:bg-gray-800"
        }`}
      >
        {sceneFormat.format.toUpperCase()} {sceneFormat.sizeMB}MB
        <ChevronUp className={`w-2.5 h-2.5 transition-transform ${open ? "" : "rotate-180"}`} />
      </button>

      {/* Popover */}
      {open && modelInfo && (
        <div className="absolute bottom-full left-0 mb-1.5 w-[280px] bg-gray-900 border border-gray-700/50 rounded-lg shadow-xl z-50">
          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
            <span className="text-xs font-medium text-gray-300">Model Pipeline</span>
            <button
              onClick={() => setOpen(false)}
              className="text-gray-500 hover:text-gray-300"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Pipeline stages — vertical timeline */}
          <div className="px-3 pt-3 pb-1">
            {STAGE_ORDER.map((stageId, i) => {
              const stage = modelInfo.stages.find((s) => s.id === stageId);
              if (!stage) return null;
              return (
                <StageRow
                  key={stageId}
                  stage={stage}
                  isActive={stageId === activeStageId}
                  isLast={i === STAGE_ORDER.length - 1}
                  onClick={() => {
                    if (stage.exists) switchStage(projectId, stageId);
                  }}
                />
              );
            })}
          </div>

          {/* Format toggle — only when active stage has both formats */}
          {hasBothFormats && (
            <div className="px-3 pb-3 pt-1 border-t border-gray-800">
              <div className="text-[10px] text-gray-500 mb-1.5">Load format</div>
              <div className="flex rounded-md overflow-hidden border border-gray-700">
                <button
                  onClick={() => setActiveFormat(projectId, "spz")}
                  className={`flex-1 px-2 py-1 text-[10px] font-medium transition-colors ${
                    activeFormat === "spz"
                      ? "bg-emerald-600 text-white"
                      : "bg-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  SPZ {activeStage!.files.spz!.size_mb}MB
                </button>
                <button
                  onClick={() => setActiveFormat(projectId, "ply")}
                  className={`flex-1 px-2 py-1 text-[10px] font-medium transition-colors ${
                    activeFormat === "ply"
                      ? "bg-gray-600 text-white"
                      : "bg-gray-800 text-gray-400 hover:text-white"
                  }`}
                >
                  PLY {activeStage!.files.ply!.size_mb}MB
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
