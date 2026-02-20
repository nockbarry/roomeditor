import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import {
  Play,
  Loader2,
  Clock,
  Box,
  Eye,
  Maximize,
  RefreshCw,
  ChevronDown,
  Scissors,
  Sparkles,
  BarChart3,
  Zap,
  SlidersHorizontal,
} from "lucide-react";
import { useState } from "react";

interface SettingsPanelProps {
  projectId: string;
}

const QUALITY_LEVELS = [
  { level: 1, label: "Quick",    views: 16,  chunked: false },
  { level: 2, label: "Standard", views: 32,  chunked: false },
  { level: 3, label: "High",     views: 64,  chunked: false },
  { level: 4, label: "Maximum",  views: 96,  chunked: false },
  { level: 5, label: "Chunked",  views: 999, chunked: true  },
];

const FPS_OPTIONS = [
  { value: 1, label: "1 fps", desc: "Slow pans" },
  { value: 2, label: "2 fps", desc: "Default" },
  { value: 5, label: "5 fps", desc: "Fast motion" },
  { value: 10, label: "10 fps", desc: "Maximum" },
];

const RESOLUTION_OPTIONS = [
  { value: 0, label: "Auto" },
  { value: 448, label: "448px" },
  { value: 336, label: "336px" },
  { value: 224, label: "224px" },
];

function autoResLabel(views: number): string {
  if (views <= 32) return "448px";
  if (views <= 64) return "336px";
  return "224px";
}

function estimateVram(views: number, resolution: number, chunked: boolean): string {
  const res = resolution > 0 ? resolution : (views <= 32 ? 448 : views <= 64 ? 336 : 224);
  const effectiveViews = chunked ? 32 : Math.min(views, 128);
  const gb = 4 + effectiveViews * (res / 448) ** 2 * 0.27;
  return `~${Math.min(gb, 16).toFixed(1)}GB`;
}

function estimateTime(views: number, chunked: boolean, chunkSize: number, selectedCount: number): string {
  if (chunked && selectedCount > 0) {
    const stride = Math.max(1, chunkSize - 8);
    const nChunks = Math.max(1, Math.ceil((selectedCount - 8) / stride));
    const secPerChunk = 5;
    return `~${nChunks * secPerChunk}s (${nChunks} chunks)`;
  }
  const effectiveViews = Math.min(views, 128);
  if (effectiveViews <= 16) return "~3-5s";
  if (effectiveViews <= 32) return "~5-8s";
  if (effectiveViews <= 64) return "~8-15s";
  return "~15-30s";
}

function getQualityLevel(maxViews: number, chunked: boolean): number {
  if (chunked) return 5;
  const match = QUALITY_LEVELS.find((q) => !q.chunked && q.views === maxViews);
  return match?.level ?? 0; // 0 = custom
}

export function SettingsPanel({ projectId }: SettingsPanelProps) {
  const maxViews = useAnySplatStore((s) => s.maxViews);
  const resolution = useAnySplatStore((s) => s.resolution);
  const chunked = useAnySplatStore((s) => s.chunked);
  const chunkSize = useAnySplatStore((s) => s.chunkSize);
  const chunkOverlap = useAnySplatStore((s) => s.chunkOverlap);
  const fps = useAnySplatStore((s) => s.fps);
  const setMaxViews = useAnySplatStore((s) => s.setMaxViews);
  const setResolution = useAnySplatStore((s) => s.setResolution);
  const setChunked = useAnySplatStore((s) => s.setChunked);
  const setChunkSize = useAnySplatStore((s) => s.setChunkSize);
  const setChunkOverlap = useAnySplatStore((s) => s.setChunkOverlap);
  const setFps = useAnySplatStore((s) => s.setFps);
  const isRunning = useAnySplatStore((s) => s.isRunning);
  const isExtracting = useAnySplatStore((s) => s.isExtracting);
  const isPruning = useAnySplatStore((s) => s.isPruning);
  const isRefining = useAnySplatStore((s) => s.isRefining);
  const runAnySplat = useAnySplatStore((s) => s.runAnySplat);
  const pruneSplat = useAnySplatStore((s) => s.pruneSplat);
  const refineSplat = useAnySplatStore((s) => s.refineSplat);
  const extractFrames = useAnySplatStore((s) => s.extractFrames);
  const lastRun = useAnySplatStore((s) => s.lastRun);
  const qualityStats = useAnySplatStore((s) => s.qualityStats);
  const pruneResult = useAnySplatStore((s) => s.pruneResult);
  const refineResult = useAnySplatStore((s) => s.refineResult);
  const refineStep = useAnySplatStore((s) => s.refineStep);
  const refineTotal = useAnySplatStore((s) => s.refineTotal);
  const refineLoss = useAnySplatStore((s) => s.refineLoss);
  const selectedCount = useAnySplatStore((s) => s.selectedCount);
  const totalFrames = useAnySplatStore((s) => s.frames.length);
  const plyUrl = useAnySplatStore((s) => s.plyUrl);

  const [showCustomize, setShowCustomize] = useState(false);

  const qualityLevel = getQualityLevel(maxViews, chunked);
  const activeLevel = QUALITY_LEVELS.find((q) => q.level === qualityLevel);
  const anyProcessing = isRunning || isPruning || isRefining || isExtracting;
  const hasSplat = !!plyUrl;

  const handleQualityChange = (level: number) => {
    const q = QUALITY_LEVELS.find((q) => q.level === level);
    if (!q) return;
    setMaxViews(q.views);
    setChunked(q.chunked);
    setResolution(0); // Auto
    if (q.chunked) {
      setChunkSize(32);
      setChunkOverlap(8);
    }
  };

  // Compute chunk count for visualization
  const nChunks = chunked && selectedCount > 0
    ? Math.max(1, Math.ceil((selectedCount - chunkOverlap) / Math.max(1, chunkSize - chunkOverlap)))
    : 0;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Frame Extraction */}
      <div className="px-3 py-3 border-b border-gray-800">
        <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 block">
          Frame Extraction
        </label>
        <div className="flex items-center gap-2 mb-2">
          <select
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
            className="flex-1 bg-gray-900 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 focus:outline-none focus:border-gray-600"
          >
            {FPS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label} — {opt.desc}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={() => extractFrames(projectId, fps)}
          disabled={anyProcessing}
          className="w-full bg-gray-800 hover:bg-gray-700 disabled:opacity-50 px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
        >
          {isExtracting ? (
            <><Loader2 className="w-3 h-3 animate-spin" /> Extracting...</>
          ) : (
            <><RefreshCw className="w-3 h-3" /> Re-extract Frames</>
          )}
        </button>
        {totalFrames > 0 && (
          <p className="text-[10px] text-gray-600 mt-1 text-center">
            {selectedCount} of {totalFrames} frames selected
          </p>
        )}
      </div>

      {/* Quality slider */}
      <div className="px-3 py-3 border-b border-gray-800">
        <div className="flex justify-between items-center mb-2">
          <label className="text-[10px] uppercase tracking-wider text-gray-500">
            Quality
          </label>
          <span className="text-xs text-emerald-400 font-medium">
            {activeLevel?.label ?? "Custom"}
          </span>
        </div>
        <input
          type="range" min={1} max={5} step={1}
          value={qualityLevel || 2}
          onChange={(e) => handleQualityChange(Number(e.target.value))}
          className="w-full accent-emerald-500"
        />
        <div className="flex justify-between text-[9px] text-gray-600 mt-1">
          <span>Quick</span>
          <span>Standard</span>
          <span>High</span>
          <span>Max</span>
          <span>Chunked</span>
        </div>
      </div>

      {/* VRAM / Time estimates */}
      <div className="px-3 py-2 border-b border-gray-800">
        <div className="flex justify-between text-[10px] text-gray-500">
          <span>
            <Zap className="w-3 h-3 inline mr-0.5" />
            VRAM: {estimateVram(maxViews, resolution, chunked)} / 16GB
          </span>
          <span>
            <Clock className="w-3 h-3 inline mr-0.5" />
            {estimateTime(maxViews, chunked, chunkSize, selectedCount)}
          </span>
        </div>
        {!chunked && selectedCount > maxViews && (
          <div className="text-[10px] text-amber-400/70 mt-1">
            {selectedCount} selected but only {maxViews} will be used
          </div>
        )}
      </div>

      {/* Chunk visualization — shown when chunked */}
      {chunked && selectedCount > 0 && (
        <div className="px-3 py-2 border-b border-gray-800">
          <div className="text-[10px] text-gray-500 mb-1.5">
            {nChunks} chunk{nChunks !== 1 ? "s" : ""}, {chunkOverlap}v overlap
          </div>
          <div className="flex gap-0.5 h-2">
            {Array.from({ length: Math.min(nChunks, 30) }).map((_, i) => (
              <div key={i} className="flex-1 bg-emerald-600/40 rounded-sm" />
            ))}
            {nChunks > 30 && (
              <span className="text-[9px] text-gray-500 ml-1">+{nChunks - 30}</span>
            )}
          </div>
        </div>
      )}

      {/* Customize toggle */}
      <div className="px-3 py-2 border-b border-gray-800">
        <button
          onClick={() => setShowCustomize(!showCustomize)}
          className="w-full flex items-center justify-between text-[10px] uppercase tracking-wider text-gray-500 hover:text-gray-400"
        >
          <span className="flex items-center gap-1">
            <SlidersHorizontal className="w-3 h-3" />
            Customize
          </span>
          <ChevronDown className={`w-3 h-3 transition-transform ${showCustomize ? "rotate-180" : ""}`} />
        </button>

        {showCustomize && (
          <div className="mt-2 space-y-3">
            <div>
              <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                <span>Max Views</span>
                <span className="text-emerald-400 font-medium">
                  {chunked ? `${selectedCount} (all)` : maxViews}
                </span>
              </div>
              <input
                type="range" min={8} max={128} step={8}
                value={chunked ? 128 : maxViews}
                onChange={(e) => { setMaxViews(Number(e.target.value)); setChunked(false); }}
                disabled={chunked}
                className="w-full accent-emerald-500 disabled:opacity-40"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                <span>Resolution</span>
                <span className="text-gray-400">
                  {resolution === 0 ? `Auto (${autoResLabel(maxViews)})` : `${resolution}px`}
                </span>
              </div>
              <div className="flex gap-1">
                {RESOLUTION_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setResolution(opt.value)}
                    className={`flex-1 py-1 rounded text-[10px] transition-colors ${
                      resolution === opt.value
                        ? "bg-emerald-600/20 text-emerald-400 border border-emerald-600/40"
                        : "bg-gray-900 text-gray-500 border border-gray-800 hover:border-gray-700"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={chunked} onChange={(e) => setChunked(e.target.checked)} className="accent-emerald-500" />
                <span className="text-xs text-gray-400">Chunked merge</span>
              </label>
              <p className="text-[10px] text-gray-600 mt-0.5 ml-5">
                Process in overlapping chunks. More overlap = fewer seams, slower.
              </p>
            </div>

            {chunked && (
              <div className="ml-5 space-y-2">
                <div>
                  <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                    <span>Chunk Size</span>
                    <span className="text-gray-400">{chunkSize}v</span>
                  </div>
                  <input type="range" min={16} max={64} step={8} value={chunkSize}
                    onChange={(e) => setChunkSize(Number(e.target.value))}
                    className="w-full accent-emerald-500" />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                    <span>Overlap</span>
                    <span className="text-gray-400">{chunkOverlap}v</span>
                  </div>
                  <input type="range" min={2} max={16} step={2} value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(Number(e.target.value))}
                    className="w-full accent-emerald-500" />
                  <p className="text-[9px] text-gray-600 mt-0.5">
                    Higher overlap reduces seams between chunks
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Rebuild button */}
      <div className="px-3 py-3 border-b border-gray-800">
        <button
          onClick={() => runAnySplat(projectId)}
          disabled={anyProcessing || selectedCount === 0}
          className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          {isRunning ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> {chunked ? "Chunking..." : "Building..."}</>
          ) : (
            <><Play className="w-4 h-4" /> Rebuild</>
          )}
        </button>
        {selectedCount === 0 && (
          <p className="text-[10px] text-red-400 mt-1 text-center">No frames selected</p>
        )}
      </div>

      {/* Post-rebuild actions */}
      {hasSplat && (
        <div className="px-3 py-3 border-b border-gray-800 space-y-1.5">
          <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-1 block">
            Post-Process
          </label>
          <button
            onClick={() => pruneSplat(projectId)}
            disabled={anyProcessing}
            className="w-full bg-gray-800 hover:bg-gray-700 disabled:opacity-50 px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
          >
            {isPruning ? (
              <><Loader2 className="w-3 h-3 animate-spin" /> Pruning...</>
            ) : (
              <><Scissors className="w-3 h-3" /> Prune Floaters</>
            )}
          </button>
          <button
            onClick={() => refineSplat(projectId, 2000)}
            disabled={anyProcessing}
            className="w-full bg-blue-900/40 hover:bg-blue-900/60 disabled:opacity-50 px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1.5 text-blue-300"
          >
            {isRefining ? (
              <><Loader2 className="w-3 h-3 animate-spin" /> Refining{refineTotal > 0 ? ` ${refineStep}/${refineTotal}` : "..."}</>
            ) : (
              <><Sparkles className="w-3 h-3" /> Refine (2k iters)</>
            )}
          </button>
          {isRefining && refineTotal > 0 && (
            <div className="space-y-1">
              <div className="w-full bg-gray-800 rounded-full h-1">
                <div
                  className="bg-blue-500 h-1 rounded-full transition-all"
                  style={{ width: `${Math.min(100, (refineStep / refineTotal) * 100)}%` }}
                />
              </div>
              {refineLoss !== null && (
                <p className="text-[9px] text-gray-600 text-center">loss: {refineLoss.toFixed(5)}</p>
              )}
            </div>
          )}
          {pruneResult && (
            <p className="text-[10px] text-gray-500 text-center">
              Pruned {pruneResult.n_pruned.toLocaleString()} ({pruneResult.n_before.toLocaleString()} → {pruneResult.n_after.toLocaleString()})
            </p>
          )}
          {refineResult && (
            <p className="text-[10px] text-blue-400/70 text-center">
              Refined to {refineResult.n_gaussians.toLocaleString()} gaussians ({refineResult.iterations_run} iters)
            </p>
          )}
        </div>
      )}

      {/* Run stats */}
      {lastRun && (
        <div className="px-3 py-3 border-b border-gray-800">
          <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 block">
            Last Run
          </label>
          <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-400">
            <div className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-gray-600" /> {lastRun.duration_sec}s
            </div>
            <div className="flex items-center gap-1.5">
              <Box className="w-3 h-3 text-gray-600" /> {lastRun.n_gaussians.toLocaleString()}
            </div>
            <div className="flex items-center gap-1.5">
              <Eye className="w-3 h-3 text-gray-600" /> {lastRun.views_used}v
            </div>
            <div className="flex items-center gap-1.5">
              <Maximize className="w-3 h-3 text-gray-600" /> {lastRun.resolution_used}px
            </div>
          </div>
          {lastRun.cleanup_stats && (lastRun.cleanup_stats.n_opacity_removed + lastRun.cleanup_stats.n_scale_removed + lastRun.cleanup_stats.n_floater_removed > 0) && (
            <div className="mt-2 text-[10px] text-emerald-400/80 bg-emerald-900/20 rounded px-2 py-1.5">
              Cleaned: removed{" "}
              {[
                lastRun.cleanup_stats.n_floater_removed > 0 && `${lastRun.cleanup_stats.n_floater_removed.toLocaleString()} floaters`,
                lastRun.cleanup_stats.n_scale_removed > 0 && `${lastRun.cleanup_stats.n_scale_removed.toLocaleString()} outliers`,
                lastRun.cleanup_stats.n_opacity_removed > 0 && `${lastRun.cleanup_stats.n_opacity_removed.toLocaleString()} transparent`,
              ].filter(Boolean).join(", ")}
              {" "}({lastRun.cleanup_stats.n_before.toLocaleString()} → {lastRun.cleanup_stats.n_after.toLocaleString()})
            </div>
          )}
        </div>
      )}

      {/* Quality Stats */}
      {qualityStats && qualityStats.n_gaussians > 0 && (
        <div className="px-3 py-3">
          <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 flex items-center gap-1">
            <BarChart3 className="w-3 h-3" /> Quality
          </label>
          <div className="space-y-1 text-[10px]">
            <div className="flex justify-between">
              <span className="text-gray-500">Total</span>
              <span className="text-gray-300">{qualityStats.n_gaussians.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Effective (opa{">"}0.1)</span>
              <span className="text-gray-300">{qualityStats.n_effective.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Transparent</span>
              <span className={qualityStats.frac_transparent > 0.3 ? "text-amber-400" : "text-gray-300"}>
                {(qualityStats.frac_transparent * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Opaque</span>
              <span className="text-gray-300">{(qualityStats.frac_opaque * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Scale outliers</span>
              <span className={qualityStats.frac_scale_outlier > 0.05 ? "text-amber-400" : "text-gray-300"}>
                {(qualityStats.frac_scale_outlier * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Density</span>
              <span className="text-gray-300">{qualityStats.density.toFixed(0)} gs/m³</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
