import { useState } from "react";
import { Play, ChevronDown, ChevronRight, Zap } from "lucide-react";
import type { TrainingConfig } from "../../types/api.ts";
import { getPresets, PARAM_DEFS, PARAM_DEFS_2DGS } from "../../constants/presets.ts";

type Mode = "anysplat" | "spfsplat" | "2dgs" | "3dgs";

interface Props {
  onStart: (config: TrainingConfig) => void;
  onAnySplat?: () => void;
}

export function ReconstructionSettings({ onStart, onAnySplat }: Props) {
  const [mode, setMode] = useState<Mode>("anysplat");
  const presets = getPresets(mode);
  const [selectedPreset, setSelectedPreset] = useState(1); // Standard/Balanced
  const [config, setConfig] = useState<TrainingConfig>({ ...presets[1].config });
  const [showAdvanced, setShowAdvanced] = useState(false);

  const selectPreset = (index: number) => {
    setSelectedPreset(index);
    setConfig({ ...presets[index].config });
  };

  const switchMode = (newMode: Mode) => {
    setMode(newMode);
    const newPresets = getPresets(newMode);
    setSelectedPreset(1);
    setConfig({ ...newPresets[1].config });
  };

  const updateParam = (key: keyof TrainingConfig, value: number | boolean | string) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
    if (typeof value === "number") {
      const newConfig = { ...config, [key]: value };
      const matchIdx = presets.findIndex((p) =>
        PARAM_DEFS.every((d) => p.config[d.key] === newConfig[d.key])
      );
      setSelectedPreset(matchIdx);
    }
  };

  const isAnySplat = mode === "anysplat";
  const isFeedForward = mode === "anysplat" || mode === "spfsplat";
  const allSliders = mode === "2dgs"
    ? [...PARAM_DEFS, ...PARAM_DEFS_2DGS]
    : PARAM_DEFS;

  return (
    <div className="text-center max-w-lg mx-auto">
      <h2 className="text-lg font-medium mb-2">Ready to Reconstruct</h2>
      <p className="text-sm text-gray-400 mb-4">
        Choose a reconstruction method and quality preset.
      </p>

      {/* Mode toggle */}
      <div className="flex items-center justify-center gap-1 bg-gray-900 rounded-lg p-1 mb-6 max-w-md mx-auto">
        <button
          onClick={() => switchMode("anysplat")}
          className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === "anysplat"
              ? "bg-emerald-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          Instant (AnySplat)
        </button>
        <button
          onClick={() => switchMode("spfsplat")}
          className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === "spfsplat"
              ? "bg-emerald-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          SPFSplat
        </button>
        <button
          onClick={() => switchMode("2dgs")}
          className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === "2dgs"
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          Surface (2DGS)
        </button>
        <button
          onClick={() => switchMode("3dgs")}
          className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === "3dgs"
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          Classic (3DGS)
        </button>
      </div>

      {/* AnySplat description */}
      {isAnySplat && (
        <div className="mb-4 text-xs text-gray-400 bg-gray-900 rounded-lg px-4 py-3 border border-gray-800">
          <div className="flex items-center gap-1.5 mb-1 text-emerald-400 font-medium">
            <Zap className="w-3.5 h-3.5" />
            Feed-forward reconstruction — no training needed
          </div>
          Produces 3D Gaussians in seconds from unposed images. Best for quick previews and room scanning.
        </div>
      )}

      {/* SPFSplat description */}
      {mode === "spfsplat" && (
        <div className="mb-4 text-xs text-gray-400 bg-gray-900 rounded-lg px-4 py-3 border border-gray-800">
          <div className="flex items-center gap-1.5 mb-1 text-emerald-400 font-medium">
            <Zap className="w-3.5 h-3.5" />
            Pose-free feed-forward (SPFSplat V2)
          </div>
          256px resolution, DC-only SH, max 8 views. No refinement support (no camera export). Good for comparison.
        </div>
      )}

      {/* Preset cards */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {presets.map((preset, i) => (
          <button
            key={preset.name}
            onClick={() => selectPreset(i)}
            className={`p-3 rounded-lg border text-left transition-colors ${
              selectedPreset === i
                ? isAnySplat ? "border-emerald-500 bg-emerald-500/10" : "border-blue-500 bg-blue-500/10"
                : "border-gray-700 bg-gray-900 hover:border-gray-500"
            }`}
          >
            <div className="text-sm font-medium mb-1">{preset.name}</div>
            <div className="text-xs text-gray-400 mb-2">{preset.description}</div>
            <div className="text-xs text-gray-500">{preset.estimatedTime}</div>
          </button>
        ))}
      </div>

      {/* Training-specific options (hidden for feed-forward modes) */}
      {!isFeedForward && (
        <>
          {/* SfM backend toggle */}
          <div className="mb-4">
            <div className="text-xs text-gray-400 mb-2">Pose Estimation</div>
            <div className="flex items-center justify-center gap-1 bg-gray-900 rounded-lg p-1 max-w-xs mx-auto">
              <button
                onClick={() => updateParam("sfm_backend" as keyof TrainingConfig, "colmap" as any)}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  config.sfm_backend === "colmap"
                    ? "bg-purple-600 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                COLMAP
              </button>
              <button
                onClick={() => updateParam("sfm_backend" as keyof TrainingConfig, "mast3r" as any)}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  config.sfm_backend === "mast3r"
                    ? "bg-purple-600 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                MASt3R
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {config.sfm_backend === "mast3r"
                ? "AI-based poses — faster, better with phone video"
                : "Traditional SfM — proven, needs good overlap"}
            </p>
          </div>

          {/* Feature toggles */}
          <div className="flex items-center justify-center gap-4 mb-4">
            <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={config.appearance_embeddings}
                onChange={(e) => updateParam("appearance_embeddings", e.target.checked)}
                className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              Appearance Compensation
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={config.tidi_pruning}
                onChange={(e) => updateParam("tidi_pruning", e.target.checked)}
                className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              Smart Pruning
            </label>
          </div>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300 mx-auto mb-4"
          >
            {showAdvanced ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            Advanced Settings
            {selectedPreset === -1 && (
              <span className="ml-1 text-yellow-500">(Custom)</span>
            )}
          </button>

          {/* Advanced panel */}
          {showAdvanced && (
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 mb-6 text-left">
              {allSliders.map((param) => (
                <div key={param.key} className="mb-3 last:mb-0">
                  <div className="flex justify-between text-xs mb-1">
                    <label className="text-gray-300">{param.label}</label>
                    <span className="text-gray-500 tabular-nums">
                      {param.step < 0.01
                        ? (config[param.key] as number).toFixed(3)
                        : param.step < 1
                          ? (config[param.key] as number).toFixed(2)
                          : config[param.key] as number}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={config[param.key] as number}
                    onChange={(e) => updateParam(param.key, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Start button */}
      <button
        onClick={() => {
          if (isAnySplat && onAnySplat) {
            onAnySplat();
          } else if (mode === "spfsplat") {
            onStart({ ...config, sfm_backend: "spfsplat" as any });
          } else {
            onStart(config);
          }
        }}
        className={`flex items-center gap-2 px-6 py-3 rounded-lg text-sm font-medium transition-colors mx-auto ${
          isFeedForward
            ? "bg-emerald-600 hover:bg-emerald-500"
            : "bg-blue-600 hover:bg-blue-500"
        }`}
      >
        {isFeedForward ? <Zap className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        {isAnySplat ? "Open AnySplat Studio" : mode === "spfsplat" ? "Run SPFSplat" : "Start Reconstruction"}
      </button>
    </div>
  );
}
