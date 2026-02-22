import { useState, useCallback, useEffect } from "react";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import { useEditorStore } from "../../stores/editorStore.ts";
import { SplatBrowser } from "./SplatBrowser.tsx";
import { SegmentPanel } from "./SegmentPanel.tsx";
import { CompositionPanel } from "./CompositionPanel.tsx";
import { RefinementProgress } from "./RefinementProgress.tsx";
import { Tabs, type TabDef } from "../ui/Tabs.tsx";
import {
  FolderOpen,
  Hammer,
  AlertTriangle,
  Sparkles,
  Loader2,
  Download,
  MousePointer2,
  Move,
  RotateCcw,
  Maximize2,
  Settings2,
  Layers,
  Package,
  GitCompareArrows,
  Combine,
} from "lucide-react";
import type { ToolMode } from "../../types/scene.ts";

interface EditPanelProps {
  projectId: string;
  onSwitchToBuild: () => void;
}

const REFINE_PRESETS = [
  { id: "conservative", label: "Conservative", description: "Appearance only, positions frozen. Safest.", iters: 1000 },
  { id: "balanced", label: "Balanced", description: "Moderate refinement with light densification.", iters: 2000 },
  { id: "aggressive", label: "Aggressive", description: "Full retrain with higher LRs.", iters: 5000 },
] as const;
const MESH_FORMATS = ["glb", "obj", "ply"] as const;
const TOOL_BUTTONS: { mode: ToolMode; icon: typeof MousePointer2; label: string; shortcut: string }[] = [
  { mode: "select", icon: MousePointer2, label: "Select", shortcut: "Q" },
  { mode: "translate", icon: Move, label: "Translate", shortcut: "G" },
  { mode: "rotate", icon: RotateCcw, label: "Rotate", shortcut: "R" },
  { mode: "scale", icon: Maximize2, label: "Scale", shortcut: "S" },
];

const TAB_DEFS: TabDef[] = [
  { id: "scene", label: "Scene", icon: Settings2 },
  { id: "segments", label: "Segments", icon: Layers },
  { id: "export", label: "Export", icon: Package },
  { id: "compose", label: "Compose", icon: Combine },
];

export function EditPanel({ projectId, onSwitchToBuild }: EditPanelProps) {
  const plyUrl = useAnySplatStore((s) => s.plyUrl);
  const isRefining = useAnySplatStore((s) => s.isRefining);
  const isPruning = useAnySplatStore((s) => s.isPruning);
  const refineSplat = useAnySplatStore((s) => s.refineSplat);
  const qualityStats = useAnySplatStore((s) => s.qualityStats);
  const comparisonPairs = useAnySplatStore((s) => s.comparisonPairs);
  const fetchComparisonInfo = useAnySplatStore((s) => s.fetchComparisonInfo);
  const toolMode = useEditorStore((s) => s.toolMode);
  const setToolMode = useEditorStore((s) => s.setToolMode);

  const refinePreset = useAnySplatStore((s) => s.refinePreset);
  const setRefinePreset = useAnySplatStore((s) => s.setRefinePreset);

  const [showSplatBrowser, setShowSplatBrowser] = useState(false);
  const [meshFormat, setMeshFormat] = useState<(typeof MESH_FORMATS)[number]>("glb");
  const [isExtracting, setIsExtracting] = useState(false);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [meshProgress, setMeshProgress] = useState(0);
  const [activeTab, setActiveTab] = useState("scene");

  useEffect(() => { fetchComparisonInfo(projectId); }, [projectId, fetchComparisonInfo]);

  const isForeignSplat = plyUrl && !plyUrl.includes(`/data/${projectId}/`);
  const anyProcessing = isRefining || isPruning || isExtracting;
  const hasCameras = !!plyUrl;

  const selectedPreset = REFINE_PRESETS.find((p) => p.id === refinePreset) ?? REFINE_PRESETS[1];

  const handleRefine = useCallback(() => {
    refineSplat(projectId, { preset: refinePreset });
  }, [projectId, refinePreset, refineSplat]);

  const handleExtractMesh = useCallback(async () => {
    setIsExtracting(true);
    setMeshUrl(null);
    setMeshProgress(0);

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream/${projectId}`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "mesh_progress") {
            setMeshProgress(data.progress);
          }
        } catch { /* ignore */ }
      };
    } catch {
      ws = null;
    }

    try {
      const res = await fetch(
        `/api/projects/${projectId}/extract-mesh?format=${meshFormat}`,
        { method: "POST" }
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Mesh extraction failed" }));
        throw new Error(body.detail || "Failed");
      }
      const data = await res.json();
      setMeshUrl(data.mesh_url);
    } catch (e) {
      console.error("Mesh extraction failed:", e);
    } finally {
      setIsExtracting(false);
      setMeshProgress(0);
      if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
    }
  }, [projectId, meshFormat]);

  return (
    <div className="flex flex-col h-full">
      {/* Header: Load Splat + Build link */}
      <div className="px-3 py-2 border-b border-gray-800 flex items-center gap-2">
        <div className="relative flex-1">
          <button
            onClick={() => setShowSplatBrowser(!showSplatBrowser)}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors px-2 py-1.5 rounded hover:bg-gray-800"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            Load Splat
          </button>
          {showSplatBrowser && (
            <SplatBrowser
              currentProjectId={projectId}
              onClose={() => setShowSplatBrowser(false)}
            />
          )}
        </div>
        <button
          onClick={onSwitchToBuild}
          className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-emerald-400 transition-colors"
        >
          <Hammer className="w-3 h-3" />
          Build
        </button>
      </div>

      {/* Foreign splat banner */}
      {isForeignSplat && (
        <div className="px-3 py-2 border-b border-gray-800 bg-amber-900/10">
          <div className="flex items-start gap-2 text-[11px] text-amber-400/80">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>
              Viewing a splat from another project. Segment controls are
              unavailable.
            </span>
          </div>
        </div>
      )}

      {/* Refinement progress (shown when active) */}
      <RefinementProgress projectId={projectId} />

      {/* Tool mode toolbar */}
      <div className="px-3 py-1.5 border-b border-gray-800 flex items-center gap-1">
        {TOOL_BUTTONS.map(({ mode, icon: Icon, label, shortcut }) => (
          <button
            key={mode}
            onClick={() => setToolMode(mode)}
            title={`${label} (${shortcut})`}
            className={`p-1.5 rounded transition-colors ${
              toolMode === mode
                ? "bg-violet-600/30 text-violet-300"
                : "text-gray-500 hover:text-white hover:bg-gray-800"
            }`}
          >
            <Icon className="w-3.5 h-3.5" />
          </button>
        ))}
      </div>

      {/* Tabs */}
      <Tabs tabs={TAB_DEFS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Tab content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {activeTab === "scene" && (
          <div className="flex-1 overflow-y-auto">
            {/* Refine section */}
            {!isForeignSplat && (
              <div className="px-3 py-3 border-b border-gray-800 space-y-2">
                <label className="text-[10px] uppercase tracking-wider text-gray-500 block">
                  Refine
                </label>
                <div className="flex gap-1">
                  {REFINE_PRESETS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setRefinePreset(p.id)}
                      className={`flex-1 px-2 py-1.5 rounded text-[10px] transition-colors ${
                        refinePreset === p.id
                          ? "bg-blue-600/20 text-blue-400 border border-blue-600/40"
                          : "bg-gray-900 text-gray-500 border border-gray-800 hover:border-gray-700"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-gray-500">
                  {selectedPreset.description}
                </p>
                <button
                  onClick={handleRefine}
                  disabled={anyProcessing || !hasCameras}
                  className="w-full bg-blue-900/40 hover:bg-blue-900/60 disabled:opacity-50 px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-1.5 text-blue-300"
                >
                  {isRefining ? (
                    <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Refining...</>
                  ) : (
                    <><Sparkles className="w-3.5 h-3.5" /> Refine ({selectedPreset.iters} iters)</>
                  )}
                </button>
                {!hasCameras && (
                  <p className="text-[10px] text-gray-600 text-center">
                    No cameras.json — reconstruct with AnySplat first
                  </p>
                )}
              </div>
            )}

            {/* Before/After Compare */}
            {comparisonPairs.length > 0 && (
              <div className="px-3 py-2 border-b border-gray-800 space-y-1">
                <label className="text-[10px] uppercase tracking-wider text-gray-500 block">
                  Compare
                </label>
                {comparisonPairs.map((pair) => (
                  <button
                    key={pair.label}
                    onClick={() => {
                      useAnySplatStore.setState({
                        showComparison: true,
                        comparisonBeforeUrl: pair.before_url,
                      });
                    }}
                    className="w-full bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
                  >
                    <GitCompareArrows className="w-3 h-3" />
                    {pair.label}
                  </button>
                ))}
              </div>
            )}

            {/* Quality Stats */}
            {qualityStats && qualityStats.n_gaussians > 0 && (
              <div className="px-3 py-3">
                <label className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 block">
                  Quality
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
                    <span className="text-gray-500">Density</span>
                    <span className="text-gray-300">{qualityStats.density.toFixed(0)} gs/m³</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "segments" && (
          isForeignSplat ? (
            <div className="flex-1 flex items-center justify-center p-4">
              <p className="text-xs text-gray-600 text-center">
                Load this project's own splat to use editing tools.
              </p>
            </div>
          ) : (
            <div className="flex-1 overflow-hidden">
              <SegmentPanel projectId={projectId} />
            </div>
          )
        )}

        {activeTab === "export" && (
          <div className="flex-1 overflow-y-auto">
            {!isForeignSplat && plyUrl && (
              <div className="px-3 py-3 space-y-2">
                <label className="text-[10px] uppercase tracking-wider text-gray-500 block">
                  Export Mesh
                </label>
                <div className="flex items-center gap-2">
                  <select
                    value={meshFormat}
                    onChange={(e) => setMeshFormat(e.target.value as typeof meshFormat)}
                    className="flex-1 bg-gray-900 border border-gray-800 rounded px-2 py-1 text-xs text-gray-300 focus:outline-none"
                  >
                    {MESH_FORMATS.map((f) => (
                      <option key={f} value={f}>{f.toUpperCase()}</option>
                    ))}
                  </select>
                  <button
                    onClick={handleExtractMesh}
                    disabled={anyProcessing}
                    className="flex-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
                  >
                    {isExtracting ? (
                      <><Loader2 className="w-3 h-3 animate-spin" /> Extracting...</>
                    ) : (
                      <><Download className="w-3 h-3" /> Export Mesh</>
                    )}
                  </button>
                </div>
                {isExtracting && meshProgress > 0 && (
                  <div className="w-full bg-gray-800 rounded-full h-1">
                    <div
                      className="bg-emerald-500 h-1 rounded-full transition-all"
                      style={{ width: `${Math.min(100, meshProgress * 100)}%` }}
                    />
                  </div>
                )}
                {meshUrl && (
                  <a
                    href={meshUrl}
                    download
                    className="block text-center text-xs text-emerald-400 hover:text-emerald-300 underline"
                  >
                    Download {meshFormat.toUpperCase()}
                  </a>
                )}
              </div>
            )}
            {(!plyUrl || isForeignSplat) && (
              <div className="flex-1 flex items-center justify-center p-4">
                <p className="text-xs text-gray-600 text-center">
                  Build a scene first to export.
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "compose" && (
          <CompositionPanel projectId={projectId} />
        )}
      </div>
    </div>
  );
}
