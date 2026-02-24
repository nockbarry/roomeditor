import { create } from "zustand";
import type { FrameInfo, AnySplatRunResult } from "../types/api.ts";
import * as anysplatApi from "../api/anysplat.ts";
import type { QualityStats, PruneResult, RefineResult, ComparisonPair, RefineMetrics, RefineEval, ModelInfo } from "../api/postprocess.ts";
import * as postprocessApi from "../api/postprocess.ts";
import { toast } from "./toastStore.ts";

interface AnySplatStore {
  frames: FrameInfo[];
  selectedCount: number;
  framesLoading: boolean;
  maxViews: number;
  resolution: number;
  chunked: boolean;
  chunkSize: number;
  chunkOverlap: number;
  fps: number;
  isRunning: boolean;
  isExtracting: boolean;
  isPruning: boolean;
  isRefining: boolean;
  refineStep: number;
  refineTotal: number;
  refineLoss: number | null;
  refineLossHistory: number[];
  refineGaussianCount: number | null;
  refineStartTime: number | null;
  refineMetricsHistory: RefineMetrics[];
  refineEvalHistory: RefineEval[];
  refineBaseline: { psnr: number; ssim: number } | null;
  refinePreset: string;
  lastRun: AnySplatRunResult | null;
  qualityStats: QualityStats | null;
  pruneResult: PruneResult | null;
  refineResult: RefineResult | null;
  plyUrl: string | null;
  plyVersion: number;
  comparisonPairs: ComparisonPair[];
  showComparison: boolean;
  comparisonBeforeUrl: string | null;
  modelInfo: ModelInfo | null;
  activeStageId: string | null;
  activeFormat: "ply" | "spz";

  // Actions
  extractFrames: (projectId: string, fps?: number) => Promise<void>;
  fetchFrames: (projectId: string) => Promise<void>;
  toggleFrame: (projectId: string, filename: string, selected: boolean) => Promise<void>;
  toggleAllFrames: (projectId: string, selected: boolean) => Promise<void>;
  setMaxViews: (views: number) => void;
  setResolution: (resolution: number) => void;
  setChunked: (chunked: boolean) => void;
  setChunkSize: (size: number) => void;
  setChunkOverlap: (overlap: number) => void;
  setFps: (fps: number) => void;
  runAnySplat: (projectId: string) => Promise<void>;
  pruneSplat: (projectId: string) => Promise<void>;
  refineSplat: (projectId: string, params?: {
    preset?: string; mode?: string;
  }) => Promise<void>;
  stopRefinement: (projectId: string) => Promise<void>;
  setRefinePreset: (preset: string) => void;
  fetchQualityStats: (projectId: string) => Promise<void>;
  fetchComparisonInfo: (projectId: string) => Promise<void>;
  setShowComparison: (show: boolean) => void;
  setComparisonBeforeUrl: (url: string | null) => void;
  fetchModelInfo: (projectId: string) => Promise<void>;
  switchStage: (projectId: string, stageId: string) => void;
  setActiveFormat: (projectId: string, format: "ply" | "spz") => void;
  reset: () => void;
}

/** Compute which selected frame indices will actually be used by AnySplat */
export function computeUsedFrameIndices(
  frames: FrameInfo[],
  maxViews: number,
  chunked: boolean,
): Set<number> {
  const selectedIndices: number[] = [];
  frames.forEach((f, i) => {
    if (f.selected) selectedIndices.push(i);
  });

  if (chunked || selectedIndices.length <= maxViews) {
    // All selected frames are used
    return new Set(selectedIndices);
  }

  // Subsample: take every Nth frame (mirrors run_inference_max.py logic)
  const step = Math.max(1, Math.floor(selectedIndices.length / maxViews));
  const used = new Set<number>();
  for (let i = 0; i < selectedIndices.length && used.size < maxViews; i += step) {
    used.add(selectedIndices[i]);
  }
  return used;
}

export const useAnySplatStore = create<AnySplatStore>((set, get) => ({
  frames: [],
  selectedCount: 0,
  framesLoading: false,
  maxViews: 32,
  resolution: 0,
  chunked: false,
  chunkSize: 32,
  chunkOverlap: 8,
  fps: 2,
  isRunning: false,
  isExtracting: false,
  isPruning: false,
  isRefining: false,
  refineStep: 0,
  refineTotal: 0,
  refineLoss: null,
  refineLossHistory: [],
  refineGaussianCount: null,
  refineStartTime: null,
  refineMetricsHistory: [],
  refineEvalHistory: [],
  refineBaseline: null,
  refinePreset: "balanced",
  lastRun: null,
  qualityStats: null,
  pruneResult: null,
  refineResult: null,
  plyUrl: null,
  plyVersion: 0,
  comparisonPairs: [],
  showComparison: false,
  comparisonBeforeUrl: null,
  modelInfo: null,
  activeStageId: null,
  activeFormat: "spz",

  extractFrames: async (projectId: string, fps?: number) => {
    const fpsToUse = fps ?? get().fps;
    set({ framesLoading: true, isExtracting: true });
    try {
      const manifest = await anysplatApi.extractFrames(projectId, { fps: fpsToUse });
      set({
        frames: manifest.frames,
        selectedCount: manifest.selected_count,
        framesLoading: false,
        isExtracting: false,
      });
      toast.success(`Extracted ${manifest.frames.length} frames`);
    } catch (e) {
      console.error("Failed to extract frames:", e);
      set({ framesLoading: false, isExtracting: false });
      toast.error("Failed to extract frames");
    }
  },

  fetchFrames: async (projectId: string) => {
    set({ framesLoading: true });
    try {
      const manifest = await anysplatApi.fetchFrames(projectId);
      set({
        frames: manifest.frames,
        selectedCount: manifest.selected_count,
        framesLoading: false,
      });
    } catch (e) {
      console.error("Failed to fetch frames:", e);
      set({ framesLoading: false });
    }
  },

  toggleFrame: async (projectId: string, filename: string, selected: boolean) => {
    set((state) => ({
      frames: state.frames.map((f) =>
        f.filename === filename ? { ...f, selected } : f
      ),
      selectedCount: state.selectedCount + (selected ? 1 : -1),
    }));

    try {
      const manifest = await anysplatApi.updateFrameSelection(projectId, {
        [filename]: selected,
      });
      set({ frames: manifest.frames, selectedCount: manifest.selected_count });
    } catch (e) {
      console.error("Failed to update frame selection:", e);
      get().fetchFrames(projectId);
    }
  },

  toggleAllFrames: async (projectId: string, selected: boolean) => {
    const updates: Record<string, boolean> = {};
    for (const frame of get().frames) {
      updates[frame.filename] = selected;
    }

    set((state) => ({
      frames: state.frames.map((f) => ({ ...f, selected })),
      selectedCount: selected ? state.frames.length : 0,
    }));

    try {
      const manifest = await anysplatApi.updateFrameSelection(projectId, updates);
      set({ frames: manifest.frames, selectedCount: manifest.selected_count });
    } catch (e) {
      console.error("Failed to update all frames:", e);
      get().fetchFrames(projectId);
    }
  },

  setMaxViews: (views: number) => set({ maxViews: views }),
  setResolution: (resolution: number) => set({ resolution }),
  setChunked: (chunked: boolean) => set({ chunked }),
  setChunkSize: (size: number) => set({ chunkSize: size }),
  setChunkOverlap: (overlap: number) => set({ chunkOverlap: overlap }),
  setFps: (fps: number) => set({ fps }),

  runAnySplat: async (projectId: string) => {
    const state = get();
    set({ isRunning: true, qualityStats: null, pruneResult: null, refineResult: null });
    try {
      const result = await anysplatApi.runAnySplat(projectId, {
        maxViews: state.maxViews,
        resolution: state.resolution,
        chunked: state.chunked,
        chunkSize: state.chunkSize,
        chunkOverlap: state.chunkOverlap,
      });
      set((s) => ({
        isRunning: false,
        lastRun: result,
        plyUrl: result.ply_url,
        plyVersion: s.plyVersion + 1,
      }));
      // Auto-fetch quality stats + model info after rebuild
      get().fetchQualityStats(projectId);
      get().fetchModelInfo(projectId);
      toast.success(`Reconstruction complete — ${result.n_gaussians.toLocaleString()} gaussians`);
    } catch (e) {
      console.error("Failed to run AnySplat:", e);
      set({ isRunning: false });
      toast.error(`Reconstruction failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    }
  },

  pruneSplat: async (projectId: string) => {
    set({ isPruning: true });
    try {
      const result = await postprocessApi.pruneSplat(projectId);
      set((s) => ({
        isPruning: false,
        pruneResult: result,
        plyVersion: s.plyVersion + 1,
      }));
      get().fetchQualityStats(projectId);
      get().fetchComparisonInfo(projectId);
      get().fetchModelInfo(projectId);
      toast.success(`Pruned ${result.n_pruned.toLocaleString()} floaters`);
    } catch (e) {
      console.error("Failed to prune:", e);
      set({ isPruning: false });
      toast.error("Failed to prune splat");
    }
  },

  refineSplat: async (projectId: string, params?: {
    preset?: string; mode?: string;
  }) => {
    const preset = params?.preset ?? get().refinePreset;
    set({
      isRefining: true, refineStep: 0, refineTotal: 0,
      refineLoss: null, refineLossHistory: [], refineGaussianCount: null,
      refineStartTime: Date.now(),
      refineMetricsHistory: [], refineEvalHistory: [], refineBaseline: null,
    });

    // Open WebSocket to receive streaming updates during refinement
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream/${projectId}`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "refine_snapshot") {
            set((s) => ({
              refineStep: data.step,
              refineTotal: data.total_steps,
              refineLoss: data.loss,
              refineGaussianCount: data.n_gaussians,
              refineLossHistory: [...s.refineLossHistory, data.loss],
              plyVersion: s.plyVersion + 1,
            }));
          } else if (data.type === "refine_progress") {
            set((s) => ({ refineStep: Math.round(data.progress * (s.refineTotal || 1)) }));
          } else if (data.type === "refine_metrics") {
            set((s) => ({
              refineStep: data.step,
              refineTotal: data.total_steps,
              refineLoss: data.losses.total,
              refineGaussianCount: data.n_gaussians,
              refineMetricsHistory: [...s.refineMetricsHistory, data as import("../api/postprocess.ts").RefineMetrics],
            }));
          } else if (data.type === "refine_eval") {
            const evalData = data as import("../api/postprocess.ts").RefineEval;
            set((s) => {
              const baseline = s.refineBaseline ?? { psnr: evalData.mean_psnr, ssim: evalData.mean_ssim };
              return {
                refineBaseline: s.refineBaseline ?? baseline,
                refineEvalHistory: [...s.refineEvalHistory, evalData],
              };
            });
          }
        } catch { /* ignore parse errors */ }
      };
    } catch {
      ws = null;
    }

    try {
      const result = await postprocessApi.refineSplat(projectId, { preset, mode: params?.mode });
      set((s) => ({
        isRefining: false,
        refineResult: result,
        plyVersion: s.plyVersion + 1,
      }));
      get().fetchQualityStats(projectId);
      get().fetchComparisonInfo(projectId);
      get().fetchModelInfo(projectId);
      toast.success("Refinement complete");
    } catch (e) {
      console.error("Failed to refine:", e);
      set({ isRefining: false });
      toast.error(`Refinement failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      if (ws && ws.readyState <= WebSocket.OPEN) {
        ws.close();
      }
    }
  },

  stopRefinement: async (projectId: string) => {
    try {
      await postprocessApi.stopRefinement(projectId);
      toast.success("Stop requested — finishing current step...");
    } catch (e) {
      console.error("Failed to stop refinement:", e);
      toast.error("Failed to stop refinement");
    }
  },

  setRefinePreset: (preset: string) => set({ refinePreset: preset }),

  fetchQualityStats: async (projectId: string) => {
    try {
      const stats = await postprocessApi.getQualityStats(projectId);
      set({ qualityStats: stats });
    } catch (e) {
      console.error("Failed to fetch quality stats:", e);
    }
  },

  fetchComparisonInfo: async (projectId) => {
    try {
      const data = await postprocessApi.getComparisonInfo(projectId);
      set({ comparisonPairs: data.pairs });
    } catch {
      set({ comparisonPairs: [] });
    }
  },

  setShowComparison: (show) => set({ showComparison: show }),
  setComparisonBeforeUrl: (url) => set({ comparisonBeforeUrl: url }),

  fetchModelInfo: async (projectId: string) => {
    try {
      const info = await postprocessApi.getModelInfo(projectId);
      set({
        modelInfo: info,
        activeStageId: info.current_stage,
      });
    } catch (e) {
      console.error("Failed to fetch model info:", e);
    }
  },

  switchStage: (projectId: string, stageId: string) => {
    const { modelInfo } = get();
    if (!modelInfo) return;

    const stage = modelInfo.stages.find((s) => s.id === stageId);
    if (!stage || !stage.exists) return;

    // Close any active comparison
    set({ showComparison: false, comparisonBeforeUrl: null });

    // Determine URL: prefer the active format if available, else use PLY
    const activeFormat = get().activeFormat;
    const file = stage.files[activeFormat] ?? stage.files.ply;
    if (!file) return;

    set((s) => ({
      activeStageId: stageId,
      plyUrl: file.url,
      plyVersion: s.plyVersion + 1,
    }));
  },

  setActiveFormat: (projectId: string, format: "ply" | "spz") => {
    const { modelInfo, activeStageId } = get();
    set({ activeFormat: format });

    if (!modelInfo || !activeStageId) return;

    // Only reload if we're on the current stage (which has both formats)
    const stage = modelInfo.stages.find((s) => s.id === activeStageId);
    if (!stage || !stage.exists) return;

    const file = stage.files[format] ?? stage.files.ply;
    if (!file) return;

    set((s) => ({
      plyUrl: file.url,
      plyVersion: s.plyVersion + 1,
    }));
  },

  reset: () => {
    set({
      frames: [],
      selectedCount: 0,
      framesLoading: false,
      maxViews: 32,
      resolution: 0,
      chunked: false,
      chunkSize: 32,
      chunkOverlap: 8,
      fps: 2,
      isRunning: false,
      isExtracting: false,
      isPruning: false,
      isRefining: false,
      refineStep: 0,
      refineTotal: 0,
      refineLoss: null,
      refineLossHistory: [],
      refineGaussianCount: null,
      refineStartTime: null,
      refineMetricsHistory: [],
      refineEvalHistory: [],
      refineBaseline: null,
      refinePreset: "balanced",
      lastRun: null,
      qualityStats: null,
      pruneResult: null,
      refineResult: null,
      plyUrl: null,
      plyVersion: 0,
      comparisonPairs: [],
      showComparison: false,
      comparisonBeforeUrl: null,
      modelInfo: null,
      activeStageId: null,
      activeFormat: "spz",
    });
  },
}));
