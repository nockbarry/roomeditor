import { create } from "zustand";
import type { FrameInfo, AnySplatRunResult } from "../types/api.ts";
import * as anysplatApi from "../api/anysplat.ts";
import type { QualityStats, PruneResult, RefineResult } from "../api/postprocess.ts";
import * as postprocessApi from "../api/postprocess.ts";

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
  lastRun: AnySplatRunResult | null;
  qualityStats: QualityStats | null;
  pruneResult: PruneResult | null;
  refineResult: RefineResult | null;
  plyUrl: string | null;
  plyVersion: number;

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
  refineSplat: (projectId: string, iterations?: number) => Promise<void>;
  fetchQualityStats: (projectId: string) => Promise<void>;
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
  lastRun: null,
  qualityStats: null,
  pruneResult: null,
  refineResult: null,
  plyUrl: null,
  plyVersion: 0,

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
    } catch (e) {
      console.error("Failed to extract frames:", e);
      set({ framesLoading: false, isExtracting: false });
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
      // Auto-fetch quality stats after rebuild
      get().fetchQualityStats(projectId);
    } catch (e) {
      console.error("Failed to run AnySplat:", e);
      set({ isRunning: false });
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
    } catch (e) {
      console.error("Failed to prune:", e);
      set({ isPruning: false });
    }
  },

  refineSplat: async (projectId: string, iterations?: number) => {
    set({ isRefining: true, refineStep: 0, refineTotal: iterations ?? 2000, refineLoss: null });

    // Open WebSocket to receive streaming snapshots during refinement
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
              plyVersion: s.plyVersion + 1,
            }));
          } else if (data.type === "refine_progress") {
            set({ refineStep: Math.round(data.progress * (get().refineTotal || 2000)) });
          }
        } catch { /* ignore parse errors */ }
      };
    } catch {
      // WS connection failed — refinement still works, just no streaming
      ws = null;
    }

    try {
      const result = await postprocessApi.refineSplat(projectId, { iterations });
      set((s) => ({
        isRefining: false,
        refineResult: result,
        refineStep: 0,
        refineTotal: 0,
        refineLoss: null,
        plyVersion: s.plyVersion + 1,
      }));
      get().fetchQualityStats(projectId);
    } catch (e) {
      console.error("Failed to refine:", e);
      set({ isRefining: false, refineStep: 0, refineTotal: 0, refineLoss: null });
    } finally {
      if (ws && ws.readyState <= WebSocket.OPEN) {
        ws.close();
      }
    }
  },

  fetchQualityStats: async (projectId: string) => {
    try {
      const stats = await postprocessApi.getQualityStats(projectId);
      set({ qualityStats: stats });
    } catch (e) {
      console.error("Failed to fetch quality stats:", e);
    }
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
      lastRun: null,
      qualityStats: null,
      pruneResult: null,
      refineResult: null,
      plyUrl: null,
      plyVersion: 0,
    });
  },
}));
