import { create } from "zustand";
import type { SegmentInfo } from "../types/api.ts";
import * as segApi from "../api/segments.ts";

interface SegmentStore {
  segments: SegmentInfo[];
  selectedSegmentIds: number[];
  hoveredSegmentId: number | null;
  isSegmenting: boolean;
  isAssigning: boolean;
  totalGaussians: number;
  unassignedGaussians: number;
  undoCount: number;
  segmentIndexMap: Uint8Array | null;

  // Actions
  setHoveredSegment: (id: number | null) => void;
  fetchSegmentIndexMap: (projectId: string) => Promise<void>;
  fetchSegments: (projectId: string) => Promise<void>;
  autoSegment: (projectId: string) => Promise<void>;
  clickSegment: (projectId: string, frame: string, x: number, y: number) => Promise<void>;
  assignGaussians: (projectId: string) => Promise<void>;
  selectSegment: (id: number | null) => void;
  toggleSelectSegment: (id: number) => void;
  rangeSelectSegment: (id: number) => void;
  transformSegment: (
    projectId: string,
    segmentId: number,
    transform: { translation?: number[]; rotation?: number[]; scale?: number[] }
  ) => Promise<void>;
  batchTransform: (
    projectId: string,
    transform: { translation?: number[]; rotation?: number[]; scale?: number[] }
  ) => Promise<void>;
  deleteSegment: (projectId: string, segmentId: number) => Promise<void>;
  toggleVisibility: (projectId: string, segmentId: number, visible: boolean) => Promise<void>;
  duplicateSegment: (projectId: string, segmentId: number) => Promise<void>;
  renameSegment: (projectId: string, segmentId: number, label: string) => Promise<void>;
  createBackground: (projectId: string) => Promise<void>;
  undo: (projectId: string) => Promise<void>;
  restoreOriginal: (projectId: string) => Promise<void>;
  fetchUndoCount: (projectId: string) => Promise<void>;
  reset: () => void;
}

export const useSegmentStore = create<SegmentStore>((set, get) => ({
  segments: [],
  selectedSegmentIds: [],
  hoveredSegmentId: null,
  isSegmenting: false,
  isAssigning: false,
  totalGaussians: 0,
  unassignedGaussians: 0,
  undoCount: 0,
  segmentIndexMap: null,

  setHoveredSegment: (id) => set({ hoveredSegmentId: id }),

  fetchSegmentIndexMap: async (projectId) => {
    try {
      const map = await segApi.fetchSegmentIndexMap(projectId);
      set({ segmentIndexMap: map });
    } catch (e) {
      console.error("Failed to fetch segment index map:", e);
    }
  },

  fetchSegments: async (projectId) => {
    try {
      const manifest = await segApi.getSegments(projectId);
      set({
        segments: manifest.segments,
        totalGaussians: manifest.total_gaussians,
        unassignedGaussians: manifest.unassigned_gaussians,
      });
      // If segments have gaussians assigned, fetch the index map
      if (manifest.segments.some((s) => s.n_gaussians > 0)) {
        get().fetchSegmentIndexMap(projectId);
      }
    } catch (e) {
      console.error("Failed to fetch segments:", e);
    }
  },

  autoSegment: async (projectId) => {
    set({ isSegmenting: true });
    try {
      const manifest = await segApi.autoSegment(projectId);
      set({
        segments: manifest.segments,
        totalGaussians: manifest.total_gaussians,
        unassignedGaussians: manifest.unassigned_gaussians,
        isSegmenting: false,
      });
    } catch (e) {
      console.error("Auto-segment failed:", e);
      set({ isSegmenting: false });
    }
  },

  clickSegment: async (projectId, frame, x, y) => {
    set({ isSegmenting: true });
    try {
      const segment = await segApi.clickSegment(projectId, frame, x, y);
      set((state) => ({
        segments: [...state.segments, segment],
        selectedSegmentIds: [segment.id],
        isSegmenting: false,
      }));
    } catch (e) {
      console.error("Click-segment failed:", e);
      set({ isSegmenting: false });
    }
  },

  assignGaussians: async (projectId) => {
    set({ isAssigning: true });
    try {
      const manifest = await segApi.assignGaussians(projectId);
      set({
        segments: manifest.segments,
        totalGaussians: manifest.total_gaussians,
        unassignedGaussians: manifest.unassigned_gaussians,
        isAssigning: false,
      });
      // Fetch index map after gaussian assignment
      get().fetchSegmentIndexMap(projectId);
    } catch (e) {
      console.error("Assign gaussians failed:", e);
      set({ isAssigning: false });
    }
  },

  selectSegment: (id) =>
    set({ selectedSegmentIds: id !== null ? [id] : [] }),

  toggleSelectSegment: (id) =>
    set((state) => {
      const ids = state.selectedSegmentIds;
      if (ids.includes(id)) {
        return { selectedSegmentIds: ids.filter((i) => i !== id) };
      }
      return { selectedSegmentIds: [...ids, id] };
    }),

  rangeSelectSegment: (id) =>
    set((state) => {
      const { segments, selectedSegmentIds } = state;
      if (selectedSegmentIds.length === 0) {
        return { selectedSegmentIds: [id] };
      }
      const lastSelected = selectedSegmentIds[selectedSegmentIds.length - 1];
      const allIds = segments.map((s) => s.id);
      const fromIdx = allIds.indexOf(lastSelected);
      const toIdx = allIds.indexOf(id);
      if (fromIdx === -1 || toIdx === -1) return { selectedSegmentIds: [id] };
      const start = Math.min(fromIdx, toIdx);
      const end = Math.max(fromIdx, toIdx);
      const rangeIds = allIds.slice(start, end + 1);
      const merged = new Set([...selectedSegmentIds, ...rangeIds]);
      return { selectedSegmentIds: [...merged] };
    }),

  transformSegment: async (projectId, segmentId, transform) => {
    try {
      await segApi.transformSegment(projectId, segmentId, transform);
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Transform failed:", e);
    }
  },

  batchTransform: async (projectId, transform) => {
    const { selectedSegmentIds } = get();
    if (selectedSegmentIds.length === 0) return;
    try {
      if (selectedSegmentIds.length === 1) {
        await segApi.transformSegment(projectId, selectedSegmentIds[0], transform);
      } else {
        await segApi.batchTransform(projectId, selectedSegmentIds, transform);
      }
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Batch transform failed:", e);
    }
  },

  deleteSegment: async (projectId, segmentId) => {
    try {
      await segApi.deleteSegment(projectId, segmentId);
      set((state) => ({
        segments: state.segments.filter((s) => s.id !== segmentId),
        selectedSegmentIds: state.selectedSegmentIds.filter((i) => i !== segmentId),
      }));
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Delete segment failed:", e);
    }
  },

  toggleVisibility: async (projectId, segmentId, visible) => {
    try {
      await segApi.toggleVisibility(projectId, segmentId, visible);
      set((state) => ({
        segments: state.segments.map((s) =>
          s.id === segmentId ? { ...s, visible } : s
        ),
      }));
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Toggle visibility failed:", e);
    }
  },

  duplicateSegment: async (projectId, segmentId) => {
    try {
      const result = await segApi.duplicateSegment(projectId, segmentId);
      // Refetch segments to get full data
      await get().fetchSegments(projectId);
      set({ selectedSegmentIds: [result.new_segment.id] });
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Duplicate failed:", e);
    }
  },

  renameSegment: async (projectId, segmentId, label) => {
    try {
      await segApi.renameSegment(projectId, segmentId, label);
      set((state) => ({
        segments: state.segments.map((s) =>
          s.id === segmentId ? { ...s, label } : s
        ),
      }));
    } catch (e) {
      console.error("Rename failed:", e);
    }
  },

  createBackground: async (projectId) => {
    try {
      await segApi.createBackground(projectId);
      await get().fetchSegments(projectId);
    } catch (e) {
      console.error("Create background failed:", e);
    }
  },

  undo: async (projectId) => {
    try {
      const result = await segApi.undo(projectId);
      set({ undoCount: result.remaining });
      await get().fetchSegments(projectId);
    } catch (e) {
      console.error("Undo failed:", e);
    }
  },

  restoreOriginal: async (projectId) => {
    try {
      await segApi.restoreOriginal(projectId);
      set({ undoCount: 0 });
      await get().fetchSegments(projectId);
    } catch (e) {
      console.error("Restore original failed:", e);
    }
  },

  fetchUndoCount: async (projectId) => {
    try {
      const result = await segApi.getUndoStack(projectId);
      set({ undoCount: result.count });
    } catch {
      // ignore
    }
  },

  reset: () =>
    set({
      segments: [],
      selectedSegmentIds: [],
      hoveredSegmentId: null,
      isSegmenting: false,
      isAssigning: false,
      totalGaussians: 0,
      unassignedGaussians: 0,
      undoCount: 0,
      segmentIndexMap: null,
    }),
}));
