import { create } from "zustand";
import type { SegmentInfo } from "../types/api.ts";
import * as segApi from "../api/segments.ts";
import { toast } from "./toastStore.ts";

interface SegmentStore {
  segments: SegmentInfo[];
  selectedSegmentIds: number[];
  hoveredSegmentId: number | null;
  isSegmenting: boolean;
  isAssigning: boolean;
  isClassifying: boolean;
  isMerging: boolean;
  isSplitting: boolean;
  isInpainting: boolean;
  totalGaussians: number;
  unassignedGaussians: number;
  undoCount: number;
  segmentIndexMap: Uint8Array | null;

  // Actions
  setHoveredSegment: (id: number | null) => void;
  fetchSegmentIndexMap: (projectId: string) => Promise<void>;
  fetchSegments: (projectId: string) => Promise<void>;
  autoSegment: (projectId: string) => Promise<void>;
  autoSegmentFull: (projectId: string) => Promise<void>;
  clickSegment: (projectId: string, frame: string, x: number, y: number) => Promise<void>;
  assignGaussians: (projectId: string) => Promise<void>;
  classifySegments: (projectId: string) => Promise<void>;
  mergeSegments: (projectId: string, segmentIds: number[], label?: string) => Promise<void>;
  splitSegment: (projectId: string, segmentId: number, nClusters?: number) => Promise<void>;
  adjustLighting: (
    projectId: string,
    segmentId: number,
    params: { brightness?: number; color_tint?: number[]; sh_scale?: number }
  ) => Promise<void>;
  inpaintRemove: (projectId: string, segmentId: number) => Promise<void>;
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
  isClassifying: false,
  isMerging: false,
  isSplitting: false,
  isInpainting: false,
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
      toast.success(`Detected ${manifest.segments.length} objects`);
    } catch (e) {
      console.error("Auto-segment failed:", e);
      set({ isSegmenting: false });
      toast.error("Auto-segment failed");
    }
  },

  autoSegmentFull: async (projectId) => {
    set({ isSegmenting: true });
    try {
      const manifest = await segApi.autoSegmentFull(projectId);
      set({
        segments: manifest.segments,
        totalGaussians: manifest.total_gaussians,
        unassignedGaussians: manifest.unassigned_gaussians,
        isSegmenting: false,
      });
      // Fetch index map since gaussians are already assigned
      get().fetchSegmentIndexMap(projectId);
      toast.success(`Segmented scene — ${manifest.segments.length} objects`);
    } catch (e) {
      console.error("Auto-segment-full failed:", e);
      set({ isSegmenting: false });
      toast.error("Scene segmentation failed");
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
      toast.success(`Added segment: ${segment.label}`);
    } catch (e) {
      console.error("Click-segment failed:", e);
      set({ isSegmenting: false });
      toast.error("Click segmentation failed");
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
      toast.success("Gaussians assigned to segments");
    } catch (e) {
      console.error("Assign gaussians failed:", e);
      set({ isAssigning: false });
      toast.error("Gaussian assignment failed");
    }
  },

  classifySegments: async (projectId) => {
    set({ isClassifying: true });
    try {
      const manifest = await segApi.classifySegments(projectId);
      set({
        segments: manifest.segments,
        isClassifying: false,
      });
      toast.success("Segments labeled with CLIP");
    } catch (e) {
      console.error("Classification failed:", e);
      set({ isClassifying: false });
      toast.error("CLIP classification failed");
    }
  },

  mergeSegments: async (projectId, segmentIds, label) => {
    set({ isMerging: true });
    try {
      const manifest = await segApi.mergeSegments(projectId, segmentIds, label);
      set({
        segments: manifest.segments,
        totalGaussians: manifest.total_gaussians,
        unassignedGaussians: manifest.unassigned_gaussians,
        selectedSegmentIds: [],
        isMerging: false,
      });
      get().fetchSegmentIndexMap(projectId);
      get().fetchUndoCount(projectId);
      toast.success(`Merged ${segmentIds.length} segments`);
    } catch (e) {
      console.error("Merge failed:", e);
      set({ isMerging: false });
      toast.error("Merge failed");
    }
  },

  splitSegment: async (projectId, segmentId, nClusters = 2) => {
    set({ isSplitting: true });
    try {
      const manifest = await segApi.splitSegment(projectId, segmentId, nClusters);
      set({
        segments: manifest.segments,
        totalGaussians: manifest.total_gaussians,
        unassignedGaussians: manifest.unassigned_gaussians,
        selectedSegmentIds: [],
        isSplitting: false,
      });
      get().fetchSegmentIndexMap(projectId);
      get().fetchUndoCount(projectId);
      toast.success(`Split into ${nClusters} segments`);
    } catch (e) {
      console.error("Split failed:", e);
      set({ isSplitting: false });
      toast.error("Split failed");
    }
  },

  adjustLighting: async (projectId, segmentId, params) => {
    try {
      await segApi.adjustLighting(projectId, segmentId, params);
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Lighting adjustment failed:", e);
      toast.error("Lighting adjustment failed");
    }
  },

  inpaintRemove: async (projectId, segmentId) => {
    set({ isInpainting: true });
    try {
      await segApi.inpaintRemoveSegment(projectId, segmentId);
      await get().fetchSegments(projectId);
      set({ isInpainting: false });
      get().fetchUndoCount(projectId);
      toast.success("Object removed and scene inpainted");
    } catch (e) {
      console.error("Inpaint failed:", e);
      set({ isInpainting: false });
      toast.error("Inpainting failed");
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
      toast.error("Transform failed");
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
      toast.error("Transform failed");
    }
  },

  deleteSegment: async (projectId, segmentId) => {
    // Optimistic: remove segment immediately
    const prev = get().segments;
    const prevSelected = get().selectedSegmentIds;
    set((state) => ({
      segments: state.segments.filter((s) => s.id !== segmentId),
      selectedSegmentIds: state.selectedSegmentIds.filter((i) => i !== segmentId),
    }));
    try {
      await segApi.deleteSegment(projectId, segmentId);
      get().fetchUndoCount(projectId);
      toast.success("Segment deleted");
    } catch (e) {
      console.error("Delete segment failed:", e);
      set({ segments: prev, selectedSegmentIds: prevSelected });
      toast.error("Failed to delete segment");
    }
  },

  toggleVisibility: async (projectId, segmentId, visible) => {
    // Optimistic: update visibility immediately
    const prev = get().segments;
    set((state) => ({
      segments: state.segments.map((s) =>
        s.id === segmentId ? { ...s, visible } : s
      ),
    }));
    try {
      await segApi.toggleVisibility(projectId, segmentId, visible);
      get().fetchUndoCount(projectId);
    } catch (e) {
      console.error("Toggle visibility failed:", e);
      set({ segments: prev });
      toast.error("Failed to toggle visibility");
    }
  },

  duplicateSegment: async (projectId, segmentId) => {
    try {
      const result = await segApi.duplicateSegment(projectId, segmentId);
      await get().fetchSegments(projectId);
      set({ selectedSegmentIds: [result.new_segment.id] });
      get().fetchUndoCount(projectId);
      toast.success(`Duplicated segment`);
    } catch (e) {
      console.error("Duplicate failed:", e);
      toast.error("Failed to duplicate segment");
    }
  },

  renameSegment: async (projectId, segmentId, label) => {
    // Optimistic: update label immediately
    const prev = get().segments;
    set((state) => ({
      segments: state.segments.map((s) =>
        s.id === segmentId ? { ...s, label } : s
      ),
    }));
    try {
      await segApi.renameSegment(projectId, segmentId, label);
    } catch (e) {
      console.error("Rename failed:", e);
      set({ segments: prev });
      toast.error("Failed to rename segment");
    }
  },

  createBackground: async (projectId) => {
    try {
      await segApi.createBackground(projectId);
      await get().fetchSegments(projectId);
      toast.success("Background segment created");
    } catch (e) {
      console.error("Create background failed:", e);
      toast.error("Failed to create background");
    }
  },

  undo: async (projectId) => {
    try {
      const result = await segApi.undo(projectId);
      set({ undoCount: result.remaining });
      await get().fetchSegments(projectId);
      toast.info("Undo applied");
    } catch (e) {
      console.error("Undo failed:", e);
      toast.error("Undo failed");
    }
  },

  restoreOriginal: async (projectId) => {
    try {
      await segApi.restoreOriginal(projectId);
      set({ undoCount: 0 });
      await get().fetchSegments(projectId);
      toast.success("Restored to original");
    } catch (e) {
      console.error("Restore original failed:", e);
      toast.error("Failed to restore original");
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
      isClassifying: false,
      isMerging: false,
      isSplitting: false,
      isInpainting: false,
      totalGaussians: 0,
      unassignedGaussians: 0,
      undoCount: 0,
      segmentIndexMap: null,
    }),
}));
