import { create } from "zustand";
import type { ToolMode, CameraMode } from "../types/scene.ts";
import * as editApi from "../api/editing.ts";
import type { EditRequest, HistoryEntry } from "../api/editing.ts";
import { toast } from "./toastStore.ts";

interface EditorStore {
  toolMode: ToolMode;
  cameraMode: CameraMode;
  selectedObjectId: string | null;
  showGrid: boolean;

  // Scene editing state
  sceneLoaded: boolean;
  undoCount: number;
  redoCount: number;
  editHistory: HistoryEntry[];
  isDirty: boolean;
  isSaving: boolean;

  // Brush/eraser state
  brushRadius: number;
  brushMode: "select" | "deselect";
  brushSelection: Set<number>;

  // Crop state
  cropMode: "delete-inside" | "delete-outside";
  cropBox: { min: [number, number, number]; max: [number, number, number] } | null;
  cropSphere: { center: [number, number, number]; radius: number } | null;

  // Actions
  setToolMode: (mode: ToolMode) => void;
  setCameraMode: (mode: CameraMode) => void;
  selectObject: (id: string | null) => void;
  toggleGrid: () => void;

  // Scene editing actions
  loadScene: (projectId: string) => Promise<void>;
  saveScene: (projectId: string) => Promise<void>;
  applyEdit: (projectId: string, edit: EditRequest) => Promise<editApi.EditResult | null>;
  undo: (projectId: string) => Promise<void>;
  redo: (projectId: string) => Promise<void>;
  fetchHistory: (projectId: string) => Promise<void>;

  // Brush actions
  setBrushRadius: (radius: number) => void;
  setBrushMode: (mode: "select" | "deselect") => void;
  addToBrushSelection: (indices: number[]) => void;
  removeFromBrushSelection: (indices: number[]) => void;
  clearBrushSelection: () => void;

  // Crop actions
  setCropMode: (mode: "delete-inside" | "delete-outside") => void;
  setCropBox: (box: { min: [number, number, number]; max: [number, number, number] } | null) => void;
  setCropSphere: (sphere: { center: [number, number, number]; radius: number } | null) => void;

  resetEditing: () => void;
}

export const useEditorStore = create<EditorStore>((set, get) => ({
  toolMode: "select",
  cameraMode: "orbit",
  selectedObjectId: null,
  showGrid: true,

  sceneLoaded: false,
  undoCount: 0,
  redoCount: 0,
  editHistory: [],
  isDirty: false,
  isSaving: false,

  brushRadius: 0.1,
  brushMode: "select",
  brushSelection: new Set(),

  cropMode: "delete-inside",
  cropBox: null,
  cropSphere: null,

  setToolMode: (mode) => set({ toolMode: mode }),
  setCameraMode: (mode) => set({ cameraMode: mode }),
  selectObject: (id) => set({ selectedObjectId: id }),
  toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),

  loadScene: async (projectId: string) => {
    try {
      const result = await editApi.loadScene(projectId);
      set({
        sceneLoaded: true,
        undoCount: result.undo_count,
        redoCount: result.redo_count,
        isDirty: result.dirty,
      });
    } catch (e) {
      console.error("Failed to load scene into memory:", e);
    }
  },

  saveScene: async (projectId: string) => {
    set({ isSaving: true });
    try {
      const result = await editApi.saveScene(projectId);
      set({ isSaving: false, isDirty: false });
      toast.success(`Scene saved (${result.save_time_sec}s)`);
    } catch (e) {
      console.error("Failed to save scene:", e);
      set({ isSaving: false });
      toast.error("Failed to save scene");
    }
  },

  applyEdit: async (projectId: string, edit: EditRequest) => {
    try {
      const result = await editApi.editScene(projectId, edit);
      set({
        undoCount: result.undo_count,
        redoCount: result.redo_count,
        isDirty: true,
      });
      return result;
    } catch (e) {
      console.error("Edit failed:", e);
      toast.error("Edit failed");
      return null;
    }
  },

  undo: async (projectId: string) => {
    try {
      const result = await editApi.undoEdit(projectId);
      set({
        undoCount: result.undo_count,
        redoCount: result.redo_count,
        isDirty: true,
      });
      toast.info(`Undo: ${result.label}`);
    } catch (e) {
      console.error("Undo failed:", e);
    }
  },

  redo: async (projectId: string) => {
    try {
      const result = await editApi.redoEdit(projectId);
      set({
        undoCount: result.undo_count,
        redoCount: result.redo_count,
        isDirty: true,
      });
      toast.info(`Redo: ${result.label}`);
    } catch (e) {
      console.error("Redo failed:", e);
    }
  },

  fetchHistory: async (projectId: string) => {
    try {
      const history = await editApi.getHistory(projectId);
      set({
        undoCount: history.undo_count,
        redoCount: history.redo_count,
        editHistory: history.undo,
        isDirty: history.dirty,
      });
    } catch {
      // ignore
    }
  },

  setBrushRadius: (radius) => set({ brushRadius: Math.max(0.01, radius) }),
  setBrushMode: (mode) => set({ brushMode: mode }),
  addToBrushSelection: (indices) =>
    set((state) => {
      const next = new Set(state.brushSelection);
      for (const i of indices) next.add(i);
      return { brushSelection: next };
    }),
  removeFromBrushSelection: (indices) =>
    set((state) => {
      const next = new Set(state.brushSelection);
      for (const i of indices) next.delete(i);
      return { brushSelection: next };
    }),
  clearBrushSelection: () => set({ brushSelection: new Set() }),

  setCropMode: (mode) => set({ cropMode: mode }),
  setCropBox: (box) => set({ cropBox: box }),
  setCropSphere: (sphere) => set({ cropSphere: sphere }),

  resetEditing: () =>
    set({
      sceneLoaded: false,
      undoCount: 0,
      redoCount: 0,
      editHistory: [],
      isDirty: false,
      isSaving: false,
      brushRadius: 0.1,
      brushMode: "select",
      brushSelection: new Set(),
      cropMode: "delete-inside",
      cropBox: null,
      cropSphere: null,
    }),
}));
