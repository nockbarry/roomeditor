import { create } from "zustand";
import type { ToolMode, CameraMode } from "../types/scene.ts";

interface EditorStore {
  toolMode: ToolMode;
  cameraMode: CameraMode;
  selectedObjectId: string | null;
  showGrid: boolean;

  setToolMode: (mode: ToolMode) => void;
  setCameraMode: (mode: CameraMode) => void;
  selectObject: (id: string | null) => void;
  toggleGrid: () => void;
}

export const useEditorStore = create<EditorStore>((set) => ({
  toolMode: "select",
  cameraMode: "orbit",
  selectedObjectId: null,
  showGrid: true,

  setToolMode: (mode) => set({ toolMode: mode }),
  setCameraMode: (mode) => set({ cameraMode: mode }),
  selectObject: (id) => set({ selectedObjectId: id }),
  toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),
}));
