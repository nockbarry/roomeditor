import { create } from "zustand";
import { toast } from "./toastStore.ts";
import * as segApi from "../api/segments.ts";

export interface ImportedObject {
  id: string;
  sourceProjectId: string;
  segmentId: number;
  label: string;
  plyUrl: string;
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
}

interface CompositionStore {
  importedObjects: ImportedObject[];
  selectedImportId: string | null;

  importSegment: (
    sourceProjectId: string,
    segmentId: number,
    label: string
  ) => Promise<void>;
  removeImported: (id: string) => void;
  updateImported: (id: string, update: Partial<ImportedObject>) => void;
  selectImported: (id: string | null) => void;
  reset: () => void;
}

let importCount = 0;

export const useCompositionStore = create<CompositionStore>((set, get) => ({
  importedObjects: [],
  selectedImportId: null,

  importSegment: async (sourceProjectId, segmentId, label) => {
    try {
      const result = await segApi.exportSegmentPly(sourceProjectId, segmentId);
      const obj: ImportedObject = {
        id: `import_${++importCount}_${Date.now()}`,
        sourceProjectId,
        segmentId,
        label,
        plyUrl: result.ply_url,
        position: [0, 0, 0],
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
        visible: true,
      };
      set((s) => ({ importedObjects: [...s.importedObjects, obj] }));
      toast.success(`Imported "${label}"`);
    } catch (e) {
      console.error("Import failed:", e);
      toast.error("Failed to import segment");
    }
  },

  removeImported: (id) => {
    set((s) => ({
      importedObjects: s.importedObjects.filter((o) => o.id !== id),
      selectedImportId: s.selectedImportId === id ? null : s.selectedImportId,
    }));
  },

  updateImported: (id, update) => {
    set((s) => ({
      importedObjects: s.importedObjects.map((o) =>
        o.id === id ? { ...o, ...update } : o
      ),
    }));
  },

  selectImported: (id) => set({ selectedImportId: id }),

  reset: () =>
    set({
      importedObjects: [],
      selectedImportId: null,
    }),
}));
