import { create } from "zustand";
import { persist } from "zustand/middleware";
import { useAnySplatStore } from "./anysplatStore.ts";

export interface UserPreset {
  id: string;
  name: string;
  maxViews: number;
  resolution: number;
  chunked: boolean;
  chunkSize: number;
  chunkOverlap: number;
  fps: number;
}

interface PresetStore {
  userPresets: UserPreset[];
  savePreset: (name: string) => void;
  deletePreset: (id: string) => void;
  applyPreset: (id: string) => void;
}

let presetId = 0;

export const usePresetStore = create<PresetStore>()(
  persist(
    (set, get) => ({
      userPresets: [],

      savePreset: (name) => {
        const as = useAnySplatStore.getState();
        const preset: UserPreset = {
          id: `preset_${Date.now()}_${++presetId}`,
          name,
          maxViews: as.maxViews,
          resolution: as.resolution,
          chunked: as.chunked,
          chunkSize: as.chunkSize,
          chunkOverlap: as.chunkOverlap,
          fps: as.fps,
        };
        set((s) => ({ userPresets: [...s.userPresets, preset] }));
      },

      deletePreset: (id) => {
        set((s) => ({
          userPresets: s.userPresets.filter((p) => p.id !== id),
        }));
      },

      applyPreset: (id) => {
        const preset = get().userPresets.find((p) => p.id === id);
        if (!preset) return;
        const as = useAnySplatStore.getState();
        as.setMaxViews(preset.maxViews);
        as.setResolution(preset.resolution);
        as.setChunked(preset.chunked);
        as.setChunkSize(preset.chunkSize);
        as.setChunkOverlap(preset.chunkOverlap);
        as.setFps(preset.fps);
      },
    }),
    { name: "roomeditor-presets" }
  )
);
