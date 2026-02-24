import { useCallback } from "react";
import { useEditorStore } from "../stores/editorStore.ts";
import { useAnySplatStore } from "../stores/anysplatStore.ts";
import * as editApi from "../api/editing.ts";
import { toast } from "../stores/toastStore.ts";

export interface CropBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface CropSphere {
  center: [number, number, number];
  radius: number;
}

/**
 * Actions for crop box and crop sphere tools.
 * The visual overlay (wireframe box/sphere) is rendered by CropOverlay.tsx.
 * This hook provides the apply/cancel actions.
 */
export function useCropActions(projectId: string) {
  const cropMode = useEditorStore((s) => s.cropMode);

  const applyCropBox = useCallback(
    async (box: CropBox) => {
      try {
        const result = await editApi.deleteRegion(projectId, {
          shape: "box",
          min: box.min,
          max: box.max,
          mode: cropMode === "delete-inside" ? "inside" : "outside",
        });
        useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
        useEditorStore.setState({
          undoCount: result.undo_count,
          redoCount: result.redo_count,
          isDirty: true,
        });
        toast.success(
          `Deleted ${result.n_affected} gaussians ${cropMode === "delete-inside" ? "inside" : "outside"} box`,
        );
      } catch (e) {
        console.error("Crop box failed:", e);
        toast.error("Crop failed");
      }
    },
    [projectId, cropMode],
  );

  const applyCropSphere = useCallback(
    async (sphere: CropSphere) => {
      try {
        const result = await editApi.deleteRegion(projectId, {
          shape: "sphere",
          center: sphere.center,
          radius: sphere.radius,
          mode: cropMode === "delete-inside" ? "inside" : "outside",
        });
        useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
        useEditorStore.setState({
          undoCount: result.undo_count,
          redoCount: result.redo_count,
          isDirty: true,
        });
        toast.success(
          `Deleted ${result.n_affected} gaussians ${cropMode === "delete-inside" ? "inside" : "outside"} sphere`,
        );
      } catch (e) {
        console.error("Crop sphere failed:", e);
        toast.error("Crop failed");
      }
    },
    [projectId, cropMode],
  );

  return { applyCropBox, applyCropSphere };
}
