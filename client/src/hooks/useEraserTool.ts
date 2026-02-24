import { useCallback } from "react";
import { useEditorStore } from "../stores/editorStore.ts";
import { useAnySplatStore } from "../stores/anysplatStore.ts";
import { toast } from "../stores/toastStore.ts";

/**
 * Eraser tool actions. The visual stroke uses useBrushTool with toolMode="eraser".
 * This hook provides the finalize action: soft-delete the brush selection on mouseup.
 */
export function useEraserActions(projectId: string) {
  const brushSelection = useEditorStore((s) => s.brushSelection);
  const clearBrushSelection = useEditorStore((s) => s.clearBrushSelection);
  const applyEdit = useEditorStore((s) => s.applyEdit);

  const finalizeErase = useCallback(async () => {
    if (brushSelection.size === 0) return;

    const indices = Array.from(brushSelection);
    clearBrushSelection();

    const result = await applyEdit(projectId, {
      type: "delete",
      indices,
    });

    if (result) {
      // Bump ply version to trigger scene reload
      useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
      toast.success(`Erased ${result.n_affected} gaussians`);
    }
  }, [projectId, brushSelection, clearBrushSelection, applyEdit]);

  return { finalizeErase };
}
