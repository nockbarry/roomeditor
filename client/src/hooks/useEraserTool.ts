import { useCallback } from "react";
import { useEditorStore } from "../stores/editorStore.ts";
import { toast } from "../stores/toastStore.ts";

/**
 * Eraser tool actions. The visual stroke uses useBrushTool with toolMode="eraser".
 * This hook provides the finalize action: soft-delete the brush selection on mouseup.
 * Deleted gaussians are hidden client-side via splatRgba (no scene reload needed).
 */
export function useEraserActions(projectId: string) {
  const brushSelection = useEditorStore((s) => s.brushSelection);
  const clearBrushSelection = useEditorStore((s) => s.clearBrushSelection);
  const addDeletedIndices = useEditorStore((s) => s.addDeletedIndices);
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
      // Hide deleted gaussians client-side (instant, no scene reload)
      addDeletedIndices(indices);
      toast.success(`Erased ${result.n_affected} gaussians`);
    }
  }, [projectId, brushSelection, clearBrushSelection, addDeletedIndices, applyEdit]);

  return { finalizeErase };
}
