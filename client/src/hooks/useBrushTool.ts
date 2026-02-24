import { useEffect, useRef, useCallback, type RefObject } from "react";
import { useEditorStore } from "../stores/editorStore.ts";
import { queryRadius, type KDTreeNode } from "./useKDTree.ts";

/**
 * Brush select tool: hold LMB + drag over scene to select gaussians
 * within brushRadius of the 3D hit point. Uses client-side KD-tree
 * for real-time feedback.
 *
 * When active (toolMode === "brush"), overrides the default pick/hover behavior.
 */
export function useBrushTool(
  containerRef: RefObject<HTMLDivElement | null>,
  kdTreeRef: RefObject<KDTreeNode | null>,
  positionsRef: RefObject<Float32Array | null>,
  getHitPoint: (event: MouseEvent) => { x: number; y: number; z: number } | null,
) {
  const toolMode = useEditorStore((s) => s.toolMode);
  const brushRadius = useEditorStore((s) => s.brushRadius);
  const brushMode = useEditorStore((s) => s.brushMode);
  const addToBrushSelection = useEditorStore((s) => s.addToBrushSelection);
  const removeFromBrushSelection = useEditorStore((s) => s.removeFromBrushSelection);

  const isDragging = useRef(false);

  const handleBrushStroke = useCallback(
    (event: MouseEvent) => {
      const tree = kdTreeRef.current;
      const positions = positionsRef.current;
      if (!tree || !positions) return;

      const hit = getHitPoint(event);
      if (!hit) return;

      const indices = queryRadius(tree, positions, hit.x, hit.y, hit.z, brushRadius);
      if (indices.length === 0) return;

      const arr = Array.from(indices);
      if (brushMode === "deselect") {
        removeFromBrushSelection(arr);
      } else {
        addToBrushSelection(arr);
      }
    },
    [kdTreeRef, positionsRef, brushRadius, brushMode, getHitPoint, addToBrushSelection, removeFromBrushSelection],
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el || (toolMode !== "brush" && toolMode !== "eraser")) return;

    const onDown = (e: MouseEvent) => {
      if (e.button !== 0) return; // LMB only
      isDragging.current = true;
      handleBrushStroke(e);
    };

    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      handleBrushStroke(e);
    };

    const onUp = () => {
      isDragging.current = false;
    };

    el.addEventListener("mousedown", onDown);
    el.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    return () => {
      el.removeEventListener("mousedown", onDown);
      el.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      isDragging.current = false;
    };
  }, [containerRef, toolMode, handleBrushStroke]);
}
