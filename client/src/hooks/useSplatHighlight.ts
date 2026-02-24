import { useEffect, useRef, type RefObject } from "react";
import { SplatMesh, RgbaArray } from "@sparkjsdev/spark";
import { useSegmentStore } from "../stores/segmentStore.ts";
import { useEditorStore } from "../stores/editorStore.ts";

/**
 * Per-splat RGBA highlighting with multiple visual states.
 *
 * States (splatRgba is a multiplicative blend with original color):
 * - Selected:      (200, 255, 220, 255) — subtle emerald tint, preserves color
 * - Hovered:       (200, 210, 255, 240) — slight blue tint
 * - Brush preview: (255, 200, 150, 200) — orange tint for brush/eraser preview
 * - Dimmed:        (180, 180, 180, 220) — desaturated but still readable (~70%)
 * - No active state: splatRgba = null → original colors
 */
export function useSplatHighlight(
  splatMeshRef: RefObject<SplatMesh | null>,
): void {
  const selectedSegmentIds = useSegmentStore((s) => s.selectedSegmentIds);
  const hoveredSegmentId = useSegmentStore((s) => s.hoveredSegmentId);
  const segments = useSegmentStore((s) => s.segments);
  const segmentIndexMap = useSegmentStore((s) => s.segmentIndexMap);
  const brushSelection = useEditorStore((s) => s.brushSelection);
  const toolMode = useEditorStore((s) => s.toolMode);

  const rgbaRef = useRef<RgbaArray | null>(null);

  useEffect(() => {
    const mesh = splatMeshRef.current;
    if (!mesh) return;

    const hasSelection = selectedSegmentIds.length > 0;
    const hasHover = hoveredSegmentId !== null;
    const hasBrush = brushSelection.size > 0;
    const isBrushTool = toolMode === "brush" || toolMode === "eraser";

    // If nothing active, clear highlight
    if (
      (!hasSelection && !hasHover && !hasBrush) ||
      (!segmentIndexMap && !hasBrush) ||
      (segmentIndexMap && segmentIndexMap.length !== mesh.numSplats && !hasBrush)
    ) {
      if (rgbaRef.current) {
        mesh.splatRgba = null;
        rgbaRef.current.dispose();
        rgbaRef.current = null;
      }
      return;
    }

    const numSplats = mesh.numSplats;
    const rgba = new Uint8Array(numSplats * 4);

    // Build sets of selected segment values (1-based indices into segments array)
    const selectedValues = new Set<number>();
    for (const segId of selectedSegmentIds) {
      const idx = segments.findIndex((s) => s.id === segId);
      if (idx >= 0) selectedValues.add(idx + 1);
    }

    let hoveredValue = -1;
    if (hoveredSegmentId !== null) {
      const idx = segments.findIndex((s) => s.id === hoveredSegmentId);
      if (idx >= 0) hoveredValue = idx + 1;
    }

    const anythingActive = hasSelection || hasHover || hasBrush;

    for (let i = 0; i < numSplats; i++) {
      const off = i * 4;
      const segValue = segmentIndexMap ? segmentIndexMap[i] : 0;

      if (hasBrush && brushSelection.has(i)) {
        // Brush/eraser preview: orange for brush, red for eraser
        if (toolMode === "eraser") {
          rgba[off] = 255;
          rgba[off + 1] = 100;
          rgba[off + 2] = 100;
          rgba[off + 3] = 200;
        } else {
          rgba[off] = 255;
          rgba[off + 1] = 200;
          rgba[off + 2] = 150;
          rgba[off + 3] = 200;
        }
      } else if (selectedValues.has(segValue)) {
        // Selected: emerald tint
        rgba[off] = 200;
        rgba[off + 1] = 255;
        rgba[off + 2] = 220;
        rgba[off + 3] = 255;
      } else if (segValue === hoveredValue) {
        // Hovered: blue tint
        rgba[off] = 200;
        rgba[off + 1] = 210;
        rgba[off + 2] = 255;
        rgba[off + 3] = 240;
      } else if (anythingActive && !isBrushTool) {
        // Dimmed: desaturated but still readable
        rgba[off] = 180;
        rgba[off + 1] = 180;
        rgba[off + 2] = 180;
        rgba[off + 3] = 220;
      } else {
        // Identity: no modification
        rgba[off] = 255;
        rgba[off + 1] = 255;
        rgba[off + 2] = 255;
        rgba[off + 3] = 255;
      }
    }

    if (rgbaRef.current) {
      rgbaRef.current.dispose();
    }

    const rgbaArray = new RgbaArray({ array: rgba, count: numSplats });
    rgbaRef.current = rgbaArray;
    mesh.splatRgba = rgbaArray;
  }, [selectedSegmentIds, hoveredSegmentId, segments, segmentIndexMap, brushSelection, toolMode, splatMeshRef]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rgbaRef.current) {
        const mesh = splatMeshRef.current;
        if (mesh) mesh.splatRgba = null;
        rgbaRef.current.dispose();
        rgbaRef.current = null;
      }
    };
  }, [splatMeshRef]);
}
