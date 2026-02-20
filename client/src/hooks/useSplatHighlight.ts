import { useEffect, useRef, type RefObject } from "react";
import { SplatMesh, RgbaArray } from "@sparkjsdev/spark";
import { useSegmentStore } from "../stores/segmentStore.ts";

/**
 * Uses splatRgba (GPU per-splat RGBA texture) to highlight splats based on
 * selection and hover state.
 *
 * Three visual states:
 * - Selected:  (255, 255, 255, 255) — full brightness
 * - Hovered:   (180, 190, 255, 220) — slight blue tint
 * - Dimmed:    (80, 80, 80, 160) — unselected/unhovered when something is active
 * - No selection and no hover: mesh.splatRgba = null → original colors
 */
export function useSplatHighlight(
  splatMeshRef: RefObject<SplatMesh | null>,
): void {
  const selectedSegmentIds = useSegmentStore((s) => s.selectedSegmentIds);
  const hoveredSegmentId = useSegmentStore((s) => s.hoveredSegmentId);
  const segments = useSegmentStore((s) => s.segments);
  const segmentIndexMap = useSegmentStore((s) => s.segmentIndexMap);

  // Track the RgbaArray we created so we can dispose it
  const rgbaRef = useRef<RgbaArray | null>(null);

  useEffect(() => {
    const mesh = splatMeshRef.current;
    if (!mesh) return;

    const hasSelection = selectedSegmentIds.length > 0;
    const hasHover = hoveredSegmentId !== null;

    // If nothing active, or index map doesn't match mesh, clear highlight
    if (
      (!hasSelection && !hasHover) ||
      !segmentIndexMap ||
      segmentIndexMap.length !== mesh.numSplats
    ) {
      if (rgbaRef.current) {
        mesh.splatRgba = null;
        rgbaRef.current.dispose();
        rgbaRef.current = null;
      }
      return;
    }

    // Build sets of selected and hovered segment values (1-based indices into segments array)
    const selectedValues = new Set<number>();
    for (const segId of selectedSegmentIds) {
      const idx = segments.findIndex((s) => s.id === segId);
      if (idx >= 0) selectedValues.add(idx + 1); // 1-based like the index map
    }

    let hoveredValue = -1;
    if (hoveredSegmentId !== null) {
      const idx = segments.findIndex((s) => s.id === hoveredSegmentId);
      if (idx >= 0) hoveredValue = idx + 1;
    }

    const numSplats = segmentIndexMap.length;
    const rgba = new Uint8Array(numSplats * 4);

    for (let i = 0; i < numSplats; i++) {
      const segValue = segmentIndexMap[i];
      const off = i * 4;

      if (selectedValues.has(segValue)) {
        // Selected: full brightness
        rgba[off] = 255;
        rgba[off + 1] = 255;
        rgba[off + 2] = 255;
        rgba[off + 3] = 255;
      } else if (segValue === hoveredValue) {
        // Hovered (not selected): blue tint
        rgba[off] = 180;
        rgba[off + 1] = 190;
        rgba[off + 2] = 255;
        rgba[off + 3] = 220;
      } else {
        // Dimmed
        rgba[off] = 80;
        rgba[off + 1] = 80;
        rgba[off + 2] = 80;
        rgba[off + 3] = 160;
      }
    }

    // Dispose old RgbaArray if any
    if (rgbaRef.current) {
      rgbaRef.current.dispose();
    }

    const rgbaArray = new RgbaArray({ array: rgba, count: numSplats });
    rgbaRef.current = rgbaArray;
    mesh.splatRgba = rgbaArray;
  }, [selectedSegmentIds, hoveredSegmentId, segments, segmentIndexMap, splatMeshRef]);

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
