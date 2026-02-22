import { useEffect, useRef } from "react";
import { useCollabStore } from "../stores/collabStore.ts";
import { useSegmentStore } from "../stores/segmentStore.ts";
import { useEditorStore } from "../stores/editorStore.ts";

/**
 * Hook that connects to the collaboration WebSocket and syncs
 * local state changes (cursor, selection, transforms) to the server.
 */
export function useCollabSync(projectId: string) {
  const connect = useCollabStore((s) => s.connect);
  const disconnect = useCollabStore((s) => s.disconnect);
  const sendSelectSegment = useCollabStore((s) => s.sendSelectSegment);
  const sendTransformStart = useCollabStore((s) => s.sendTransformStart);
  const sendTransformEnd = useCollabStore((s) => s.sendTransformEnd);

  // Connect/disconnect on mount/unmount
  useEffect(() => {
    connect(projectId);
    return () => disconnect();
  }, [projectId, connect, disconnect]);

  // Sync segment selection to collaborators
  const prevSelectedRef = useRef<number[]>([]);
  useEffect(() => {
    return useSegmentStore.subscribe((state) => {
      const ids = state.selectedSegmentIds;
      const prev = prevSelectedRef.current;
      if (ids.length !== prev.length || ids.some((id, i) => id !== prev[i])) {
        prevSelectedRef.current = ids;
        // Send the primary selection (first selected, or null)
        sendSelectSegment(ids.length > 0 ? ids[0] : null);
      }
    });
  }, [sendSelectSegment]);

  // Sync transform start/end via editor store tool mode
  const prevToolRef = useRef<string>("select");
  useEffect(() => {
    return useEditorStore.subscribe((state) => {
      const tool = state.toolMode;
      const prev = prevToolRef.current;
      if (tool !== prev) {
        const selectedIds = useSegmentStore.getState().selectedSegmentIds;
        if (selectedIds.length > 0) {
          const segId = selectedIds[0];
          // Starting a transform tool
          if (prev === "select" && tool !== "select") {
            sendTransformStart(segId);
          }
          // Returning to select (transform ended)
          if (prev !== "select" && tool === "select") {
            sendTransformEnd(segId);
          }
        }
        prevToolRef.current = tool;
      }
    });
  }, [sendTransformStart, sendTransformEnd]);
}
