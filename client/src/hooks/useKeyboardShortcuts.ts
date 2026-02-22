import { useEffect } from "react";
import { useEditorStore } from "../stores/editorStore.ts";
import { useSegmentStore } from "../stores/segmentStore.ts";
import { useAnySplatStore } from "../stores/anysplatStore.ts";

interface KeyboardShortcutOptions {
  projectId: string;
  mode: "build" | "edit";
  onToggleMode: () => void;
}

export function useKeyboardShortcuts({
  projectId,
  mode,
  onToggleMode,
}: KeyboardShortcutOptions) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Skip when focus is on form elements
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      // Tab — toggle Build/Edit mode
      if (e.key === "Tab") {
        e.preventDefault();
        onToggleMode();
        return;
      }

      // Only handle edit-mode shortcuts below
      if (mode !== "edit") return;

      const setToolMode = useEditorStore.getState().setToolMode;

      // Tool mode shortcuts
      if (e.key === "q" || e.key === "Q") {
        e.preventDefault();
        setToolMode("select");
        return;
      }
      if (e.key === "g" || e.key === "G") {
        e.preventDefault();
        setToolMode("translate");
        return;
      }
      if (e.key === "r" || e.key === "R") {
        if (e.ctrlKey || e.metaKey) return; // Don't capture browser refresh
        e.preventDefault();
        setToolMode("rotate");
        return;
      }
      if (e.key === "s" || e.key === "S") {
        if (e.ctrlKey || e.metaKey) return; // Don't capture browser save
        e.preventDefault();
        setToolMode("scale");
        return;
      }

      // Ctrl+Z — undo
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        const segStore = useSegmentStore.getState();
        if (segStore.undoCount > 0) {
          segStore.undo(projectId).then(() => {
            useAnySplatStore.setState((s) => ({
              plyVersion: s.plyVersion + 1,
            }));
          });
        }
        return;
      }

      // Delete/Backspace — delete selected segments
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        const segStore = useSegmentStore.getState();
        const ids = segStore.selectedSegmentIds;
        if (ids.length > 0) {
          // Delete the first selected segment
          segStore.deleteSegment(projectId, ids[0]).then(() => {
            useAnySplatStore.setState((s) => ({
              plyVersion: s.plyVersion + 1,
            }));
          });
        }
        return;
      }

      // Escape — deselect all
      if (e.key === "Escape") {
        e.preventDefault();
        useSegmentStore.getState().selectSegment(null);
        return;
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [projectId, mode, onToggleMode]);
}
