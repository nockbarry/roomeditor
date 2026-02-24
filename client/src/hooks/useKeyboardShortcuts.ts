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

      const ctrl = e.ctrlKey || e.metaKey;

      // Tab — toggle Build/Edit mode
      if (e.key === "Tab") {
        e.preventDefault();
        onToggleMode();
        return;
      }

      // Ctrl+Z — undo (works in both modes)
      if (ctrl && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        const editorStore = useEditorStore.getState();
        if (editorStore.sceneLoaded && editorStore.undoCount > 0) {
          // In-memory undo, then save to disk + reload viewer
          editorStore.undo(projectId).then(async () => {
            await editorStore.saveScene(projectId);
            useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
          });
        } else {
          // Fallback to file-based undo
          const segStore = useSegmentStore.getState();
          if (segStore.undoCount > 0) {
            segStore.undo(projectId).then(() => {
              useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
            });
          }
        }
        return;
      }

      // Ctrl+Shift+Z — redo
      if (ctrl && e.key === "z" && e.shiftKey) {
        e.preventDefault();
        const editorStore = useEditorStore.getState();
        if (editorStore.sceneLoaded && editorStore.redoCount > 0) {
          editorStore.redo(projectId).then(async () => {
            await editorStore.saveScene(projectId);
            useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
          });
        }
        return;
      }

      // Ctrl+S — save scene
      if (ctrl && e.key === "s") {
        e.preventDefault();
        const editorStore = useEditorStore.getState();
        if (editorStore.sceneLoaded && editorStore.isDirty) {
          editorStore.saveScene(projectId);
        }
        return;
      }

      // V — toggle camera mode (works in both build/edit modes)
      if (!ctrl && !e.altKey && e.key.toLowerCase() === "v") {
        e.preventDefault();
        const { cameraMode, setCameraMode } = useEditorStore.getState();
        setCameraMode(cameraMode === "orbit" ? "fps" : "orbit");
        return;
      }

      // Only handle edit-mode shortcuts below
      if (mode !== "edit") return;

      const setToolMode = useEditorStore.getState().setToolMode;

      // Tool mode shortcuts (single keys, no modifiers)
      if (!ctrl && !e.altKey) {
        const lower = e.key.toLowerCase();

        if (lower === "q") { e.preventDefault(); setToolMode("select"); return; }
        if (lower === "g") { e.preventDefault(); setToolMode("translate"); return; }
        if (lower === "r") { e.preventDefault(); setToolMode("rotate"); return; }
        if (lower === "s") { e.preventDefault(); setToolMode("scale"); return; }
        if (lower === "b") { e.preventDefault(); setToolMode("brush"); return; }
        if (lower === "e") { e.preventDefault(); setToolMode("eraser"); return; }

        // [ / ] — brush size
        if (e.key === "[") {
          e.preventDefault();
          const { brushRadius } = useEditorStore.getState();
          useEditorStore.getState().setBrushRadius(brushRadius * 0.8);
          return;
        }
        if (e.key === "]") {
          e.preventDefault();
          const { brushRadius } = useEditorStore.getState();
          useEditorStore.getState().setBrushRadius(brushRadius * 1.25);
          return;
        }

        // X / Shift+X — crop mode toggle
        if (lower === "x") {
          const toolMode = useEditorStore.getState().toolMode;
          if (toolMode === "crop-box" || toolMode === "crop-sphere") {
            e.preventDefault();
            useEditorStore.setState({
              cropMode: e.shiftKey ? "delete-outside" : "delete-inside",
            });
            return;
          }
        }
      }

      // Delete/Backspace — delete brush selection or selected segments
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        const editorStore = useEditorStore.getState();
        if (editorStore.brushSelection.size > 0) {
          // Delete brush selection via in-memory editing
          const indices = Array.from(editorStore.brushSelection);
          editorStore.clearBrushSelection();
          editorStore.applyEdit(projectId, { type: "delete", indices }).then(() => {
            useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
          });
        } else {
          // Delete selected segment
          const segStore = useSegmentStore.getState();
          const ids = segStore.selectedSegmentIds;
          if (ids.length > 0) {
            segStore.deleteSegment(projectId, ids[0]).then(() => {
              useAnySplatStore.setState((s) => ({ plyVersion: s.plyVersion + 1 }));
            });
          }
        }
        return;
      }

      // Escape — deselect / cancel
      if (e.key === "Escape") {
        e.preventDefault();
        const editorStore = useEditorStore.getState();
        if (editorStore.brushSelection.size > 0) {
          editorStore.clearBrushSelection();
        } else {
          useSegmentStore.getState().selectSegment(null);
          setToolMode("select");
        }
        return;
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [projectId, mode, onToggleMode]);
}
