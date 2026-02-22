import { useEffect, useState, useRef, useCallback } from "react";
import { useAnySplatStore } from "../stores/anysplatStore.ts";
import { useProjectStore } from "../stores/projectStore.ts";
import { useSegmentStore } from "../stores/segmentStore.ts";
import { CompactUploader } from "../components/studio/CompactUploader.tsx";
import { FrameGallery } from "../components/studio/FrameGallery.tsx";
import { SplatViewer } from "../components/studio/SplatViewer.tsx";
import { SettingsPanel } from "../components/studio/SettingsPanel.tsx";
import { EditPanel } from "../components/studio/EditPanel.tsx";
import { ToastContainer } from "../components/ui/ToastContainer.tsx";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts.ts";
import { useCollabSync } from "../hooks/useCollabSync.ts";
import { PresenceIndicator } from "../components/studio/PresenceIndicator.tsx";
import { Hammer, Pencil } from "lucide-react";

interface AnySplatStudioProps {
  projectId: string;
}

type StudioMode = "build" | "edit";

export function AnySplatStudio({ projectId }: AnySplatStudioProps) {
  const currentProject = useProjectStore((s) => s.currentProject);
  const fetchFrames = useAnySplatStore((s) => s.fetchFrames);
  const extractFrames = useAnySplatStore((s) => s.extractFrames);
  const reset = useAnySplatStore((s) => s.reset);
  const resetSegments = useSegmentStore((s) => s.reset);
  const fetchSegments = useSegmentStore((s) => s.fetchSegments);
  const plyUrl = useAnySplatStore((s) => s.plyUrl);
  const isRunning = useAnySplatStore((s) => s.isRunning);

  // Initialize to "edit" if project already has a splat
  const [mode, setMode] = useState<StudioMode>(() =>
    currentProject?.status === "ready" ? "edit" : "build"
  );

  const toggleMode = useCallback(() => {
    setMode((m) => (m === "build" ? "edit" : "build"));
  }, []);

  useKeyboardShortcuts({ projectId, mode, onToggleMode: toggleMode });
  useCollabSync(projectId);

  // Auto-switch to Edit when rebuild completes (isRunning: true → false, plyUrl exists)
  const wasRunning = useRef(isRunning);
  useEffect(() => {
    if (wasRunning.current && !isRunning && plyUrl) {
      setMode("edit");
    }
    wasRunning.current = isRunning;
  }, [isRunning, plyUrl]);

  useEffect(() => {
    // On mount, try to load existing frames
    fetchFrames(projectId).then(() => {
      const state = useAnySplatStore.getState();
      // If no frames but project has sources, auto-extract
      if (state.frames.length === 0 && currentProject?.video_filename) {
        extractFrames(projectId);
      }
    });

    // Load existing segments
    fetchSegments(projectId);

    // If project already has a PLY, set the URL so viewer shows it
    if (currentProject?.status === "ready") {
      useAnySplatStore.setState({
        plyUrl: `/data/${projectId}/scene.ply`,
        plyVersion: 1,
      });
      // Fetch quality stats for existing splat
      useAnySplatStore.getState().fetchQualityStats(projectId);
    }

    return () => {
      reset();
      resetSegments();
    };
  }, [projectId]);

  return (
    <div className="flex-1 flex overflow-hidden">
      <ToastContainer />
      {/* Left panel — Build mode only */}
      {mode === "build" && (
        <div className="w-[280px] shrink-0 border-r border-gray-800 flex flex-col bg-gray-950">
          <CompactUploader projectId={projectId} />
          <FrameGallery projectId={projectId} />
        </div>
      )}

      {/* Center — Splat Viewer */}
      <div className="flex-1 min-w-0 flex flex-col p-2 relative">
        <SplatViewer projectId={projectId} />

        {/* Floating Build/Edit toggle + presence */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2">
          <div className="flex bg-gray-900/80 backdrop-blur-sm rounded-lg p-0.5 border border-gray-700/50 shadow-lg">
            <button
              onClick={() => setMode("build")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                mode === "build"
                  ? "bg-emerald-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <Hammer className="w-3.5 h-3.5" />
              Build
            </button>
            <button
              onClick={() => setMode("edit")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                mode === "edit"
                  ? "bg-violet-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              <Pencil className="w-3.5 h-3.5" />
              Edit
            </button>
          </div>
          <PresenceIndicator />
        </div>
      </div>

      {/* Right panel — mode-dependent */}
      <div className="w-[280px] shrink-0 border-l border-gray-800 bg-gray-950 flex flex-col">
        {mode === "build" ? (
          <SettingsPanel projectId={projectId} />
        ) : (
          <EditPanel projectId={projectId} onSwitchToBuild={() => setMode("build")} />
        )}
      </div>
    </div>
  );
}
