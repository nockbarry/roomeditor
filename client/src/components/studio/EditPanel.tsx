import { useState } from "react";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import { SplatBrowser } from "./SplatBrowser.tsx";
import { SegmentPanel } from "./SegmentPanel.tsx";
import { FolderOpen, Hammer, AlertTriangle } from "lucide-react";

interface EditPanelProps {
  projectId: string;
  onSwitchToBuild: () => void;
}

export function EditPanel({ projectId, onSwitchToBuild }: EditPanelProps) {
  const plyUrl = useAnySplatStore((s) => s.plyUrl);
  const [showSplatBrowser, setShowSplatBrowser] = useState(false);

  const isForeignSplat = plyUrl && !plyUrl.includes(`/data/${projectId}/`);

  return (
    <div className="flex flex-col h-full">
      {/* Header: Load Splat + Build link */}
      <div className="px-3 py-2 border-b border-gray-800 flex items-center gap-2">
        <div className="relative flex-1">
          <button
            onClick={() => setShowSplatBrowser(!showSplatBrowser)}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors px-2 py-1.5 rounded hover:bg-gray-800"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            Load Splat
          </button>
          {showSplatBrowser && (
            <SplatBrowser
              currentProjectId={projectId}
              onClose={() => setShowSplatBrowser(false)}
            />
          )}
        </div>
        <button
          onClick={onSwitchToBuild}
          className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-emerald-400 transition-colors"
        >
          <Hammer className="w-3 h-3" />
          Build
        </button>
      </div>

      {/* Foreign splat banner */}
      {isForeignSplat && (
        <div className="px-3 py-2 border-b border-gray-800 bg-amber-900/10">
          <div className="flex items-start gap-2 text-[11px] text-amber-400/80">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>
              Viewing a splat from another project. Segment controls are
              unavailable.
            </span>
          </div>
        </div>
      )}

      {/* Segment panel (hidden for foreign splats) */}
      {isForeignSplat ? (
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-xs text-gray-600 text-center">
            Load this project's own splat to use editing tools.
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-hidden">
          <SegmentPanel projectId={projectId} />
        </div>
      )}
    </div>
  );
}
