import { useState } from "react";
import { useEditorStore } from "../../stores/editorStore.ts";
import { useProjectStore } from "../../stores/projectStore.ts";
import {
  MousePointer2,
  Move,
  RotateCcw,
  Maximize2,
  Grid3x3,
  Camera,
  Eye,
  Box,
  Loader2,
} from "lucide-react";
import { extractMesh } from "../../api/projects.ts";
import type { ToolMode } from "../../types/scene.ts";

const tools: { mode: ToolMode; icon: React.ReactNode; label: string; shortcut: string }[] = [
  { mode: "select", icon: <MousePointer2 className="w-4 h-4" />, label: "Select", shortcut: "Q" },
  { mode: "translate", icon: <Move className="w-4 h-4" />, label: "Move", shortcut: "G" },
  { mode: "rotate", icon: <RotateCcw className="w-4 h-4" />, label: "Rotate", shortcut: "R" },
  { mode: "scale", icon: <Maximize2 className="w-4 h-4" />, label: "Scale", shortcut: "S" },
];

export function Toolbar() {
  const { toolMode, setToolMode, cameraMode, setCameraMode, showGrid, toggleGrid } =
    useEditorStore();
  const currentProject = useProjectStore((s) => s.currentProject);
  const [meshExtracting, setMeshExtracting] = useState(false);

  const handleExtractMesh = async () => {
    if (!currentProject || meshExtracting) return;
    setMeshExtracting(true);
    try {
      const result = await extractMesh(currentProject.id);
      // Download the mesh file
      const link = document.createElement("a");
      link.href = `/api${result.mesh_url}`;
      link.download = "mesh.glb";
      link.click();
    } catch (e) {
      console.error("Mesh extraction failed:", e);
    } finally {
      setMeshExtracting(false);
    }
  };

  return (
    <div className="border-b border-gray-800 px-4 py-1.5 flex items-center gap-1 shrink-0">
      {/* Tool modes */}
      <div className="flex items-center gap-0.5 bg-gray-900 rounded-lg p-0.5">
        {tools.map((tool) => (
          <button
            key={tool.mode}
            onClick={() => setToolMode(tool.mode)}
            className={`px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5 transition-colors ${
              toolMode === tool.mode
                ? "bg-blue-600 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
            title={`${tool.label} (${tool.shortcut})`}
          >
            {tool.icon}
            <span className="hidden sm:inline">{tool.label}</span>
            <kbd className={`hidden sm:inline text-[10px] px-1 py-0.5 rounded ${
              toolMode === tool.mode
                ? "bg-blue-700 text-blue-200"
                : "bg-gray-800 text-gray-600"
            }`}>{tool.shortcut}</kbd>
          </button>
        ))}
      </div>

      <div className="w-px h-5 bg-gray-800 mx-2" />

      {/* Camera mode toggle */}
      <button
        onClick={() => setCameraMode(cameraMode === "orbit" ? "fps" : "orbit")}
        className="px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5 text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
        title={`Camera: ${cameraMode} (F)`}
      >
        {cameraMode === "orbit" ? (
          <Camera className="w-4 h-4" />
        ) : (
          <Eye className="w-4 h-4" />
        )}
        <span className="hidden sm:inline">
          {cameraMode === "orbit" ? "Orbit" : "FPS"}
        </span>
        <kbd className="hidden sm:inline text-[10px] px-1 py-0.5 rounded bg-gray-800 text-gray-600">F</kbd>
      </button>

      {/* Grid toggle */}
      <button
        onClick={toggleGrid}
        className={`px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5 transition-colors ${
          showGrid
            ? "text-white bg-gray-800"
            : "text-gray-400 hover:text-white hover:bg-gray-800"
        }`}
        title="Toggle Grid"
      >
        <Grid3x3 className="w-4 h-4" />
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Extract Mesh button */}
      <button
        onClick={handleExtractMesh}
        disabled={meshExtracting || !currentProject}
        className="px-3 py-1.5 rounded-md text-xs flex items-center gap-1.5 text-gray-400 hover:text-white hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Extract Mesh (GLB)"
      >
        {meshExtracting ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Box className="w-4 h-4" />
        )}
        <span className="hidden sm:inline">
          {meshExtracting ? "Extracting..." : "Extract Mesh"}
        </span>
      </button>
    </div>
  );
}
