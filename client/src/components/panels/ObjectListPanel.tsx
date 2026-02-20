import { useProjectStore } from "../../stores/projectStore.ts";
import { useEditorStore } from "../../stores/editorStore.ts";
import { Eye, EyeOff, Lock, Unlock, Box } from "lucide-react";

export function ObjectListPanel() {
  const objects = useProjectStore((s) => s.objects);
  const selectedObjectId = useEditorStore((s) => s.selectedObjectId);
  const selectObject = useEditorStore((s) => s.selectObject);

  if (objects.length === 0) {
    return (
      <div className="w-64 border-l border-gray-800 bg-gray-950 p-4 shrink-0">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Objects</h2>
        <p className="text-xs text-gray-600">
          No objects detected. Segmentation will identify objects in Phase 2.
        </p>
      </div>
    );
  }

  return (
    <div className="w-64 border-l border-gray-800 bg-gray-950 shrink-0 flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-800">
        <h2 className="text-sm font-medium text-gray-400">
          Objects ({objects.length})
        </h2>
      </div>
      <div className="flex-1 overflow-y-auto">
        {objects.map((obj) => (
          <div
            key={obj.id}
            onClick={() => selectObject(obj.id)}
            className={`px-4 py-2 flex items-center gap-2 cursor-pointer border-b border-gray-900 transition-colors ${
              selectedObjectId === obj.id
                ? "bg-blue-900/30 border-l-2 border-l-blue-500"
                : "hover:bg-gray-900"
            }`}
          >
            <Box className="w-3.5 h-3.5 text-gray-500 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium truncate">{obj.label}</div>
              <div className="text-[10px] text-gray-600">
                {obj.gaussian_count.toLocaleString()} splats
              </div>
            </div>
            <div className="flex items-center gap-1">
              {obj.visible ? (
                <Eye className="w-3 h-3 text-gray-500" />
              ) : (
                <EyeOff className="w-3 h-3 text-gray-600" />
              )}
              {obj.locked && <Lock className="w-3 h-3 text-gray-500" />}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
