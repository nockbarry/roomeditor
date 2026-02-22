import { useCompositionStore } from "../../stores/compositionStore.ts";
import { Eye, EyeOff, Trash2, Import } from "lucide-react";

interface CompositionPanelProps {
  projectId: string;
}

export function CompositionPanel({ projectId }: CompositionPanelProps) {
  const importedObjects = useCompositionStore((s) => s.importedObjects);
  const selectedImportId = useCompositionStore((s) => s.selectedImportId);
  const removeImported = useCompositionStore((s) => s.removeImported);
  const updateImported = useCompositionStore((s) => s.updateImported);
  const selectImported = useCompositionStore((s) => s.selectImported);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-gray-800">
        <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">
          Imported Objects
        </div>
        {importedObjects.length === 0 && (
          <p className="text-[10px] text-gray-600">
            Import segments from other projects to compose scenes.
          </p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        {importedObjects.map((obj) => (
          <div
            key={obj.id}
            className={`border-b border-gray-800/50 ${
              selectedImportId === obj.id ? "bg-violet-900/20" : ""
            }`}
          >
            <div className="flex items-center px-3 py-2">
              <button
                onClick={() => selectImported(obj.id)}
                className="flex-1 flex items-center gap-2 text-left min-w-0"
              >
                <Import className="w-3 h-3 text-blue-400 shrink-0" />
                <span className="text-xs text-gray-300 truncate">{obj.label}</span>
                <span className="text-[9px] text-gray-600 shrink-0">
                  {obj.sourceProjectId.slice(0, 8)}...
                </span>
              </button>
              <button
                onClick={() => updateImported(obj.id, { visible: !obj.visible })}
                className="p-1 hover:bg-gray-800 rounded"
              >
                {obj.visible ? (
                  <Eye className="w-3 h-3 text-gray-500" />
                ) : (
                  <EyeOff className="w-3 h-3 text-gray-700" />
                )}
              </button>
              <button
                onClick={() => removeImported(obj.id)}
                className="p-1 hover:bg-gray-800 rounded text-gray-600 hover:text-red-400"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>

            {/* Transform controls for selected imported object */}
            {selectedImportId === obj.id && (
              <div className="px-3 pb-2 space-y-1">
                <div className="text-[10px] text-gray-600 mb-1">Position</div>
                <div className="grid grid-cols-3 gap-1">
                  {(["X", "Y", "Z"] as const).map((axis, i) => (
                    <div key={axis} className="flex gap-0.5">
                      <button
                        onClick={() => {
                          const pos = [...obj.position] as [number, number, number];
                          pos[i] -= 0.1;
                          updateImported(obj.id, { position: pos });
                        }}
                        className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                      >
                        -{axis}
                      </button>
                      <button
                        onClick={() => {
                          const pos = [...obj.position] as [number, number, number];
                          pos[i] += 0.1;
                          updateImported(obj.id, { position: pos });
                        }}
                        className="flex-1 bg-gray-800 hover:bg-gray-700 text-[10px] py-1 rounded"
                      >
                        +{axis}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="px-3 py-2 border-t border-gray-800">
        <p className="text-[9px] text-gray-600">
          Tip: Export a segment's PLY from another project, then load it here.
        </p>
      </div>
    </div>
  );
}
