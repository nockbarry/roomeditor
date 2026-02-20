import { useProjectStore } from "../../stores/projectStore.ts";
import { useEditorStore } from "../../stores/editorStore.ts";

export function PropertiesPanel() {
  const objects = useProjectStore((s) => s.objects);
  const selectedObjectId = useEditorStore((s) => s.selectedObjectId);

  const selectedObject = objects.find((o) => o.id === selectedObjectId);

  if (!selectedObject) {
    return (
      <div className="p-4">
        <p className="text-xs text-gray-600">Select an object to edit properties.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div>
        <label className="text-xs text-gray-500 block mb-1">Label</label>
        <div className="text-sm">{selectedObject.label}</div>
      </div>

      <div>
        <label className="text-xs text-gray-500 block mb-1">Position</label>
        <div className="grid grid-cols-3 gap-1">
          {["X", "Y", "Z"].map((axis, i) => (
            <div key={axis} className="text-xs">
              <span className="text-gray-600">{axis}: </span>
              <span>{selectedObject.translation[i].toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs text-gray-500 block mb-1">Scale</label>
        <div className="grid grid-cols-3 gap-1">
          {["X", "Y", "Z"].map((axis, i) => (
            <div key={axis} className="text-xs">
              <span className="text-gray-600">{axis}: </span>
              <span>{selectedObject.scale[i].toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="text-[10px] text-gray-600">
        Gaussians: {selectedObject.gaussian_count.toLocaleString()}
        <br />
        Source: {selectedObject.source}
      </div>
    </div>
  );
}
