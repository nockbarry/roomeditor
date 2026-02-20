import { useState, useRef } from "react";
import { useProjectStore } from "../../stores/projectStore.ts";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import { Upload, Loader2 } from "lucide-react";

interface CompactUploaderProps {
  projectId: string;
}

const ACCEPT_STRING = "video/mp4,video/quicktime,video/webm,image/jpeg,image/png,image/webp";
const ALL_VALID_TYPES = [
  "video/mp4", "video/quicktime", "video/webm", "video/x-msvideo",
  "image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp",
];

export function CompactUploader({ projectId }: CompactUploaderProps) {
  const uploadFiles = useProjectStore((s) => s.uploadFiles);
  const extractFrames = useAnySplatStore((s) => s.extractFrames);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    for (const file of files) {
      if (!ALL_VALID_TYPES.includes(file.type)) {
        setError(`Unsupported: ${file.name}`);
        return;
      }
    }

    setError(null);
    setUploading(true);
    try {
      await uploadFiles(projectId, files);
      await extractFrames(projectId);
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="px-3 py-2">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`border border-dashed rounded-lg p-3 text-center cursor-pointer transition-colors ${
          dragging
            ? "border-emerald-500 bg-emerald-500/10"
            : "border-gray-700 hover:border-gray-600"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_STRING}
          multiple
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
          className="hidden"
        />
        {uploading ? (
          <Loader2 className="w-4 h-4 mx-auto animate-spin text-gray-400" />
        ) : (
          <>
            <Upload className="w-4 h-4 mx-auto mb-1 text-gray-500" />
            <p className="text-xs text-gray-500">Add media</p>
          </>
        )}
      </div>
      {error && <p className="text-[10px] text-red-400 mt-1">{error}</p>}
    </div>
  );
}
