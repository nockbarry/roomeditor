import { useState, useRef } from "react";
import { useProjectStore } from "../../stores/projectStore.ts";
import { Upload, Film, Image, Loader2, X, FileVideo, FileImage } from "lucide-react";

interface VideoUploaderProps {
  projectId: string;
  onUploaded: () => void;
}

const VALID_VIDEO_TYPES = [
  "video/mp4",
  "video/quicktime",
  "video/webm",
  "video/x-msvideo",
];
const VALID_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/tiff",
  "image/bmp",
];
const ALL_VALID_TYPES = [...VALID_VIDEO_TYPES, ...VALID_IMAGE_TYPES];
const ACCEPT_STRING = "video/mp4,video/quicktime,video/webm,image/jpeg,image/png,image/webp";

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function VideoUploader({ projectId, onUploaded }: VideoUploaderProps) {
  const uploadFiles = useProjectStore((s) => s.uploadFiles);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (newFiles: FileList | File[]) => {
    const valid: File[] = [];
    for (const file of Array.from(newFiles)) {
      if (!ALL_VALID_TYPES.includes(file.type)) {
        setError(`Unsupported: ${file.name}. Use MP4/MOV/WEBM or JPG/PNG.`);
        return;
      }
      valid.push(file);
    }
    setError(null);
    setSelectedFiles((prev) => [...prev, ...valid]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      await uploadFiles(projectId, selectedFiles);
      onUploaded();
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = ""; // reset so same file can be re-selected
  };

  const totalSize = selectedFiles.reduce((acc, f) => acc + f.size, 0);
  const videoCount = selectedFiles.filter((f) =>
    VALID_VIDEO_TYPES.includes(f.type)
  ).length;
  const imageCount = selectedFiles.length - videoCount;

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
          dragging
            ? "border-blue-500 bg-blue-500/10"
            : "border-gray-700 hover:border-gray-600"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_STRING}
          multiple
          onChange={handleChange}
          className="hidden"
        />
        <Upload className="w-6 h-6 mx-auto mb-2 text-gray-500" />
        <p className="text-sm text-gray-400">
          Drop videos and/or photos here, or click to browse
        </p>
        <p className="text-xs text-gray-600 mt-1">
          Multiple files supported — combine videos + photos of the same space
        </p>
      </div>

      {/* Selected files list */}
      {selectedFiles.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-800 flex justify-between items-center">
            <span className="text-xs text-gray-400">
              {videoCount > 0 && `${videoCount} video${videoCount > 1 ? "s" : ""}`}
              {videoCount > 0 && imageCount > 0 && ", "}
              {imageCount > 0 && `${imageCount} image${imageCount > 1 ? "s" : ""}`}
              {" "} ({formatSize(totalSize)})
            </span>
            <button
              onClick={() => setSelectedFiles([])}
              className="text-xs text-gray-500 hover:text-white"
            >
              Clear all
            </button>
          </div>
          <div className="max-h-40 overflow-y-auto">
            {selectedFiles.map((file, i) => (
              <div
                key={`${file.name}-${i}`}
                className="px-3 py-1.5 flex items-center gap-2 text-xs border-b border-gray-900 last:border-0"
              >
                {VALID_VIDEO_TYPES.includes(file.type) ? (
                  <FileVideo className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                ) : (
                  <FileImage className="w-3.5 h-3.5 text-green-400 shrink-0" />
                )}
                <span className="flex-1 truncate text-gray-300">{file.name}</span>
                <span className="text-gray-600 shrink-0">
                  {formatSize(file.size)}
                </span>
                {!uploading && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(i);
                    }}
                    className="text-gray-600 hover:text-red-400"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload button */}
      {selectedFiles.length > 0 && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          {uploading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Uploading...
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" />
              Upload {selectedFiles.length} file{selectedFiles.length > 1 ? "s" : ""}
            </>
          )}
        </button>
      )}

      {error && (
        <p className="text-xs text-red-400 mt-1">{error}</p>
      )}
    </div>
  );
}
