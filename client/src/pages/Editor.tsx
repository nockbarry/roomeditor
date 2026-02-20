import { useEffect, useCallback, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useProjectStore } from "../stores/projectStore.ts";
import { extractFrames } from "../api/anysplat.ts";
import { Viewport } from "../components/editor/Viewport.tsx";
import { Toolbar } from "../components/editor/Toolbar.tsx";
import { ObjectListPanel } from "../components/panels/ObjectListPanel.tsx";
import { PipelineProgress } from "../components/upload/PipelineProgress.tsx";
import { VideoUploader } from "../components/upload/VideoUploader.tsx";
import { ReconstructionSettings } from "../components/upload/ReconstructionSettings.tsx";
import { AnySplatStudio } from "./AnySplatStudio.tsx";
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  Upload,
  CheckCircle,
  Box,
  Clock,
  FlaskConical,
} from "lucide-react";

type ViewMode = "studio" | "training";

function formatGaussianCount(n: number | null): string {
  if (!n) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffSec < 60) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}

function formatFullTimestamp(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function Editor() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const {
    currentProject,
    loadProject,
    jobProgress,
    trainingPreview,
    latestSnapshot,
    startReconstruction,
  } = useProjectStore();

  const [viewMode, setViewMode] = useState<ViewMode>("studio");

  const handleEnterStudio = useCallback(async () => {
    if (!currentProject) return;
    setViewMode("studio");
    await extractFrames(currentProject.id);
    await loadProject(currentProject.id);
  }, [currentProject, loadProject]);

  useEffect(() => {
    if (projectId) {
      loadProject(projectId);
    }
  }, [projectId, loadProject]);

  if (!currentProject) {
    return (
      <div className="h-full bg-gray-950 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-gray-500" />
      </div>
    );
  }

  const isAnySplat = currentProject.reconstruction_mode === "anysplat";

  const renderContent = () => {
    // AnySplat projects: always use AnySplatStudio (has its own Build/Edit toggle)
    if (isAnySplat) {
      return <AnySplatStudio projectId={currentProject.id} />;
    }

    switch (currentProject.status) {
      case "created":
        return (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center max-w-md">
              <Upload className="w-12 h-12 mx-auto mb-4 text-gray-500" />
              <h2 className="text-lg font-medium mb-2">Upload Media</h2>
              <p className="text-sm text-gray-400 mb-6">
                Upload a video or photos of the room. For video, walk slowly
                around the room in a loop. For photos, capture from multiple angles.
              </p>
              <VideoUploader
                projectId={currentProject.id}
                onUploaded={() => loadProject(currentProject.id)}
              />
            </div>
          </div>
        );

      case "uploaded":
        if (viewMode === "training") {
          return (
            <div className="flex-1 flex items-center justify-center">
              <ReconstructionSettings
                onStart={(config) => startReconstruction(currentProject.id, config)}
                onAnySplat={handleEnterStudio}
              />
            </div>
          );
        }
        return <AnySplatStudio projectId={currentProject.id} />;

      case "processing":
        return (
          <div className="flex-1 flex items-center justify-center">
            <PipelineProgress progress={jobProgress} preview={trainingPreview} snapshot={latestSnapshot} />
          </div>
        );

      case "ready":
        return (
          <>
            <Toolbar />
            <div className="flex-1 flex overflow-hidden">
              <Viewport projectId={currentProject.id} />
              <ObjectListPanel />
            </div>
          </>
        );

      case "error":
        return (
          <div className="flex-1 flex items-center justify-center">
            <div className="max-w-lg mx-auto">
              <div className="text-center mb-6">
                <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-400" />
                <h2 className="text-lg font-medium mb-2">Reconstruction Failed</h2>
                <p className="text-sm text-red-300">
                  {currentProject.error_message || "An unknown error occurred."}
                </p>
              </div>
              <ReconstructionSettings
                onStart={(config) => startReconstruction(currentProject.id, config)}
                onAnySplat={handleEnterStudio}
              />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="h-full bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-4 py-2 flex items-center gap-3 shrink-0">
        <button
          onClick={() => navigate("/")}
          className="text-gray-400 hover:text-white p-1"
          title="Back to projects"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-sm font-medium">{currentProject.name}</h1>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            currentProject.status === "ready"
              ? "bg-green-900/30 text-green-400"
              : currentProject.status === "processing"
                ? "bg-yellow-900/30 text-yellow-400"
                : currentProject.status === "error"
                  ? "bg-red-900/30 text-red-400"
                  : currentProject.status === "uploaded"
                    ? "bg-blue-900/30 text-blue-400"
                    : "bg-gray-800 text-gray-500"
          }`}
        >
          {currentProject.status === "ready" && <CheckCircle className="w-3 h-3 inline mr-1" />}
          {currentProject.status === "processing" && <Loader2 className="w-3 h-3 inline mr-1 animate-spin" />}
          {currentProject.status === "error" && <AlertCircle className="w-3 h-3 inline mr-1" />}
          {currentProject.status === "created" ? "Awaiting upload" :
           currentProject.status === "uploaded" ? "Ready to reconstruct" :
           currentProject.status === "processing" ? "Processing" :
           currentProject.status === "ready" ? "Ready" :
           currentProject.status === "error" ? "Failed" : currentProject.status}
        </span>

        {/* Training pipeline link for uploaded non-anysplat projects */}
        {!isAnySplat && currentProject.status === "uploaded" && viewMode !== "training" && (
          <button
            onClick={() => setViewMode("training")}
            className="flex items-center gap-1 text-[10px] text-gray-600 hover:text-gray-400 transition-colors"
          >
            <FlaskConical className="w-3 h-3" />
            Training pipeline
          </button>
        )}

        {currentProject.gaussian_count && (
          <span className="text-xs text-gray-400 flex items-center gap-1">
            <Box className="w-3 h-3" />
            {formatGaussianCount(currentProject.gaussian_count)} Gaussians
          </span>
        )}

        <div className="flex-1" />
        <span
          className="text-xs text-gray-500 flex items-center gap-1"
          title={formatFullTimestamp(currentProject.created_at)}
        >
          <Clock className="w-3 h-3" />
          Created {formatRelativeTime(currentProject.created_at)}
        </span>
      </header>

      {renderContent()}
    </div>
  );
}
