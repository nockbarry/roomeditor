import { Loader2, CheckCircle, AlertCircle, Circle } from "lucide-react";
import type { JobProgress, TrainingPreview, TrainingSnapshot } from "../../types/api.ts";
import { ProgressiveViewer } from "./ProgressiveViewer.tsx";

interface PipelineProgressProps {
  progress: JobProgress | null;
  preview: TrainingPreview | null;
  snapshot: TrainingSnapshot | null;
}

const PIPELINE_STEPS = [
  { key: "Extracting frames", label: "Extract Frames", threshold: 0.10 },
  { key: "Selecting keyframes", label: "Select Keyframes", threshold: 0.14 },
  { key: "Running COLMAP", altKey: "Running MASt3R", label: "Camera Poses", threshold: 0.45 },
  { key: "Training 3D Gaussians", label: "Train 3D Gaussians", threshold: 0.9 },
  { key: "Finalizing", label: "Finalize", threshold: 1.0 },
];

export function PipelineProgress({ progress, preview, snapshot }: PipelineProgressProps) {
  if (!progress) {
    return (
      <div className="text-center">
        <Loader2 className="w-8 h-8 mx-auto mb-3 text-blue-500 animate-spin" />
        <p className="text-sm text-gray-400">Connecting to pipeline...</p>
      </div>
    );
  }

  if (progress.status === "failed") {
    return (
      <div className="text-center max-w-md">
        <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-400" />
        <h2 className="text-lg font-medium mb-2">Pipeline Failed</h2>
        <p className="text-sm text-red-300">
          {progress.error_message || "An unknown error occurred."}
        </p>
      </div>
    );
  }

  const currentStepKey = progress.current_step || "";
  const isTraining = currentStepKey.includes("Training 3D Gaussians");
  const overallPercent = Math.round(progress.progress * 100);

  return (
    <div className="w-full max-w-2xl">
      <h2 className="text-lg font-medium mb-6 text-center">
        Building 3D Scene
      </h2>

      {/* Training preview — 3D snapshot viewer if available, fallback to 2D */}
      {isTraining && snapshot && (
        <div className="mb-6">
          <ProgressiveViewer snapshot={snapshot} />
        </div>
      )}
      {isTraining && !snapshot && preview && (
        <div className="mb-6">
          <div className="rounded-lg overflow-hidden bg-gray-900 border border-gray-800">
            <img
              src={`data:image/jpeg;base64,${preview.preview_base64}`}
              alt="Training preview"
              className="w-full h-auto"
            />
          </div>
          <div className="flex justify-between items-center mt-2 text-xs text-gray-400">
            <span>
              Step {preview.step.toLocaleString()} / {preview.total_steps.toLocaleString()}
            </span>
            <span>Loss: {preview.loss.toFixed(4)}</span>
            <span>{preview.n_gaussians.toLocaleString()} Gaussians</span>
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-800 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${Math.round((preview.step / preview.total_steps) * 100)}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Overall progress bar */}
      <div className="mb-6 max-w-sm mx-auto">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Overall Progress</span>
          <span>{overallPercent}%</span>
        </div>
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${overallPercent}%` }}
          />
        </div>
      </div>

      {/* Step list */}
      <div className="space-y-3 max-w-sm mx-auto">
        {PIPELINE_STEPS.map((step) => {
          const isActive = currentStepKey.includes(step.key) ||
            ("altKey" in step && step.altKey && currentStepKey.includes(step.altKey));
          const isDone = progress.progress >= step.threshold;
          const isPending = !isActive && !isDone;

          return (
            <div
              key={step.key}
              className={`flex items-center gap-3 text-sm ${
                isActive
                  ? "text-blue-400"
                  : isDone
                    ? "text-green-400"
                    : "text-gray-600"
              }`}
            >
              {isActive ? (
                <Loader2 className="w-4 h-4 animate-spin shrink-0" />
              ) : isDone ? (
                <CheckCircle className="w-4 h-4 shrink-0" />
              ) : (
                <Circle className="w-4 h-4 shrink-0" />
              )}
              <span>{step.label}</span>
            </div>
          );
        })}
      </div>

      {progress.status === "completed" && (
        <div className="mt-6 text-center">
          <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-400" />
          <p className="text-sm text-green-400">Reconstruction complete!</p>
        </div>
      )}
    </div>
  );
}
