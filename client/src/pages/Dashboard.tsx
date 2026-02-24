import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useProjectStore } from "../stores/projectStore.ts";
import { VideoUploader } from "../components/upload/VideoUploader.tsx";
import {
  Plus,
  Trash2,
  Box,
  Loader2,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Clock,
} from "lucide-react";
import type { Project } from "../types/api.ts";

const statusIcons: Record<string, React.ReactNode> = {
  created: <Box className="w-4 h-4 text-gray-400" />,
  uploaded: <Box className="w-4 h-4 text-blue-400" />,
  processing: <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />,
  ready: <CheckCircle className="w-4 h-4 text-green-400" />,
  error: <AlertCircle className="w-4 h-4 text-red-400" />,
};

const statusLabels: Record<string, string> = {
  created: "Awaiting upload",
  uploaded: "Ready to reconstruct",
  processing: "Processing",
  ready: "Ready",
  error: "Failed",
};

function formatGaussianCount(n: number | null): string {
  if (!n) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function formatFileSize(bytes: number | null): string {
  if (!bytes) return "";
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${Math.round(bytes / (1024 * 1024))} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
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

export function Dashboard() {
  const navigate = useNavigate();
  const {
    projects,
    loading,
    error,
    fetchProjects,
    createProject,
    deleteProject,
  } = useProjectStore();

  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectId, setNewProjectId] = useState<string | null>(null);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreate = async () => {
    if (!newProjectName.trim()) return;
    const project = await createProject(newProjectName.trim());
    setNewProjectName("");
    setNewProjectId(project.id);
  };

  const handleVideoUploaded = () => {
    if (newProjectId) {
      navigate(`/project/${newProjectId}`);
      setNewProjectId(null);
      setShowNewProject(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm("Delete this project and all its data?")) {
      await deleteProject(id);
    }
  };

  const readyCount = projects.filter((p) => p.status === "ready").length;

  return (
    <div className="h-full bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">Room Editor</h1>
          {projects.length > 0 && (
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">
              {readyCount} project{readyCount !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/benchmarks"
            className="flex items-center gap-2 text-gray-400 hover:text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors hover:bg-gray-800"
          >
            <BarChart3 className="w-4 h-4" />
            Benchmarks
          </Link>
          <button
            onClick={() => {
              setShowNewProject(true);
              setNewProjectId(null);
            }}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Project
          </button>
        </div>
      </header>

      <div className="p-6 max-w-4xl mx-auto flex-1 overflow-y-auto w-full">
        {/* New Project Flow */}
        {showNewProject && (
          <div className="mb-8 bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-lg font-medium mb-4">New Project</h2>

            {!newProjectId ? (
              <div className="flex gap-3">
                <input
                  type="text"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                  placeholder="Project name (e.g., Living Room)"
                  className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
                  autoFocus
                />
                <button
                  onClick={handleCreate}
                  disabled={!newProjectName.trim()}
                  className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowNewProject(false)}
                  className="text-gray-400 hover:text-white px-3 py-2 text-sm"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <div>
                <p className="text-sm text-gray-400 mb-4">
                  Upload a video or photos of the room. For video, walk slowly
                  around the room in a loop. For photos, capture from multiple angles.
                </p>
                <VideoUploader
                  projectId={newProjectId}
                  onUploaded={handleVideoUploaded}
                />
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-4 bg-red-900/30 text-red-300 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* Project List */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-gray-500" />
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-20 text-gray-500">
            <Box className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="mb-2">No projects yet.</p>
            <p className="text-sm">Create one to get started with 3D reconstruction.</p>
          </div>
        ) : (
          <div className="grid gap-2">
            {projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => navigate(`/project/${project.id}`)}
                onDelete={(e) => handleDelete(e, project.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProjectCard({
  project,
  onClick,
  onDelete,
}: {
  project: Project;
  onClick: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const gsCount = formatGaussianCount(project.gaussian_count);
  const plySize = formatFileSize(project.ply_size_bytes);
  const spzSize = formatFileSize(project.spz_size_bytes);

  return (
    <div
      onClick={onClick}
      className="bg-gray-900 hover:bg-gray-800 border border-gray-800 rounded-xl px-4 py-3 flex items-center justify-between cursor-pointer transition-colors group"
    >
      <div className="flex items-center gap-3 min-w-0 flex-1">
        {statusIcons[project.status]}
        <div className="min-w-0 flex-1">
          <div className="font-medium truncate">{project.name}</div>
          <div className="text-xs text-gray-500 mt-0.5 flex items-center gap-2 flex-wrap">
            {gsCount && (
              <span className="text-gray-400">{gsCount} Gaussians</span>
            )}
            {(plySize || spzSize) && (
              <>
                <span className="text-gray-700">|</span>
                {spzSize && (
                  <span className="text-emerald-400/80">SPZ {spzSize}</span>
                )}
                {plySize && (
                  <span className="text-gray-500">PLY {plySize}</span>
                )}
              </>
            )}
            {(gsCount || plySize || spzSize) && <span className="text-gray-700">|</span>}
            <span title={formatFullTimestamp(project.created_at)}>
              {formatRelativeTime(project.created_at)}
            </span>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3 shrink-0 ml-3">
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            project.status === "ready"
              ? "bg-green-900/30 text-green-400"
              : project.status === "processing"
                ? "bg-yellow-900/30 text-yellow-400"
                : project.status === "error"
                  ? "bg-red-900/30 text-red-400"
                  : "bg-gray-800 text-gray-500"
          }`}
        >
          {statusLabels[project.status] || project.status}
        </span>
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 p-1 transition-opacity"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
