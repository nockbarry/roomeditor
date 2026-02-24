import { useEffect, useState } from "react";
import { listProjects } from "../../api/projects.ts";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import type { Project } from "../../types/api.ts";
import { Box, Loader2, X } from "lucide-react";

interface SplatBrowserProps {
  currentProjectId: string;
  onClose: () => void;
}

function formatCount(n: number | null): string {
  if (!n) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function formatSize(bytes: number | null): string {
  if (!bytes) return "";
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}G`;
  if (bytes >= 1024 * 1024) return `${Math.round(bytes / (1024 * 1024))}M`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)}K`;
  return `${bytes}B`;
}

export function SplatBrowser({ currentProjectId, onClose }: SplatBrowserProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listProjects()
      .then((all) => {
        // Show projects that have a scene.ply (status "ready" or anysplat with a ply)
        const withSplats = all.filter(
          (p) => p.status === "ready" || (p.gaussian_count && p.gaussian_count > 0)
        );
        setProjects(withSplats);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleLoad = (project: Project) => {
    const plyUrl = `/data/${project.id}/scene.ply`;
    useAnySplatStore.setState((s) => ({
      plyUrl,
      plyVersion: s.plyVersion + 1,
    }));
    onClose();
  };

  return (
    <div className="absolute top-full left-0 mt-1 w-72 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-medium text-gray-300">Load Splat from Project</span>
        <button onClick={onClose} className="text-gray-500 hover:text-white">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="max-h-64 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-6 text-xs text-gray-500">
            No projects with splats found
          </div>
        ) : (
          projects.map((p) => (
            <button
              key={p.id}
              onClick={() => handleLoad(p)}
              className={`w-full text-left px-3 py-2 hover:bg-gray-800 transition-colors border-b border-gray-800/50 last:border-b-0 ${
                p.id === currentProjectId ? "bg-gray-800/50" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-200 truncate flex-1">{p.name}</span>
                {p.id === currentProjectId && (
                  <span className="text-[9px] text-emerald-400 ml-2 shrink-0">current</span>
                )}
              </div>
              <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                <span className="text-[10px] text-gray-500 flex items-center gap-0.5">
                  <Box className="w-2.5 h-2.5" />
                  {formatCount(p.gaussian_count)}
                </span>
                {p.spz_size_bytes && (
                  <span className="text-[9px] text-emerald-400/80">
                    SPZ {formatSize(p.spz_size_bytes)}
                  </span>
                )}
                {p.ply_size_bytes && (
                  <span className="text-[9px] text-gray-500">
                    PLY {formatSize(p.ply_size_bytes)}
                  </span>
                )}
                <span className={`text-[9px] px-1.5 py-0.5 rounded ${
                  p.reconstruction_mode === "anysplat"
                    ? "bg-emerald-900/30 text-emerald-400"
                    : "bg-blue-900/30 text-blue-400"
                }`}>
                  {p.reconstruction_mode ?? "training"}
                </span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
