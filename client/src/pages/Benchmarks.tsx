import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  BarChart3,
  Loader2,
  ChevronDown,
  ChevronRight,
  X,
  AlertCircle,
} from "lucide-react";
import { getBenchmarks } from "../api/benchmarks.ts";
import type { BenchmarkResult, BenchmarksResponse } from "../types/api.ts";

type MetricKey = "psnr" | "ssim" | "lpips";

const METRIC_LABELS: Record<MetricKey, string> = {
  psnr: "PSNR",
  ssim: "SSIM",
  lpips: "LPIPS",
};

// Higher is better for PSNR/SSIM, lower for LPIPS
const HIGHER_IS_BETTER: Record<MetricKey, boolean> = {
  psnr: true,
  ssim: true,
  lpips: false,
};

function getMetricValue(r: BenchmarkResult, metric: MetricKey): number | null {
  const m = r.metrics;
  if (!m) return null;
  if (metric === "psnr") return m.mean_psnr ?? null;
  if (metric === "ssim") return m.mean_ssim ?? null;
  if (metric === "lpips") return m.mean_lpips ?? null;
  return null;
}

function formatMetric(value: number | null, metric: MetricKey): string {
  if (value == null) return "--";
  if (metric === "psnr") return value.toFixed(2);
  return value.toFixed(4);
}

function formatTime(seconds: number | null): string {
  if (seconds == null || seconds <= 0) return "--";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

function formatCount(n: number | null): string {
  if (n == null || n <= 0) return "--";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

// Find best value per scene column
function findBestPerScene(
  rows: BenchmarkResult[],
  scenes: string[],
  metric: MetricKey
): Record<string, number> {
  const best: Record<string, number> = {};
  const hib = HIGHER_IS_BETTER[metric];
  for (const scene of scenes) {
    let bestVal: number | null = null;
    for (const r of rows) {
      if (r.scene !== scene) continue;
      const v = getMetricValue(r, metric);
      if (v == null) continue;
      if (bestVal == null || (hib ? v > bestVal : v < bestVal)) {
        bestVal = v;
      }
    }
    if (bestVal != null) best[scene] = bestVal;
  }
  return best;
}

// Group results by method, computing mean across scenes
function buildMethodRows(
  allResults: BenchmarkResult[],
  dataset: string,
  scenes: string[]
) {
  const sceneSet = new Set(scenes);
  const filtered = allResults.filter(
    (r) => r.dataset === dataset && sceneSet.has(r.scene)
  );

  // Group by method
  const byMethod = new Map<string, Map<string, BenchmarkResult>>();
  for (const r of filtered) {
    if (!byMethod.has(r.method)) byMethod.set(r.method, new Map());
    byMethod.get(r.method)!.set(r.scene, r);
  }

  return byMethod;
}

export function Benchmarks() {
  const [data, setData] = useState<BenchmarksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>("psnr");
  const [expandedMethod, setExpandedMethod] = useState<string | null>(null);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);

  useEffect(() => {
    getBenchmarks()
      .then((res) => {
        setData(res);
        if (res.datasets.length > 0) {
          setSelectedDataset(res.datasets[0]);
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="h-full bg-gray-950 text-white flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-gray-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full bg-gray-950 text-white">
        <Header />
        <div className="p-6 max-w-6xl mx-auto">
          <div className="bg-red-900/30 text-red-300 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const scenes = data.scenes_by_dataset[selectedDataset] ?? [];
  const allResults = [...data.results, ...data.published];
  const methodMap = buildMethodRows(allResults, selectedDataset, scenes);
  const bestPerScene = findBestPerScene(allResults, scenes, selectedMetric);

  // Sort methods: local first, then published
  const methods = Array.from(methodMap.keys()).sort((a, b) => {
    const aPub = a.startsWith("published-");
    const bPub = b.startsWith("published-");
    if (aPub !== bPub) return aPub ? 1 : -1;
    return a.localeCompare(b);
  });

  return (
    <div className="h-full bg-gray-950 text-white flex flex-col">
      <Header />

      <div className="p-6 max-w-[90rem] mx-auto flex-1 overflow-y-auto">
        {/* Dataset tabs */}
        {data.datasets.length > 1 && (
          <div className="flex gap-2 mb-4">
            {data.datasets.map((ds) => (
              <button
                key={ds}
                onClick={() => {
                  setSelectedDataset(ds);
                  setExpandedMethod(null);
                }}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  selectedDataset === ds
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:text-white"
                }`}
              >
                {ds}
              </button>
            ))}
          </div>
        )}

        {/* Metric selector */}
        <div className="flex gap-2 mb-6">
          {(Object.keys(METRIC_LABELS) as MetricKey[]).map((key) => (
            <button
              key={key}
              onClick={() => setSelectedMetric(key)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                selectedMetric === key
                  ? "bg-gray-700 text-white"
                  : "bg-gray-900 text-gray-400 hover:text-white"
              }`}
            >
              {METRIC_LABELS[key]}
            </button>
          ))}
        </div>

        {/* Comparison table */}
        {methods.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left px-4 py-3 text-gray-400 font-medium w-48">
                    Method
                  </th>
                  {scenes.map((scene) => (
                    <th
                      key={scene}
                      className="text-right px-4 py-3 text-gray-400 font-medium"
                    >
                      {scene}
                    </th>
                  ))}
                  <th className="text-right px-4 py-3 text-gray-400 font-medium border-l border-gray-800">
                    Mean
                  </th>
                  <th className="text-right px-4 py-3 text-gray-400 font-medium">
                    Time
                  </th>
                  <th className="text-right px-4 py-3 text-gray-400 font-medium">
                    #GS
                  </th>
                </tr>
              </thead>
              <tbody>
                {methods.map((method) => {
                  const sceneMap = methodMap.get(method)!;
                  const isPublished = method.startsWith("published-");
                  const isExpanded = expandedMethod === method;
                  const hasPerView = !isPublished && Array.from(sceneMap.values()).some(
                    (r) => r.per_view.length > 0
                  );

                  // Compute mean metric
                  const vals: number[] = [];
                  let totalTime = 0;
                  let timeCount = 0;
                  let totalGS = 0;
                  let gsCount = 0;
                  for (const scene of scenes) {
                    const r = sceneMap.get(scene);
                    if (!r) continue;
                    const v = getMetricValue(r, selectedMetric);
                    if (v != null) vals.push(v);
                    if (r.training_time_sec != null && r.training_time_sec > 0) {
                      totalTime += r.training_time_sec;
                      timeCount++;
                    }
                    if (r.num_gaussians != null && r.num_gaussians > 0) {
                      totalGS += r.num_gaussians;
                      gsCount++;
                    }
                  }
                  const meanVal =
                    vals.length > 0
                      ? vals.reduce((a, b) => a + b, 0) / vals.length
                      : null;

                  return (
                    <MethodRow
                      key={method}
                      method={method}
                      scenes={scenes}
                      sceneMap={sceneMap}
                      metric={selectedMetric}
                      bestPerScene={bestPerScene}
                      isPublished={isPublished}
                      isExpanded={isExpanded}
                      hasPerView={hasPerView}
                      meanVal={meanVal}
                      meanTime={timeCount > 0 ? totalTime / timeCount : null}
                      meanGS={gsCount > 0 ? Math.round(totalGS / gsCount) : null}
                      onToggle={() =>
                        setExpandedMethod(isExpanded ? null : method)
                      }
                      onThumbnailClick={setLightboxUrl}
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Lightbox */}
      {lightboxUrl && (
        <Lightbox url={lightboxUrl} onClose={() => setLightboxUrl(null)} />
      )}
    </div>
  );
}

function Header() {
  return (
    <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-3 shrink-0">
      <Link
        to="/"
        className="text-gray-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="w-5 h-5" />
      </Link>
      <BarChart3 className="w-5 h-5 text-blue-400" />
      <h1 className="text-xl font-semibold">Benchmarks</h1>
    </header>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-20 text-gray-500">
      <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
      <p className="mb-1">No benchmark results yet.</p>
      <p className="text-sm">
        Run{" "}
        <code className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-300">
          python scripts/run_benchmark.py
        </code>{" "}
        to generate results.
      </p>
    </div>
  );
}

function MethodRow({
  method,
  scenes,
  sceneMap,
  metric,
  bestPerScene,
  isPublished,
  isExpanded,
  hasPerView,
  meanVal,
  meanTime,
  meanGS,
  onToggle,
  onThumbnailClick,
}: {
  method: string;
  scenes: string[];
  sceneMap: Map<string, BenchmarkResult>;
  metric: MetricKey;
  bestPerScene: Record<string, number>;
  isPublished: boolean;
  isExpanded: boolean;
  hasPerView: boolean;
  meanVal: number | null;
  meanTime: number | null;
  meanGS: number | null;
  onToggle: () => void;
  onThumbnailClick: (url: string) => void;
}) {
  const rowClass = isPublished ? "text-gray-500 italic" : "text-gray-200";
  const canExpand = hasPerView;

  return (
    <>
      <tr
        className={`border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors ${
          canExpand ? "cursor-pointer" : ""
        } ${rowClass}`}
        onClick={canExpand ? onToggle : undefined}
      >
        <td className="px-4 py-2.5 font-medium">
          <div className="flex items-center gap-2">
            {canExpand ? (
              isExpanded ? (
                <ChevronDown className="w-4 h-4 text-gray-500 flex-shrink-0" />
              ) : (
                <ChevronRight className="w-4 h-4 text-gray-500 flex-shrink-0" />
              )
            ) : (
              <span className="w-4" />
            )}
            {method}
          </div>
        </td>
        {scenes.map((scene) => {
          const r = sceneMap.get(scene);
          const v = r ? getMetricValue(r, metric) : null;
          const isBest = v != null && bestPerScene[scene] === v;
          return (
            <td
              key={scene}
              className={`text-right px-4 py-2.5 tabular-nums ${
                isBest ? "text-green-400 font-medium" : ""
              }`}
            >
              {formatMetric(v, metric)}
            </td>
          );
        })}
        <td className="text-right px-4 py-2.5 tabular-nums border-l border-gray-800 font-medium">
          {formatMetric(meanVal, metric)}
        </td>
        <td className="text-right px-4 py-2.5 tabular-nums text-gray-400">
          {formatTime(meanTime)}
        </td>
        <td className="text-right px-4 py-2.5 tabular-nums text-gray-400">
          {formatCount(meanGS)}
        </td>
      </tr>

      {/* Per-view detail panel */}
      {isExpanded &&
        scenes.map((scene) => {
          const r = sceneMap.get(scene);
          if (!r || r.per_view.length === 0) return null;
          return (
            <tr key={`${method}-${scene}-detail`}>
              <td
                colSpan={scenes.length + 4}
                className="px-4 py-3 bg-gray-950/50"
              >
                <PerViewPanel
                  result={r}
                  scene={scene}
                  onThumbnailClick={onThumbnailClick}
                />
              </td>
            </tr>
          );
        })}
    </>
  );
}

function PerViewPanel({
  result,
  scene,
  onThumbnailClick,
}: {
  result: BenchmarkResult;
  scene: string;
  onThumbnailClick: (url: string) => void;
}) {
  return (
    <div className="ml-10">
      <div className="text-xs text-gray-500 mb-2 font-medium">
        {result.method} / {scene} — per-view breakdown
      </div>

      {/* Per-view sub-table */}
      <div className="overflow-x-auto mb-3">
        <table className="text-xs">
          <thead>
            <tr className="text-gray-500">
              <th className="text-left pr-4 py-1 font-medium">View</th>
              <th className="text-right px-3 py-1 font-medium">PSNR</th>
              <th className="text-right px-3 py-1 font-medium">SSIM</th>
              <th className="text-right px-3 py-1 font-medium">LPIPS</th>
            </tr>
          </thead>
          <tbody>
            {result.per_view.map((pv) => (
              <tr key={pv.name} className="text-gray-300">
                <td className="pr-4 py-0.5">{pv.name}</td>
                <td className="text-right px-3 py-0.5 tabular-nums">
                  {pv.psnr.toFixed(2)}
                </td>
                <td className="text-right px-3 py-0.5 tabular-nums">
                  {pv.ssim.toFixed(4)}
                </td>
                <td className="text-right px-3 py-0.5 tabular-nums">
                  {pv.lpips.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Render thumbnails */}
      {result.render_urls.length > 0 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {result.render_urls.map((url) => (
            <button
              key={url}
              onClick={(e) => {
                e.stopPropagation();
                onThumbnailClick(url);
              }}
              className="flex-shrink-0 rounded border border-gray-700 hover:border-blue-500 overflow-hidden transition-colors"
            >
              <img
                src={url}
                alt=""
                className="h-16 w-auto object-cover"
                loading="lazy"
              />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Lightbox({
  url,
  onClose,
}: {
  url: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
      >
        <X className="w-6 h-6" />
      </button>
      <img
        src={url}
        alt=""
        className="max-w-full max-h-full object-contain rounded-lg"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
