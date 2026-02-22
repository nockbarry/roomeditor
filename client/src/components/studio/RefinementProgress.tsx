import { useState } from "react";
import { useAnySplatStore } from "../../stores/anysplatStore.ts";
import { Loader2, Square, ChevronDown, ChevronUp } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
} from "recharts";

export function RefinementProgress({ projectId }: { projectId: string }) {
  const isRefining = useAnySplatStore((s) => s.isRefining);
  const refineStep = useAnySplatStore((s) => s.refineStep);
  const refineTotal = useAnySplatStore((s) => s.refineTotal);
  const refineLoss = useAnySplatStore((s) => s.refineLoss);
  const refineGaussianCount = useAnySplatStore((s) => s.refineGaussianCount);
  const refineStartTime = useAnySplatStore((s) => s.refineStartTime);
  const metricsHistory = useAnySplatStore((s) => s.refineMetricsHistory);
  const evalHistory = useAnySplatStore((s) => s.refineEvalHistory);
  const baseline = useAnySplatStore((s) => s.refineBaseline);
  const stopRefinement = useAnySplatStore((s) => s.stopRefinement);

  const [showCharts, setShowCharts] = useState(true);

  if (!isRefining || (refineTotal === 0 && metricsHistory.length === 0)) return null;

  const total = refineTotal || (metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1].total_steps : 0);
  const pct = total > 0 ? Math.min(100, (refineStep / total) * 100) : 0;

  // ETA calculation
  let etaStr = "";
  if (refineStartTime && refineStep > 0 && total > 0) {
    const elapsed = (Date.now() - refineStartTime) / 1000;
    const secPerStep = elapsed / refineStep;
    const remaining = secPerStep * (total - refineStep);
    if (remaining < 60) {
      etaStr = `~${Math.round(remaining)}s left`;
    } else {
      etaStr = `~${Math.round(remaining / 60)}m left`;
    }
  }

  // PSNR delta vs baseline
  let psnrDelta: number | null = null;
  if (baseline && evalHistory.length > 0) {
    const latest = evalHistory[evalHistory.length - 1];
    psnrDelta = latest.mean_psnr - baseline.psnr;
  }

  // Prepare chart data — downsample if too many points
  const lossData = metricsHistory.map((m) => ({
    step: m.step,
    loss: m.losses.total,
  }));

  const evalData = evalHistory.map((e) => ({
    step: e.step,
    psnr: e.mean_psnr,
  }));

  return (
    <div className="border-b border-gray-800 bg-blue-950/20">
      {/* Header bar — always visible */}
      <div className="px-3 py-2 space-y-1.5">
        <div className="flex items-center gap-1.5 text-xs">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />
          <span className="font-medium text-blue-300">Refining...</span>

          {psnrDelta !== null && (
            <span
              className={`text-[10px] font-mono ${
                psnrDelta >= 0 ? "text-emerald-400" : "text-red-400"
              }`}
            >
              {psnrDelta >= 0 ? "+" : ""}
              {psnrDelta.toFixed(1)} dB
            </span>
          )}

          <span className="text-gray-500 ml-auto text-[10px]">{etaStr}</span>

          <button
            onClick={() => stopRefinement(projectId)}
            title="Stop refinement"
            className="p-1 rounded hover:bg-red-900/40 text-red-400 hover:text-red-300 transition-colors"
          >
            <Square className="w-3 h-3 fill-current" />
          </button>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-gray-800 rounded-full h-1.5">
          <div
            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>

        {/* Stats row */}
        <div className="flex items-center gap-3 text-[10px] text-gray-500">
          <span>
            {refineStep.toLocaleString()} / {total.toLocaleString()}
          </span>
          {refineLoss !== null && (
            <span>loss: {refineLoss.toFixed(5)}</span>
          )}
          {refineGaussianCount !== null && refineGaussianCount > 0 && (
            <span className="ml-auto">{refineGaussianCount.toLocaleString()} gs</span>
          )}
        </div>
      </div>

      {/* Collapsible charts section */}
      {(lossData.length > 1 || evalData.length > 0) && (
        <>
          <button
            onClick={() => setShowCharts(!showCharts)}
            className="w-full px-3 py-1 flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-400 transition-colors border-t border-gray-800/50"
          >
            {showCharts ? (
              <ChevronUp className="w-3 h-3" />
            ) : (
              <ChevronDown className="w-3 h-3" />
            )}
            Charts
          </button>

          {showCharts && (
            <div className="px-3 pb-2 space-y-2">
              {/* Loss chart */}
              {lossData.length > 1 && (
                <div>
                  <div className="text-[9px] text-gray-600 mb-0.5">Loss</div>
                  <ResponsiveContainer width="100%" height={120}>
                    <LineChart data={lossData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        dataKey="step"
                        stroke="#6b7280"
                        tick={{ fontSize: 9 }}
                        tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)}
                      />
                      <YAxis
                        stroke="#6b7280"
                        tick={{ fontSize: 9 }}
                        width={40}
                        tickFormatter={(v: number) => v.toFixed(3)}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", fontSize: 10 }}
                        labelFormatter={(label) => `Step ${label}`}
                        formatter={(value) => [Number(value).toFixed(5), "Loss"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#3b82f6"
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* PSNR chart */}
              {evalData.length > 0 && (
                <div>
                  <div className="text-[9px] text-gray-600 mb-0.5">PSNR (dB)</div>
                  <ResponsiveContainer width="100%" height={120}>
                    <LineChart data={evalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        dataKey="step"
                        stroke="#6b7280"
                        tick={{ fontSize: 9 }}
                        tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)}
                      />
                      <YAxis
                        stroke="#6b7280"
                        tick={{ fontSize: 9 }}
                        width={40}
                        domain={["auto", "auto"]}
                        tickFormatter={(v: number) => v.toFixed(1)}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", fontSize: 10 }}
                        labelFormatter={(label) => `Step ${label}`}
                        formatter={(value) => [`${Number(value).toFixed(2)} dB`, "PSNR"]}
                      />
                      {baseline && (
                        <ReferenceLine
                          y={baseline.psnr}
                          stroke="#f59e0b"
                          strokeDasharray="5 5"
                          strokeWidth={1}
                          label={{
                            value: "baseline",
                            position: "insideTopRight",
                            fill: "#f59e0b",
                            fontSize: 9,
                          }}
                        />
                      )}
                      <Line
                        type="monotone"
                        dataKey="psnr"
                        stroke="#10b981"
                        strokeWidth={1.5}
                        dot={{ r: 3, fill: "#10b981" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
