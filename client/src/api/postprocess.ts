import { api } from "./client.ts";

export interface PruneResult {
  n_before: number;
  n_after: number;
  n_pruned: number;
}

export interface QualityStats {
  n_gaussians: number;
  n_effective: number;
  frac_transparent: number;
  frac_opaque: number;
  opacity_mean: number;
  log_scale_mean: number;
  log_scale_std: number;
  frac_scale_outlier: number;
  mean_nn_dist: number;
  bbox_volume: number;
  density: number;
}

export interface RefineResult {
  status: string;
  n_gaussians: number;
  iterations_run: number;
  ply_url: string;
  pre_stats: Record<string, number> | null;
  post_stats: Record<string, number> | null;
}

export interface RefinePresetInfo {
  label: string;
  description: string;
  iterations: number;
}

export interface RefineMetrics {
  step: number;
  total_steps: number;
  losses: { total: number; l1: number; ssim: number; depth_tv: number };
  lr: { means: number };
  n_gaussians: number;
  active_sh_degree: number;
}

export interface RefineEval {
  step: number;
  total_steps: number;
  mean_psnr: number;
  mean_ssim: number;
  per_view: Array<{ index: number; psnr: number; ssim: number }>;
}

export async function pruneSplat(
  projectId: string,
  params?: { opacityThreshold?: number; maxGaussians?: number }
): Promise<PruneResult> {
  return api.post<PruneResult>(`/projects/${projectId}/prune`, {
    opacity_threshold: params?.opacityThreshold ?? 0.01,
    max_gaussians: params?.maxGaussians ?? 2_000_000,
  });
}

export async function getQualityStats(
  projectId: string
): Promise<QualityStats> {
  return api.get<QualityStats>(`/projects/${projectId}/quality-stats`);
}

export interface ComparisonPair {
  label: string;
  before_url: string;
  after_url: string;
}

export async function getComparisonInfo(
  projectId: string
): Promise<{ pairs: ComparisonPair[] }> {
  return api.get(`/projects/${projectId}/comparison-info`);
}

export async function refineSplat(
  projectId: string,
  params?: { preset?: string; mode?: string }
): Promise<RefineResult> {
  return api.post<RefineResult>(`/projects/${projectId}/refine`, {
    preset: params?.preset ?? "balanced",
    mode: params?.mode ?? "3dgs",
  });
}

export async function stopRefinement(projectId: string): Promise<void> {
  await api.post(`/projects/${projectId}/refine/stop`, {});
}

export async function getRefinePresets(): Promise<Record<string, RefinePresetInfo>> {
  return api.get(`/projects/config/refine-presets`);
}
