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

export async function refineSplat(
  projectId: string,
  params?: { iterations?: number; mode?: string }
): Promise<RefineResult> {
  return api.post<RefineResult>(`/projects/${projectId}/refine`, {
    iterations: params?.iterations ?? 2000,
    mode: params?.mode ?? "3dgs",
  });
}
