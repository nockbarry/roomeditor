import type { TrainingConfig } from "../types/api.ts";

export interface Preset {
  name: string;
  description: string;
  estimatedTime: string;
  config: TrainingConfig;
}

export const PRESETS_3DGS: Preset[] = [
  {
    name: "Fast",
    description: "Quick preview, may have floaters",
    estimatedTime: "~5 min",
    config: {
      iterations: 7_000,
      sh_degree: 2,
      mode: "3dgs",
      sfm_backend: "colmap",
      mast3r_image_size: 512,
      depth_reg_weight: 0.1,
      opacity_reg_weight: 0.0,
      scale_reg_weight: 0.0,
      flatten_reg_weight: 0.0,
      distortion_weight: 0.0,
      normal_weight: 0.0,
      prune_opa: 0.005,
      densify_until_pct: 0.5,
      appearance_embeddings: false,
      tidi_pruning: false,
    },
  },
  {
    name: "Balanced",
    description: "Good quality with moderate floater reduction",
    estimatedTime: "~10 min",
    config: {
      iterations: 15_000,
      sh_degree: 2,
      mode: "3dgs",
      sfm_backend: "colmap",
      mast3r_image_size: 512,
      depth_reg_weight: 0.2,
      opacity_reg_weight: 0.01,
      scale_reg_weight: 0.01,
      flatten_reg_weight: 0.0,
      distortion_weight: 0.0,
      normal_weight: 0.0,
      prune_opa: 0.01,
      densify_until_pct: 0.55,
      appearance_embeddings: true,
      tidi_pruning: true,
    },
  },
  {
    name: "Quality",
    description: "Best quality, aggressive floater removal",
    estimatedTime: "~15 min",
    config: {
      iterations: 25_000,
      sh_degree: 2,
      mode: "3dgs",
      sfm_backend: "colmap",
      mast3r_image_size: 512,
      depth_reg_weight: 0.3,
      opacity_reg_weight: 0.02,
      scale_reg_weight: 0.02,
      flatten_reg_weight: 0.01,
      distortion_weight: 0.0,
      normal_weight: 0.0,
      prune_opa: 0.02,
      densify_until_pct: 0.6,
      appearance_embeddings: true,
      tidi_pruning: true,
    },
  },
];

export const PRESETS_2DGS: Preset[] = [
  {
    name: "Fast",
    description: "Quick preview with surface mode",
    estimatedTime: "~6 min",
    config: {
      iterations: 7_000,
      sh_degree: 2,
      mode: "2dgs",
      sfm_backend: "colmap",
      mast3r_image_size: 512,
      depth_reg_weight: 0.1,
      opacity_reg_weight: 0.0,
      scale_reg_weight: 0.0,
      flatten_reg_weight: 0.0,
      distortion_weight: 0.05,
      normal_weight: 0.005,
      prune_opa: 0.005,
      densify_until_pct: 0.5,
      appearance_embeddings: false,
      tidi_pruning: false,
    },
  },
  {
    name: "Balanced",
    description: "Clean surfaces with floater reduction",
    estimatedTime: "~12 min",
    config: {
      iterations: 15_000,
      sh_degree: 2,
      mode: "2dgs",
      sfm_backend: "colmap",
      mast3r_image_size: 512,
      depth_reg_weight: 0.2,
      opacity_reg_weight: 0.01,
      scale_reg_weight: 0.01,
      flatten_reg_weight: 0.0,
      distortion_weight: 0.1,
      normal_weight: 0.01,
      prune_opa: 0.01,
      densify_until_pct: 0.55,
      appearance_embeddings: true,
      tidi_pruning: true,
    },
  },
  {
    name: "Quality",
    description: "Best surfaces, ideal for mesh extraction",
    estimatedTime: "~18 min",
    config: {
      iterations: 25_000,
      sh_degree: 2,
      mode: "2dgs",
      sfm_backend: "colmap",
      mast3r_image_size: 512,
      depth_reg_weight: 0.3,
      opacity_reg_weight: 0.02,
      scale_reg_weight: 0.02,
      flatten_reg_weight: 0.0,
      distortion_weight: 0.2,
      normal_weight: 0.02,
      prune_opa: 0.02,
      densify_until_pct: 0.6,
      appearance_embeddings: true,
      tidi_pruning: true,
    },
  },
];

export const PRESETS_ANYSPLAT: Preset[] = [
  {
    name: "Quick",
    description: "16 views @ 448px, instant results",
    estimatedTime: "~3 sec",
    config: {
      iterations: 0,
      sh_degree: 0,
      mode: "3dgs",
      sfm_backend: "anysplat",
      mast3r_image_size: 448,
      depth_reg_weight: 0,
      opacity_reg_weight: 0,
      scale_reg_weight: 0,
      flatten_reg_weight: 0,
      distortion_weight: 0,
      normal_weight: 0,
      prune_opa: 0,
      densify_until_pct: 0,
      appearance_embeddings: false,
      tidi_pruning: false,
      anysplat_max_views: 16,
      anysplat_chunked: false,
    },
  },
  {
    name: "Standard",
    description: "32 views @ 448px, best detail",
    estimatedTime: "~5 sec",
    config: {
      iterations: 0,
      sh_degree: 0,
      mode: "3dgs",
      sfm_backend: "anysplat",
      mast3r_image_size: 448,
      depth_reg_weight: 0,
      opacity_reg_weight: 0,
      scale_reg_weight: 0,
      flatten_reg_weight: 0,
      distortion_weight: 0,
      normal_weight: 0,
      prune_opa: 0,
      densify_until_pct: 0,
      appearance_embeddings: false,
      tidi_pruning: false,
      anysplat_max_views: 32,
      anysplat_chunked: false,
    },
  },
  {
    name: "Max Coverage",
    description: "64 views @ 336px, full room",
    estimatedTime: "~7 sec",
    config: {
      iterations: 0,
      sh_degree: 0,
      mode: "3dgs",
      sfm_backend: "anysplat",
      mast3r_image_size: 448,
      depth_reg_weight: 0,
      opacity_reg_weight: 0,
      scale_reg_weight: 0,
      flatten_reg_weight: 0,
      distortion_weight: 0,
      normal_weight: 0,
      prune_opa: 0,
      densify_until_pct: 0,
      appearance_embeddings: false,
      tidi_pruning: false,
      anysplat_max_views: 64,
      anysplat_chunked: false,
    },
  },
];

export function getPresets(mode: "3dgs" | "2dgs" | "anysplat"): Preset[] {
  if (mode === "anysplat") return PRESETS_ANYSPLAT;
  return mode === "2dgs" ? PRESETS_2DGS : PRESETS_3DGS;
}

// Slider parameter definitions for the advanced panel
export const PARAM_DEFS = [
  { key: "iterations" as const, label: "Iterations", min: 3000, max: 40000, step: 1000 },
  { key: "depth_reg_weight" as const, label: "Depth Regularization", min: 0, max: 0.5, step: 0.05 },
  { key: "opacity_reg_weight" as const, label: "Opacity Regularization", min: 0, max: 0.1, step: 0.005 },
  { key: "scale_reg_weight" as const, label: "Scale Regularization", min: 0, max: 0.1, step: 0.005 },
  { key: "flatten_reg_weight" as const, label: "Flatten Regularization", min: 0, max: 0.05, step: 0.005 },
  { key: "prune_opa" as const, label: "Prune Opacity Threshold", min: 0.001, max: 0.05, step: 0.001 },
  { key: "densify_until_pct" as const, label: "Densify Until (%)", min: 0.3, max: 0.8, step: 0.05 },
];

// Additional slider params visible only in 2DGS mode
export const PARAM_DEFS_2DGS = [
  { key: "distortion_weight" as const, label: "Distortion Loss Weight", min: 0, max: 0.5, step: 0.01 },
  { key: "normal_weight" as const, label: "Normal Consistency Weight", min: 0, max: 0.1, step: 0.005 },
];
