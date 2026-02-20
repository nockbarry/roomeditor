export interface Project {
  id: string;
  name: string;
  status: "created" | "uploaded" | "processing" | "ready" | "error";
  video_filename: string | null;
  gaussian_count: number | null;
  reconstruction_mode: "anysplat" | "training" | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface Job {
  id: string;
  project_id: string;
  job_type: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  current_step: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobProgress {
  job_id: string;
  status: string;
  progress: number;
  current_step: string | null;
  error_message: string | null;
}

export interface SceneObject {
  id: string;
  project_id: string;
  segment_id: number;
  label: string;
  gaussian_start: number;
  gaussian_count: number;
  centroid: [number, number, number] | null;
  bbox_min: [number, number, number] | null;
  bbox_max: [number, number, number] | null;
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
  visible: boolean;
  locked: boolean;
  source: "reconstruction" | "generated" | "imported";
}

export interface TrainingPreview {
  type: "training_preview";
  job_id: string;
  step: number;
  total_steps: number;
  loss: number;
  n_gaussians: number;
  preview_base64: string;
}

export interface TrainingSnapshot {
  type: "training_snapshot";
  job_id: string;
  step: number;
  total_steps: number;
  loss: number;
  n_gaussians: number;
  snapshot_url: string;
}

export interface TrainingConfig {
  iterations: number;
  sh_degree: number;
  mode: "3dgs" | "2dgs";
  sfm_backend: "colmap" | "mast3r" | "anysplat";
  mast3r_image_size: number;
  depth_reg_weight: number;
  opacity_reg_weight: number;
  scale_reg_weight: number;
  flatten_reg_weight: number;
  distortion_weight: number;
  normal_weight: number;
  prune_opa: number;
  densify_until_pct: number;
  appearance_embeddings: boolean;
  tidi_pruning: boolean;
  resume_from?: string | null;
  anysplat_max_views?: number;
  anysplat_chunked?: boolean;
}

export interface ObjectTransformUpdate {
  translation?: [number, number, number];
  rotation?: [number, number, number, number];
  scale?: [number, number, number];
  visible?: boolean;
  locked?: boolean;
}

// --- AnySplat Studio ---

export interface FrameInfo {
  filename: string;
  source_file: string;
  source_type: "video" | "image" | "";
  sharpness: number;
  selected: boolean;
}

export interface FrameManifest {
  frames: FrameInfo[];
  total: number;
  selected_count: number;
}

export interface CleanupStats {
  n_before: number;
  n_after: number;
  n_opacity_removed: number;
  n_scale_removed: number;
  n_floater_removed: number;
}

export interface AnySplatRunResult {
  status: string;
  n_gaussians: number;
  duration_sec: number;
  resolution_used: number;
  views_used: number;
  ply_url: string;
  cleanup_stats: CleanupStats | null;
}

// --- Segmentation ---

export interface SegmentInfo {
  id: number;
  label: string;
  area: number;
  bbox: [number, number, number, number];
  confidence: number;
  primary_frame: string;
  color: [number, number, number];
  n_gaussians: number;
  click_point: [number, number] | null;
  visible: boolean;
}

export interface SegmentManifest {
  segments: SegmentInfo[];
  total: number;
  primary_frame: string | null;
  total_gaussians: number;
  unassigned_gaussians: number;
}

export interface BenchmarkMetrics {
  mean_psnr: number;
  mean_ssim: number;
  mean_lpips: number;
}

export interface BenchmarkPerView {
  name: string;
  index: number;
  psnr: number;
  ssim: number;
  lpips: number;
}

export interface BenchmarkResult {
  method: string;
  dataset: string;
  scene: string;
  metrics: BenchmarkMetrics;
  config: TrainingConfig | null;
  training_time_sec: number | null;
  num_gaussians: number | null;
  per_view: BenchmarkPerView[];
  render_urls: string[];
  is_published: boolean;
}

export interface BenchmarksResponse {
  results: BenchmarkResult[];
  published: BenchmarkResult[];
  datasets: string[];
  methods: string[];
  scenes_by_dataset: Record<string, string[]>;
}
