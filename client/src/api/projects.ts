import { api } from "./client.ts";
import type { Project, Job, TrainingConfig } from "../types/api.ts";

export async function listProjects(): Promise<Project[]> {
  return api.get<Project[]>("/projects");
}

export async function createProject(name: string): Promise<Project> {
  return api.post<Project>("/projects", { name });
}

export async function getProject(id: string): Promise<Project> {
  return api.get<Project>(`/projects/${id}`);
}

export async function deleteProject(id: string): Promise<void> {
  return api.delete(`/projects/${id}`);
}

export async function uploadFiles(
  projectId: string,
  files: File[]
): Promise<Project> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const res = await fetch(`/api/projects/${projectId}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(body);
  }

  return res.json();
}

export async function listSources(
  projectId: string
): Promise<{ files: Array<{ name: string; size: number; type: string }> }> {
  return api.get(`/projects/${projectId}/sources`);
}

export async function startReconstruction(
  projectId: string,
  config?: TrainingConfig
): Promise<Job> {
  return api.post<Job>(`/projects/${projectId}/reconstruct`, config);
}

export interface Checkpoint {
  filename: string;
  step: number;
  size: number;
  modified: number;
}

export async function listCheckpoints(
  projectId: string
): Promise<{ checkpoints: Checkpoint[] }> {
  return api.get(`/projects/${projectId}/checkpoints`);
}

export interface MeshResult {
  mesh_url: string;
  vertex_count: number;
  face_count: number;
}

export async function extractMesh(
  projectId: string,
  voxelSize: number = 0.02
): Promise<MeshResult> {
  return api.post<MeshResult>(
    `/projects/${projectId}/extract-mesh?voxel_size=${voxelSize}`
  );
}
