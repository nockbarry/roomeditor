import { api } from "./client.ts";
import type { SegmentManifest, SegmentInfo } from "../types/api.ts";

export async function autoSegment(projectId: string): Promise<SegmentManifest> {
  return api.post<SegmentManifest>(`/projects/${projectId}/auto-segment`);
}

export async function clickSegment(
  projectId: string,
  frame: string,
  x: number,
  y: number
): Promise<SegmentInfo> {
  return api.post<SegmentInfo>(`/projects/${projectId}/click-segment`, { frame, x, y });
}

export async function getSegments(projectId: string): Promise<SegmentManifest> {
  return api.get<SegmentManifest>(`/projects/${projectId}/segments`);
}

export async function assignGaussians(projectId: string): Promise<SegmentManifest> {
  return api.post<SegmentManifest>(`/projects/${projectId}/assign-gaussians`);
}

export async function transformSegment(
  projectId: string,
  segmentId: number,
  transform: { translation?: number[]; rotation?: number[]; scale?: number[] }
): Promise<{ status: string }> {
  return api.put(`/projects/${projectId}/segments/${segmentId}/transform`, transform);
}

export async function deleteSegment(
  projectId: string,
  segmentId: number
): Promise<{ status: string }> {
  return api.delete(`/projects/${projectId}/segments/${segmentId}`);
}

export async function toggleVisibility(
  projectId: string,
  segmentId: number,
  visible: boolean
): Promise<{ status: string }> {
  return api.put(`/projects/${projectId}/segments/${segmentId}/visibility`, { visible });
}

export async function duplicateSegment(
  projectId: string,
  segmentId: number,
  offset: number[] = [0.5, 0, 0]
): Promise<{ status: string; new_segment: { id: number; label: string; n_gaussians: number } }> {
  return api.post(`/projects/${projectId}/segments/${segmentId}/duplicate`, { offset });
}

export async function renameSegment(
  projectId: string,
  segmentId: number,
  label: string
): Promise<{ status: string }> {
  return api.put(`/projects/${projectId}/segments/${segmentId}/rename`, { label });
}

export async function createBackground(
  projectId: string
): Promise<{ status: string; segment: { id: number; label: string; n_gaussians: number } }> {
  return api.post(`/projects/${projectId}/segments/create-background`);
}

export async function batchTransform(
  projectId: string,
  segmentIds: number[],
  transform: { translation?: number[]; rotation?: number[]; scale?: number[] }
): Promise<{ status: string }> {
  return api.put(`/projects/${projectId}/segments/batch-transform`, {
    segment_ids: segmentIds,
    transform,
  });
}

export async function undo(projectId: string): Promise<{ status: string; remaining: number }> {
  return api.post(`/projects/${projectId}/undo`);
}

export async function restoreOriginal(projectId: string): Promise<{ status: string }> {
  return api.post(`/projects/${projectId}/restore-original`);
}

export async function fetchSegmentIndexMap(projectId: string): Promise<Uint8Array> {
  const res = await fetch(`/api/projects/${projectId}/segment-index-map`);
  if (!res.ok) throw new Error(`Failed to fetch segment index map: ${res.status}`);
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

export async function getUndoStack(
  projectId: string
): Promise<{ checkpoints: { id: string; label: string; timestamp: number }[]; count: number }> {
  return api.get(`/projects/${projectId}/undo-stack`);
}
