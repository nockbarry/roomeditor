import { api } from "./client.ts";
import type { FrameManifest, AnySplatRunResult } from "../types/api.ts";

export async function extractFrames(
  projectId: string,
  params?: { fps?: number }
): Promise<FrameManifest> {
  return api.post<FrameManifest>(`/projects/${projectId}/extract-frames`, params ?? {});
}

export async function fetchFrames(projectId: string): Promise<FrameManifest> {
  return api.get<FrameManifest>(`/projects/${projectId}/frames`);
}

export async function updateFrameSelection(
  projectId: string,
  updates: Record<string, boolean>
): Promise<FrameManifest> {
  return api.put<FrameManifest>(`/projects/${projectId}/frames`, { updates });
}

export async function runAnySplat(
  projectId: string,
  params: {
    maxViews: number;
    resolution: number;
    chunked?: boolean;
    chunkSize?: number;
    chunkOverlap?: number;
  }
): Promise<AnySplatRunResult> {
  return api.post<AnySplatRunResult>(`/projects/${projectId}/anysplat-run`, {
    max_views: params.maxViews,
    resolution: params.resolution,
    chunked: params.chunked ?? false,
    chunk_size: params.chunkSize ?? 32,
    chunk_overlap: params.chunkOverlap ?? 8,
  });
}
