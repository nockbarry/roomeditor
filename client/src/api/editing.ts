import { api } from "./client.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type EditType = "transform" | "delete" | "undelete" | "lighting" | "property";

export interface EditRequest {
  type: EditType;
  indices: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
  brightness?: number;
  color_tint?: number[];
  sh_scale?: number;
  attr?: string;
  values?: number[];
}

export interface EditResult {
  status: string;
  label: string;
  undo_count: number;
  redo_count: number;
  n_affected: number;
}

export interface HistoryEntry {
  label: string;
  n_affected: number;
}

export interface SceneHistory {
  undo: HistoryEntry[];
  redo: HistoryEntry[];
  undo_count: number;
  redo_count: number;
  dirty: boolean;
}

export interface SceneLoadResult {
  status: string;
  n_gaussians: number;
  n_deleted: number;
  undo_count: number;
  redo_count: number;
  dirty: boolean;
}

export interface SceneSaveResult {
  status: string;
  n_gaussians: number;
  save_time_sec: number;
}

export interface QueryResult {
  indices: number[];
  count: number;
}

export interface DeleteRegionRequest {
  shape: "box" | "sphere";
  min?: number[];
  max?: number[];
  center?: number[];
  radius?: number;
  mode: "inside" | "outside";
}

export interface UndoRedoResult {
  status: string;
  label: string;
  undo_count: number;
  redo_count: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function loadScene(projectId: string): Promise<SceneLoadResult> {
  return api.post<SceneLoadResult>(`/projects/${projectId}/scene/load`);
}

export async function saveScene(projectId: string): Promise<SceneSaveResult> {
  return api.post<SceneSaveResult>(`/projects/${projectId}/scene/save`);
}

export async function editScene(projectId: string, edit: EditRequest): Promise<EditResult> {
  return api.post<EditResult>(`/projects/${projectId}/scene/edit`, edit);
}

export async function undoEdit(projectId: string): Promise<UndoRedoResult> {
  return api.post<UndoRedoResult>(`/projects/${projectId}/scene/undo`);
}

export async function redoEdit(projectId: string): Promise<UndoRedoResult> {
  return api.post<UndoRedoResult>(`/projects/${projectId}/scene/redo`);
}

export async function getHistory(projectId: string): Promise<SceneHistory> {
  return api.get<SceneHistory>(`/projects/${projectId}/scene/history`);
}

export async function queryBox(
  projectId: string,
  min: number[],
  max: number[],
  excludeDeleted = true,
): Promise<QueryResult> {
  return api.post<QueryResult>(`/projects/${projectId}/scene/query-box`, {
    min,
    max,
    exclude_deleted: excludeDeleted,
  });
}

export async function querySphere(
  projectId: string,
  center: number[],
  radius: number,
  excludeDeleted = true,
): Promise<QueryResult> {
  return api.post<QueryResult>(`/projects/${projectId}/scene/query-sphere`, {
    center,
    radius,
    exclude_deleted: excludeDeleted,
  });
}

export async function deleteRegion(
  projectId: string,
  request: DeleteRegionRequest,
): Promise<EditResult> {
  return api.post<EditResult>(`/projects/${projectId}/scene/delete-region`, request);
}
