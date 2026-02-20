import { api } from "./client.ts";
import type { SceneObject, ObjectTransformUpdate } from "../types/api.ts";

export async function listObjects(
  projectId: string
): Promise<SceneObject[]> {
  return api.get<SceneObject[]>(`/projects/${projectId}/objects`);
}

export async function updateObjectTransform(
  projectId: string,
  objectId: string,
  update: ObjectTransformUpdate
): Promise<SceneObject> {
  return api.put<SceneObject>(
    `/projects/${projectId}/objects/${objectId}/transform`,
    update
  );
}

export async function deleteObject(
  projectId: string,
  objectId: string
): Promise<void> {
  return api.delete(`/projects/${projectId}/objects/${objectId}`);
}
