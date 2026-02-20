export type ToolMode = "select" | "translate" | "rotate" | "scale";

export type CameraMode = "orbit" | "fps";

export interface EditorState {
  toolMode: ToolMode;
  cameraMode: CameraMode;
  selectedObjectId: string | null;
  showGrid: boolean;
}
