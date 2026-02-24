export type ToolMode = "select" | "translate" | "rotate" | "scale" | "brush" | "eraser" | "crop-box" | "crop-sphere";

export type CameraMode = "orbit" | "fps";

export interface EditorState {
  toolMode: ToolMode;
  cameraMode: CameraMode;
  selectedObjectId: string | null;
  showGrid: boolean;
}
