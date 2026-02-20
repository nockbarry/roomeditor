import type { JobProgress, TrainingPreview, TrainingSnapshot } from "../types/api.ts";

export function connectJobWebSocket(
  jobId: string,
  onProgress: (data: JobProgress) => void,
  onPreview?: (data: TrainingPreview) => void,
  onSnapshot?: (data: TrainingSnapshot) => void,
  onError?: (error: Event) => void
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/jobs/${jobId}`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "ping") return;
      if (data.type === "training_preview") {
        onPreview?.(data as TrainingPreview);
        return;
      }
      if (data.type === "training_snapshot") {
        onSnapshot?.(data as TrainingSnapshot);
        return;
      }
      onProgress(data as JobProgress);
    } catch (e) {
      console.error("Failed to parse WebSocket message:", e);
    }
  };

  ws.onerror = (event) => {
    console.error("WebSocket error:", event);
    onError?.(event);
  };

  return ws;
}
