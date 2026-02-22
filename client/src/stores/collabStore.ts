import { create } from "zustand";
import { useAnySplatStore } from "./anysplatStore.ts";

export interface RemoteUser {
  user_id: string;
  color: string;
  cursor?: [number, number, number];
  selectedSegment?: number;
}

interface CollabStore {
  connected: boolean;
  users: RemoteUser[];
  lockedSegments: Map<number, string>; // segment_id -> user_id
  ws: WebSocket | null;

  connect: (projectId: string) => void;
  disconnect: () => void;
  sendCursorMove: (position: [number, number, number]) => void;
  sendSelectSegment: (segmentId: number | null) => void;
  sendTransformStart: (segmentId: number) => void;
  sendTransformEnd: (segmentId: number) => void;
  isSegmentLocked: (segmentId: number) => boolean;
}

export const useCollabStore = create<CollabStore>((set, get) => ({
  connected: false,
  users: [],
  lockedSegments: new Map(),
  ws: null,

  connect: (projectId) => {
    const existing = get().ws;
    if (existing && existing.readyState <= WebSocket.OPEN) {
      existing.close();
    }

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/collab/${projectId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => set({ connected: true });
    ws.onclose = () => set({ connected: false, users: [], ws: null });

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "presence") {
          set({ users: data.users || [] });
        } else if (data.type === "cursor_move") {
          set((s) => ({
            users: s.users.map((u) =>
              u.user_id === data.user_id
                ? { ...u, cursor: data.position, color: data.color }
                : u
            ),
          }));
        } else if (data.type === "select_segment") {
          set((s) => ({
            users: s.users.map((u) =>
              u.user_id === data.user_id
                ? { ...u, selectedSegment: data.segment_id }
                : u
            ),
          }));
        } else if (data.type === "segment_locked") {
          set((s) => {
            const locks = new Map(s.lockedSegments);
            locks.set(data.segment_id, data.user_id);
            return { lockedSegments: locks };
          });
        } else if (data.type === "segment_unlocked") {
          set((s) => {
            const locks = new Map(s.lockedSegments);
            locks.delete(data.segment_id);
            return { lockedSegments: locks };
          });
        } else if (data.type === "lock_state") {
          const locks = new Map<number, string>();
          for (const [k, v] of Object.entries(data.locks)) {
            locks.set(Number(k), v as string);
          }
          set({ lockedSegments: locks });
        } else if (data.type === "ply_changed") {
          // Trigger PLY reload
          useAnySplatStore.setState((s) => ({
            plyVersion: s.plyVersion + 1,
          }));
        }
      } catch {
        // ignore parse errors
      }
    };

    set({ ws });
  },

  disconnect: () => {
    const ws = get().ws;
    if (ws && ws.readyState <= WebSocket.OPEN) {
      ws.close();
    }
    set({ ws: null, connected: false, users: [], lockedSegments: new Map() });
  },

  sendCursorMove: (position) => {
    const ws = get().ws;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cursor_move", position }));
    }
  },

  sendSelectSegment: (segmentId) => {
    const ws = get().ws;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "select_segment", segment_id: segmentId }));
    }
  },

  sendTransformStart: (segmentId) => {
    const ws = get().ws;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "transform_start", segment_id: segmentId }));
    }
  },

  sendTransformEnd: (segmentId) => {
    const ws = get().ws;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "transform_end", segment_id: segmentId }));
    }
  },

  isSegmentLocked: (segmentId) => {
    return get().lockedSegments.has(segmentId);
  },
}));
