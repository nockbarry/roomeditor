import { create } from "zustand";
import type { Project, Job, JobProgress, SceneObject, TrainingPreview, TrainingSnapshot, TrainingConfig } from "../types/api.ts";
import { api } from "../api/client.ts";
import * as projectApi from "../api/projects.ts";
import * as objectApi from "../api/objects.ts";
import { connectJobWebSocket } from "../api/ws.ts";

interface ProjectStore {
  // Project list
  projects: Project[];
  loading: boolean;
  error: string | null;

  // Current project
  currentProject: Project | null;
  objects: SceneObject[];

  // Active job
  activeJob: Job | null;
  jobProgress: JobProgress | null;
  trainingPreview: TrainingPreview | null;
  latestSnapshot: TrainingSnapshot | null;
  jobWs: WebSocket | null;

  // Actions
  fetchProjects: () => Promise<void>;
  createProject: (name: string) => Promise<Project>;
  deleteProject: (id: string) => Promise<void>;
  loadProject: (id: string) => Promise<void>;
  uploadFiles: (projectId: string, files: File[]) => Promise<void>;
  startReconstruction: (projectId: string, config?: TrainingConfig) => Promise<void>;
  watchJob: (jobId: string) => void;
  stopWatchingJob: () => void;
  fetchObjects: (projectId: string) => Promise<void>;
}

export const useProjectStore = create<ProjectStore>((set, get) => ({
  projects: [],
  loading: false,
  error: null,
  currentProject: null,
  objects: [],
  activeJob: null,
  jobProgress: null,
  trainingPreview: null,
  latestSnapshot: null,
  jobWs: null,

  fetchProjects: async () => {
    set({ loading: true, error: null });
    try {
      const projects = await projectApi.listProjects();
      set({ projects, loading: false });
    } catch (e) {
      set({ error: String(e), loading: false });
    }
  },

  createProject: async (name: string) => {
    const project = await projectApi.createProject(name);
    set((state) => ({ projects: [project, ...state.projects] }));
    return project;
  },

  deleteProject: async (id: string) => {
    await projectApi.deleteProject(id);
    set((state) => ({
      projects: state.projects.filter((p) => p.id !== id),
    }));
  },

  loadProject: async (id: string) => {
    const project = await projectApi.getProject(id);
    set({ currentProject: project });
    if (project.status === "ready") {
      await get().fetchObjects(id);
    }
    // Auto-reconnect to active job if project is processing
    if (project.status === "processing" && !get().jobWs) {
      try {
        const jobs = await api.get<Job[]>(`/jobs/project/${id}`);
        const activeJob = jobs.find((j) => j.status === "running" || j.status === "pending");
        if (activeJob) {
          set({ activeJob, jobProgress: {
            job_id: activeJob.id,
            status: activeJob.status,
            progress: activeJob.progress,
            current_step: activeJob.current_step,
            error_message: activeJob.error_message,
          }});
          get().watchJob(activeJob.id);
        }
      } catch (e) {
        console.error("Failed to reconnect to active job:", e);
      }
    }
  },

  uploadFiles: async (projectId: string, files: File[]) => {
    const updated = await projectApi.uploadFiles(projectId, files);
    set({ currentProject: updated });
    set((state) => ({
      projects: state.projects.map((p) => (p.id === projectId ? updated : p)),
    }));
  },

  startReconstruction: async (projectId: string, config?: TrainingConfig) => {
    const job = await projectApi.startReconstruction(projectId, config);
    set((state) => ({
      activeJob: job,
      jobProgress: {
        job_id: job.id,
        status: "running",
        progress: 0,
        current_step: "Starting pipeline...",
        error_message: null,
      },
      trainingPreview: null,
      latestSnapshot: null,
      currentProject: state.currentProject
        ? { ...state.currentProject, status: "processing" }
        : null,
    }));
    get().watchJob(job.id);
  },

  watchJob: (jobId: string) => {
    // Close existing connection
    get().stopWatchingJob();

    const ws = connectJobWebSocket(
      jobId,
      (progress) => {
        set({ jobProgress: progress });

        // If job completed, reload the project
        if (progress.status === "completed" || progress.status === "failed") {
          const projectId = get().currentProject?.id;
          if (projectId) {
            get().loadProject(projectId);
          }
          get().stopWatchingJob();
        }
      },
      (preview) => {
        set({ trainingPreview: preview });
      },
      (snapshot) => {
        set({ latestSnapshot: snapshot });
      },
      () => {
        set({ jobWs: null });
      }
    );

    set({ jobWs: ws });
  },

  stopWatchingJob: () => {
    const ws = get().jobWs;
    if (ws) {
      ws.close();
      set({ jobWs: null });
    }
  },

  fetchObjects: async (projectId: string) => {
    try {
      const objects = await objectApi.listObjects(projectId);
      set({ objects });
    } catch (e) {
      console.error("Failed to fetch objects:", e);
    }
  },
}));
