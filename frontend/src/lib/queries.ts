/* Typed data access: one TanStack Query hook per API resource, plus mutations that
   invalidate exactly the queries they affect. Components never call fetch directly. */
import { QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { ApiError, api } from "./api";
import type {
  AlgorithmInfo, DataReport, Dataset, DatasetPreview, Diagnosis, DriftCheck, DriftReport, ExperimentJob, Explanations, Job,
  LocalExplanation, ModelComparison, ModelDetail, ModelSummary, Project, Providers, Registry, SetTargetResponse, Stage,
  StageEvent, Task, TrainJob, User,
} from "./types";

export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        refetchOnWindowFocus: false,
        // Retry network blips and 5xx once; a 4xx won't fix itself.
        retry: (count, err) => count < 1 && !(err instanceof ApiError && err.status >= 400 && err.status < 500),
      },
    },
  });
}

export const qk = {
  projects: ["projects"] as const,
  project: (id: string) => ["project", id] as const,
  projectDatasets: (id: string) => ["project", id, "datasets"] as const,
  registry: (id: string) => ["project", id, "registry"] as const,
  dataset: (id: string) => ["dataset", id] as const,
  preview: (id: string) => ["dataset", id, "preview"] as const,
  dataReport: (id: string) => ["dataset", id, "data-report"] as const,
  datasetModels: (id: string) => ["dataset", id, "models"] as const,
  diagnosis: (id: string) => ["dataset", id, "diagnosis"] as const,
  algorithms: (task: Task) => ["algorithms", task] as const,
  model: (id: string) => ["model", id] as const,
  explanations: (id: string) => ["model", id, "explanations"] as const,
  experiments: (id: string) => ["model", id, "experiments"] as const,
  events: (id: string) => ["model", id, "events"] as const,
  drift: (id: string) => ["model", id, "drift"] as const,
  compare: (a: string, b: string) => ["compare", a, b] as const,
  job: (id: string) => ["job", id] as const,
  providers: ["assistant", "providers"] as const,
  users: ["users"] as const,
};

const get = <T,>(path: string) => () => api<T>(path);

/* ---------- reads ---------- */

export const useProjects = () => useQuery({ queryKey: qk.projects, queryFn: get<Project[]>("/projects") });
export const useProject = (id: string) => useQuery({ queryKey: qk.project(id), queryFn: get<Project>(`/projects/${id}`) });
export const useProjectDatasets = (id: string) =>
  useQuery({ queryKey: qk.projectDatasets(id), queryFn: get<Dataset[]>(`/projects/${id}/datasets`) });
export const useRegistry = (id: string) => useQuery({ queryKey: qk.registry(id), queryFn: get<Registry>(`/projects/${id}/registry`) });

export const useDataset = (id: string) => useQuery({ queryKey: qk.dataset(id), queryFn: get<Dataset>(`/datasets/${id}`) });
export const usePreview = (id: string) =>
  useQuery({ queryKey: qk.preview(id), queryFn: get<DatasetPreview>(`/datasets/${id}/preview?rows=8`) });
export const useDataReport = (id: string, enabled = true) =>
  useQuery({ queryKey: qk.dataReport(id), queryFn: get<DataReport>(`/datasets/${id}/data-report`), enabled });
export const useDatasetModels = (id: string, enabled = true) =>
  useQuery({ queryKey: qk.datasetModels(id), queryFn: get<ModelSummary[]>(`/datasets/${id}/models`), enabled });
export const useDiagnosis = (id: string, enabled = true) =>
  useQuery({ queryKey: qk.diagnosis(id), queryFn: get<Diagnosis>(`/datasets/${id}/diagnosis`), enabled });
export const useAlgorithms = (task: Task) =>
  useQuery({ queryKey: qk.algorithms(task), queryFn: get<AlgorithmInfo[]>(`/algorithms?task=${task}`), staleTime: Infinity });

export const useModel = (id: string) => useQuery({ queryKey: qk.model(id), queryFn: get<ModelDetail>(`/models/${id}`) });
export const useExplanations = (id: string) =>
  useQuery({ queryKey: qk.explanations(id), queryFn: get<Explanations>(`/models/${id}/explanations`), staleTime: Infinity });
export const useModelExperiments = (id: string) =>
  useQuery({ queryKey: qk.experiments(id), queryFn: get<ExperimentJob[]>(`/models/${id}/experiments`) });
export const useModelEvents = (id: string) => useQuery({ queryKey: qk.events(id), queryFn: get<StageEvent[]>(`/models/${id}/events`) });
export const useDrift = (id: string) => useQuery({ queryKey: qk.drift(id), queryFn: get<DriftCheck[]>(`/models/${id}/drift`) });
/** Compare `a` (reference) with `b`; disabled until both ids are set. */
export const useComparison = (a: string | null, b: string | null) =>
  useQuery({
    queryKey: qk.compare(a ?? "", b ?? ""),
    queryFn: get<ModelComparison>(`/models/compare/${a}/${b}`),
    enabled: !!a && !!b && a !== b,
  });

export const useProviders = () =>
  useQuery({ queryKey: qk.providers, queryFn: get<Providers>("/assistant/providers"), retry: false });
export const useUsers = (enabled = true) => useQuery({ queryKey: qk.users, queryFn: get<User[]>("/auth/users"), enabled });

const isDone = (j: Job | undefined) => j?.status === "completed" || j?.status === "failed";

/** Poll a background job every 1.5 s until it completes or fails; `onDone` fires once. */
export function useJob<J extends Job = Job>(id: string | null, onDone?: (job: J) => void) {
  const query = useQuery({
    queryKey: qk.job(id ?? ""),
    queryFn: get<J>(`/jobs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (isDone(q.state.data) ? false : 1500),
    retry: 3,
  });
  const handled = useRef<string | null>(null);
  const callback = useRef(onDone);
  useEffect(() => { callback.current = onDone; });
  const job = query.data;
  useEffect(() => {
    if (job && isDone(job) && handled.current !== job.id) {
      handled.current = job.id;
      callback.current?.(job);
    }
  }, [job]);
  return job ?? null;
}

export const isRunning = (j: Job | null) => !!j && (j.status === "queued" || j.status === "running");

/* ---------- writes ---------- */

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; description: string | null }) => api<Project>("/projects", { json: body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api<void>(`/projects/${id}`, { method: "DELETE" }),
    onSuccess: (_d, id) => {
      qc.removeQueries({ queryKey: qk.project(id) });
      return qc.invalidateQueries({ queryKey: qk.projects });
    },
  });
}

export function useUploadDataset(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api<Dataset>(`/projects/${projectId}/datasets`, { form });
    },
    onSuccess: () => Promise.all([
      qc.invalidateQueries({ queryKey: qk.projectDatasets(projectId) }),
      qc.invalidateQueries({ queryKey: qk.projects }),
    ]),
  });
}

export function useDeleteDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (d: Pick<Dataset, "id" | "project_id">) => api<void>(`/datasets/${d.id}`, { method: "DELETE" }),
    onSuccess: (_r, d) => {
      qc.removeQueries({ queryKey: qk.dataset(d.id) });
      return qc.invalidateQueries({ queryKey: ["project", d.project_id] });
    },
  });
}

export function useSetTarget(datasetId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (column: string) => api<SetTargetResponse>(`/datasets/${datasetId}/target`, { json: { column } }),
    // The target changes every derived view of the dataset.
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.dataset(datasetId) }),
  });
}

export function useRefreshDataReport(datasetId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<DataReport>(`/datasets/${datasetId}/data-report?refresh=true`),
    onSuccess: (report) => qc.setQueryData(qk.dataReport(datasetId), report),
  });
}

export function useTrain(datasetId: string) {
  return useMutation({
    mutationFn: (body: { algorithms: string[]; drop_columns: string[]; cv_folds: number }) =>
      api<TrainJob>(`/datasets/${datasetId}/train`, { json: body }),
  });
}

export function useUploadModel(datasetId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, name }: { file: File; name?: string }) => {
      const form = new FormData();
      form.append("file", file);
      if (name) form.append("name", name);
      return api<ModelSummary>(`/datasets/${datasetId}/models/upload`, { form });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.datasetModels(datasetId) }),
  });
}

export function useStartExperiment(modelId: string) {
  return useMutation({
    mutationFn: (budget: "quick" | "standard") => api<ExperimentJob>(`/models/${modelId}/experiments`, { json: { budget } }),
  });
}

export function useExplainRow(modelId: string) {
  return useMutation({
    mutationFn: (body: { row_id: number; method: "auto" | "lime" }) =>
      api<LocalExplanation>(`/models/${modelId}/explain`, { json: body }),
  });
}

export function useMoveStage(model: Pick<ModelSummary, "id" | "project_id" | "dataset_id">) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { stage: Stage; note: string | null; actor: string | null; acknowledge_risks: boolean }) =>
      api<unknown>(`/models/${model.id}/stage`, { json: body }),
    // Promoting archives the previous production model, so refresh every model view.
    onSuccess: () => Promise.all([
      qc.invalidateQueries({ queryKey: ["model"] }),
      qc.invalidateQueries({ queryKey: qk.registry(model.project_id) }),
      qc.invalidateQueries({ queryKey: qk.datasetModels(model.dataset_id) }),
      qc.invalidateQueries({ queryKey: ["compare"] }),
    ]),
  });
}

export function useMonitorBatch(modelId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api<DriftReport>(`/models/${modelId}/monitor`, { form });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.drift(modelId) }),
  });
}

export function useUpdateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: Partial<Pick<User, "role" | "is_active">> }) =>
      api<User>(`/auth/users/${id}`, { method: "PATCH", json: patch }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.users }),
  });
}
