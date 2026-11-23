import { Activity, Bot, Boxes, Stethoscope, Table2, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Assistant } from "@/components/Assistant";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Meta, PageHeader } from "@/components/domain/PageHeader";
import {
  Badge, Button, CardSkeleton, ConfirmDialog, EmptyState, ErrorState, InlineError, LoadingRegion, MetricTileSkeleton, Skeleton,
  TabPanel, Tabs, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { useDataset, useDeleteDataset } from "@/lib/queries";
import { DataDoctorTab } from "./DataDoctorTab";
import { DiagnosisTab } from "./DiagnosisTab";
import { ModelsTab } from "./ModelsTab";
import { OverviewTab } from "./OverviewTab";

const TABS = ["overview", "data", "models", "diagnosis", "assistant"] as const;
type Tab = (typeof TABS)[number];

export default function DatasetPage() {
  const { id = "" } = useParams();
  const ds = useDataset(id);
  const [params, setParams] = useSearchParams();
  const tab: Tab = (TABS as readonly string[]).includes(params.get("tab") ?? "") ? (params.get("tab") as Tab) : "overview";
  const setTab = (t: Tab) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const navigate = useNavigate();
  const toast = useToast();
  const remove = useDeleteDataset();
  const [confirm, setConfirm] = useState(false);

  if (ds.isError) return <ErrorState error={ds.error} onRetry={() => ds.refetch()} />;
  if (ds.isPending) {
    return (
      <LoadingRegion label="Loading dataset" className="space-y-5">
        <Skeleton className="h-3 w-48" /><Skeleton className="h-7 w-72" />
        <div className="grid gap-3 sm:grid-cols-4">{[0, 1, 2, 3].map((i) => <MetricTileSkeleton key={i} />)}</div>
        <CardSkeleton lines={6} />
      </LoadingRegion>
    );
  }
  const d = ds.data;
  const needsTarget = !d.target_column;
  const targetFirst = (
    <EmptyState icon={Stethoscope} title="Choose a prediction target first"
      description="Pick the column to predict on the Overview tab. Training, diagnosis and the assistant need it."
      action={<Button onClick={() => setTab("overview")}>Go to Overview</Button>} />
  );

  return (
    <div>
      <PageHeader
        crumbs={[{ label: "Projects", to: "/app" }, { label: "Project", to: `/app/projects/${d.project_id}` }, { label: d.original_filename }]}
        title={<span className="font-mono">{d.original_filename}</span>}
        meta={<>
          <Meta label="rows">{fmt.int(d.n_rows)}</Meta>
          <Meta label="columns">{d.n_cols}</Meta>
          <Meta label="target">{d.target_column ?? <span className="text-fg-3">not set</span>}</Meta>
          {d.task_type && <Badge tone="accent">{d.task_type.replace("_", " ")}</Badge>}
        </>}
        actions={<Button variant="danger" size="sm" icon={<Trash2 size={14} />} onClick={() => setConfirm(true)}>Delete dataset</Button>}
      />
      <Tabs label="Dataset views" idBase="dataset" value={tab} onChange={setTab} items={[
        { value: "overview", label: "Overview", icon: <Table2 size={14} /> },
        { value: "data", label: "Data Doctor", icon: <Activity size={14} /> },
        { value: "models", label: "Models", icon: <Boxes size={14} /> },
        { value: "diagnosis", label: "Diagnosis", icon: <Stethoscope size={14} /> },
        { value: "assistant", label: "Assistant", icon: <Bot size={14} /> },
      ]} />
      <TabPanel idBase="dataset" value={tab}>
        {tab === "overview" && <OverviewTab d={d} />}
        {tab === "data" && <DataDoctorTab id={id} key={d.target_column ?? "none"} />}
        {tab === "models" && (needsTarget ? targetFirst : <ModelsTab d={d} />)}
        {tab === "diagnosis" && (needsTarget ? targetFirst : <DiagnosisTab id={id} />)}
        {tab === "assistant" && (needsTarget ? targetFirst : <ErrorBoundary label="the assistant"><Assistant datasetId={id} /></ErrorBoundary>)}
      </TabPanel>

      <ConfirmDialog open={confirm} title="Delete this dataset?" confirmLabel="Delete dataset" busy={remove.isPending}
        description="Its models, experiments, explanations and drift reports are deleted too. This can't be undone."
        onCancel={() => setConfirm(false)}
        onConfirm={async () => {
          const ok = await remove.mutateAsync(d).then(() => true, () => false);
          if (ok) { toast({ title: `Deleted ${d.original_filename}`, tone: "success" }); navigate(`/app/projects/${d.project_id}`); }
        }}>
        <InlineError error={remove.error} />
      </ConfirmDialog>
    </div>
  );
}
