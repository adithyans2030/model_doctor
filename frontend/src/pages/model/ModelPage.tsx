import { Activity, Bot, FlaskConical, Gauge, GitCompareArrows, Lightbulb } from "lucide-react";
import { useParams, useSearchParams } from "react-router-dom";
import { Assistant } from "@/components/Assistant";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { PageHeader } from "@/components/domain/PageHeader";
import { StageBadge } from "@/components/domain/severity";
import { Badge, CardSkeleton, ErrorState, LoadingRegion, MetricTileSkeleton, Skeleton, TabPanel, Tabs } from "@/components/ui";
import { useModel } from "@/lib/queries";
import type { ModelConfig } from "@/lib/types";
import { ApprovalTab } from "./ApprovalTab";
import { ExperimentsTab } from "./ExperimentsTab";
import { ExplainTab } from "./ExplainTab";
import { MonitoringTab } from "./MonitoringTab";
import { PerformanceTab } from "./PerformanceTab";

const TABS = ["performance", "explain", "experiments", "approval", "monitoring", "assistant"] as const;
type Tab = (typeof TABS)[number];

export function describeConfig(c: ModelConfig): string {
  const parts = [c.algorithm.replaceAll("_", " ")];
  if (c.class_weight) parts.push(`class_weight=${c.class_weight}`);
  if (c.resampling) parts.push(c.resampling);
  if (c.dedupe) parts.push("deduplicated");
  if (c.drop_columns?.length) parts.push(`without ${c.drop_columns.join(", ")}`);
  if (c.calibration) parts.push(`${c.calibration} calibration`);
  if (c.threshold != null) parts.push(`threshold ${c.threshold}`);
  if (c.params && Object.keys(c.params).length) {
    parts.push(`tuned ${Object.entries(c.params).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toPrecision(3) : v}`).join(", ")}`);
  }
  return parts.join(" · ");
}

export default function ModelPage() {
  const { id = "" } = useParams();
  const model = useModel(id);
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab") === "registry" ? "approval" : params.get("tab") ?? "";
  const tab: Tab = (TABS as readonly string[]).includes(raw) ? (raw as Tab) : "performance";
  const setTab = (t: Tab) => setParams(t === "performance" ? {} : { tab: t }, { replace: true });

  if (model.isError) return <ErrorState error={model.error} onRetry={() => model.refetch()} />;
  if (model.isPending) {
    return (
      <LoadingRegion label="Loading model" className="space-y-5">
        <Skeleton className="h-3 w-56" /><Skeleton className="h-7 w-80" />
        <div className="grid gap-3 sm:grid-cols-4">{[0, 1, 2, 3].map((i) => <MetricTileSkeleton key={i} />)}</div>
        <CardSkeleton lines={6} />
      </LoadingRegion>
    );
  }
  const m = model.data;
  return (
    <div>
      <PageHeader
        crumbs={[
          { label: "Projects", to: "/app" }, { label: "Project", to: `/app/projects/${m.project_id}` },
          { label: "Dataset", to: `/app/datasets/${m.dataset_id}` }, { label: m.name },
        ]}
        title={m.name}
        meta={<>
          <StageBadge stage={m.stage} />
          <Badge>{m.source}</Badge>
          <span className="font-mono text-xs text-fg-3">{describeConfig(m.config)}</span>
        </>}
      />
      <Tabs label="Model views" idBase="model" value={tab} onChange={setTab} items={[
        { value: "performance", label: "Performance", icon: <Gauge size={14} />, count: m.findings.length || undefined },
        { value: "explain", label: "Explainability", icon: <Lightbulb size={14} /> },
        { value: "experiments", label: "Experiments", icon: <FlaskConical size={14} /> },
        { value: "approval", label: "Approval", icon: <GitCompareArrows size={14} /> },
        { value: "monitoring", label: "Monitoring", icon: <Activity size={14} /> },
        { value: "assistant", label: "Assistant", icon: <Bot size={14} /> },
      ]} />
      <TabPanel idBase="model" value={tab}>
        {tab === "performance" && <PerformanceTab m={m} />}
        {tab === "explain" && <ExplainTab m={m} />}
        {tab === "experiments" && <ExperimentsTab m={m} />}
        {tab === "approval" && <ApprovalTab m={m} />}
        {tab === "monitoring" && <MonitoringTab m={m} />}
        {tab === "assistant" && <ErrorBoundary label="the assistant"><Assistant datasetId={m.dataset_id} modelId={m.id} /></ErrorBoundary>}
      </TabPanel>
    </div>
  );
}
