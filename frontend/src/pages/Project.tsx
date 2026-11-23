import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Crumbs, Empty, ErrorBox, Loading, Section, StageBadge } from "@/components/legacy-ui";
import { api, fmt } from "@/lib/api";
import { useResource } from "@/lib/hooks";
import type { Dataset, ModelSummary, Project } from "@/lib/types";

type Registry = {
  production: ModelSummary | null;
  models: ModelSummary[];
  events: { id: string; model_id: string; from_stage: string | null; to_stage: string; actor: string; note: string | null; created_at: string }[];
};

export default function ProjectPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const project = useResource<Project>(`/projects/${id}`);
  const datasets = useResource<Dataset[]>(`/projects/${id}/datasets`);
  const registry = useResource<Registry>(`/projects/${id}/registry`);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const ds = await api<Dataset>(`/projects/${id}/datasets`, { form });
      navigate(`/datasets/${ds.id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm("Delete this project, its datasets and models?")) return;
    await api(`/projects/${id}`, { method: "DELETE" });
    navigate("/");
  }

  if (project.error) return <ErrorBox error={project.error} />;
  if (!project.data) return <Loading />;
  const names = Object.fromEntries((registry.data?.models || []).map((m) => [m.id, m.name]));

  return (
    <div className="space-y-4">
      <Crumbs><Link to="/">Projects</Link> / <span>{project.data.name}</span></Crumbs>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1>{project.data.name}</h1>
          {project.data.description && <p className="mt-1 text-fg-2">{project.data.description}</p>}
        </div>
        <button className="btn btn-danger" onClick={remove}>Delete project</button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Section title="Upload a dataset">
          <form className="space-y-3" onSubmit={upload}>
            <input type="file" accept=".csv,text/csv" className="block text-[13px] file:mr-3 file:cursor-pointer file:rounded-lg file:border file:border-line file:bg-surface-2 file:px-3 file:py-1.5 file:text-fg"
              onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <p className="text-xs text-fg-3">CSV only (comma, semicolon, tab or pipe). Model files are never accepted: ModelDoctor trains its own models.</p>
            <button className="btn btn-primary" disabled={!file || busy}>{busy ? "Uploading…" : "Upload & profile"}</button>
            <ErrorBox error={error} />
          </form>
        </Section>
        <Section title="Datasets">
          {!datasets.data ? <Loading /> : datasets.data.length === 0 ? <Empty>No datasets yet.</Empty> : (
            <table className="tbl">
              <thead><tr><th>File</th><th className="num">Rows</th><th>Target</th><th>Task</th></tr></thead>
              <tbody>
                {datasets.data.map((d) => (
                  <tr key={d.id}>
                    <td><Link to={`/datasets/${d.id}`}>{d.original_filename}</Link></td>
                    <td className="num">{fmt.int(d.n_rows)}</td>
                    <td>{d.target_column || <span className="text-fg-3">not set</span>}</td>
                    <td className="text-xs">{d.task_type?.replace("_", " ") || "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>

      <Section title="Model registry">
        {!registry.data ? <Loading /> : registry.data.models.length === 0 ? <Empty>No models yet. Train baselines from a dataset.</Empty> : (
          <div className="space-y-3">
            <div className="notice text-[13px]">
              Production model:{" "}
              {registry.data.production ? <Link to={`/models/${registry.data.production.id}`}>{registry.data.production.name}</Link> : "none yet"}.
              {" "}Models move candidate → approved (human approval) → production.
            </div>
            <div className="overflow-x-auto">
              <table className="tbl">
                <thead><tr><th>Version</th><th>Model</th><th>Stage</th><th>Source</th><th className="num">Macro-F1</th><th className="num">Balanced acc.</th><th className="num">Minority recall</th><th className="num">Findings</th></tr></thead>
                <tbody>
                  {registry.data.models.map((m) => (
                    <tr key={m.id}>
                      <td>v{m.version}</td>
                      <td><Link to={`/models/${m.id}`}>{m.name}</Link></td>
                      <td><StageBadge stage={m.stage} /></td>
                      <td className="text-xs">{m.source}</td>
                      <td className="num">{fmt.num(m.scores.f1_macro)}</td>
                      <td className="num">{fmt.num(m.scores.balanced_accuracy)}</td>
                      <td className="num">{fmt.pct(m.scores.worst_class?.recall)}</td>
                      <td className="num">{m.n_findings}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {registry.data.events.length > 0 && (
              <details>
                <summary className="cursor-pointer text-fg-2">Audit trail ({registry.data.events.length})</summary>
                <table className="tbl mt-1.5"><tbody>
                  {registry.data.events.map((e) => (
                    <tr key={e.id}>
                      <td className="text-xs text-fg-3">{fmt.date(e.created_at)}</td>
                      <td>{names[e.model_id] || e.model_id.slice(0, 8)}</td>
                      <td className="text-xs">{e.from_stage} → <b>{e.to_stage}</b></td>
                      <td className="text-xs">{e.actor}</td>
                      <td className="text-xs text-fg-2">{e.note}</td>
                    </tr>
                  ))}
                </tbody></table>
              </details>
            )}
          </div>
        )}
      </Section>
    </div>
  );
}
