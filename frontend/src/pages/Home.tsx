import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Empty, ErrorBox, Loading, Section } from "@/components/legacy-ui";
import { api, fmt } from "@/lib/api";
import { useResource } from "@/lib/hooks";
import type { Project } from "@/lib/types";

const STEPS = ["Data Doctor", "Model Doctor", "Explain", "Optimise", "Approve & monitor"];

export default function Home() {
  const navigate = useNavigate();
  const projects = useResource<Project[]>("/projects");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function create(e: FormEvent) {
    e.preventDefault();
    try {
      const p = await api<Project>("/projects", { json: { name, description: description || null } });
      navigate(`/projects/${p.id}`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="space-y-4">
      <div className="card p-7">
        <h1>Upload a dataset and let ModelDoctor diagnose your ML system.</h1>
        <p className="mt-2 max-w-3xl text-fg-2">
          It profiles the data, trains baselines, finds out <b>why</b> a model underperforms, explains the root causes,
          runs corrective experiments automatically and hands you a better candidate model to approve.
        </p>
        <ol className="mt-4 flex flex-wrap items-center gap-2 text-[13px] text-fg-3">
          {STEPS.map((s, i) => (
            <li key={s} className="flex items-center gap-2">
              <span className="badge">{i + 1}</span>{s}{i < STEPS.length - 1 && <span aria-hidden>→</span>}
            </li>
          ))}
        </ol>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Section title="New project">
          <form className="space-y-3" onSubmit={create}>
            <input className="field w-full" placeholder="Project name, e.g. Customer churn" value={name}
              onChange={(e) => setName(e.target.value)} required maxLength={120} />
            <textarea className="field w-full resize-y" placeholder="Description (optional)" value={description}
              onChange={(e) => setDescription(e.target.value)} rows={2} maxLength={1000} />
            <button className="btn btn-primary" disabled={!name.trim()}>Create project</button>
            <ErrorBox error={error} />
          </form>
        </Section>
        <Section title="Projects">
          {projects.loading && !projects.data && <Loading />}
          <ErrorBox error={projects.error} />
          {projects.data?.length === 0 && <Empty>No projects yet. Create one to start.</Empty>}
          {!!projects.data?.length && (
            <table className="tbl">
              <thead><tr><th>Name</th><th className="num">Datasets</th><th>Created</th></tr></thead>
              <tbody>
                {projects.data.map((p) => (
                  <tr key={p.id}>
                    <td><Link to={`/projects/${p.id}`}>{p.name}</Link>{p.description && <div className="text-xs text-fg-3">{p.description}</div>}</td>
                    <td className="num">{p.dataset_count}</td>
                    <td className="text-xs text-fg-3">{fmt.date(p.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>
    </div>
  );
}
