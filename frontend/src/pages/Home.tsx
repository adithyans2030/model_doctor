import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Database, FolderPlus, Plus, Sparkles } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/domain/PageHeader";
import { QueryView } from "@/components/domain/QueryView";
import {
  Button, Card, EmptyState, InlineError, Modal, TBody, TD, TH, THead, TR, Table, TableSkeleton, useToast,
} from "@/components/ui";
import { api, fmt } from "@/lib/api";
import { qk, useCreateProject, useProjects } from "@/lib/queries";
import type { Dataset, Project } from "@/lib/types";

/** Creates a project from the bundled synthetic churn dataset, using only the public API. */
function useLoadDemo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const csv = await fetch("/demo/churn_demo.csv");
      if (!csv.ok) throw new Error("The demo dataset isn't bundled with this build.");
      const file = new File([await csv.blob()], "churn_demo.csv", { type: "text/csv" });
      const project = await api<Project>("/projects", {
        json: { name: "Customer churn (demo)", description: "Synthetic dataset with 9 planted problems: sample_data/churn_demo.csv" },
      });
      const form = new FormData();
      form.append("file", file);
      const ds = await api<Dataset>(`/projects/${project.id}/datasets`, { form });
      await api(`/datasets/${ds.id}/target`, { json: { column: "churned" } });
      return ds;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  });
}

export default function Home() {
  const navigate = useNavigate();
  const toast = useToast();
  const projects = useProjects();
  const create = useCreateProject();
  const demo = useLoadDemo();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    const p = await create.mutateAsync({ name: name.trim(), description: description.trim() || null }).catch(() => null);
    if (p) {
      toast({ title: `Created ${p.name}`, tone: "success" });
      navigate(`/app/projects/${p.id}`);
    }
  }

  async function loadDemo() {
    const ds = await demo.mutateAsync().catch(() => null);
    if (ds) {
      toast({ title: "Demo dataset loaded", description: "Target set to 'churned'. Run the Data Doctor next.", tone: "success" });
      navigate(`/app/datasets/${ds.id}?tab=data`);
    }
  }

  const newProject = <Button variant="primary" icon={<Plus size={15} />} onClick={() => setOpen(true)}>New project</Button>;

  return (
    <div className="space-y-6">
      <PageHeader title="Projects" meta={<span>Upload a dataset, diagnose its models, test fixes, approve the winner.</span>} actions={newProject} />

      <section aria-labelledby="demo-title" className="flex flex-col gap-4 rounded-lg border border-line bg-surface p-5 sm:flex-row sm:items-center">
        <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-accent-subtle text-accent-fg"><Sparkles size={18} aria-hidden /></div>
        <div className="min-w-0 flex-1">
          <h2 id="demo-title" className="text-sm font-semibold">Try it on the demo dataset</h2>
          <p className="mt-0.5 text-sm text-fg-2">
            5,050 synthetic customers with 9 planted problems: a leaking column, class imbalance, missing values, duplicates and more.
          </p>
          <InlineError error={demo.error} className="mt-2" />
        </div>
        <Button onClick={loadDemo} loading={demo.isPending} icon={<Database size={15} />}>
          {demo.isPending ? "Uploading and profiling…" : "Load demo dataset"}
        </Button>
      </section>

      <Card title="All projects" flush>
        <QueryView query={projects} label="Loading projects" skeleton={<TableSkeleton rows={3} cols={3} />}
          isEmpty={(d) => d.length === 0}
          empty={<EmptyState icon={FolderPlus} title="No projects yet" description="A project groups datasets, models and their approval history." action={<Button icon={<Plus size={15} />} onClick={() => setOpen(true)}>Create your first project</Button>} />}>
          {(list) => (
            <Table label="Projects">
              <THead><TR><TH>Name</TH><TH numeric>Datasets</TH><TH>Created</TH><TH><span className="sr-only">Open</span></TH></TR></THead>
              <TBody>
                {list.map((p) => (
                  <TR key={p.id}>
                    <TD>
                      <Link to={`/app/projects/${p.id}`} className="font-medium text-fg">{p.name}</Link>
                      {p.description && <div className="mt-0.5 text-xs text-fg-3">{p.description}</div>}
                    </TD>
                    <TD numeric>{p.dataset_count}</TD>
                    <TD className="whitespace-nowrap text-xs text-fg-3">{fmt.date(p.created_at)}</TD>
                    <TD className="w-8 text-fg-3"><ArrowRight size={14} aria-hidden /></TD>
                  </TR>
                ))}
              </TBody>
            </Table>
          )}
        </QueryView>
      </Card>

      <Modal open={open} onClose={() => setOpen(false)} title="New project" description="Projects hold datasets, trained models and the approval trail.">
        <form id="new-project" className="space-y-3" onSubmit={submit}>
          <div>
            <label htmlFor="project-name" className="mb-1 block text-sm font-medium">Name</label>
            <input id="project-name" className="field w-full" placeholder="Project name, e.g. Customer churn" value={name}
              onChange={(e) => setName(e.target.value)} required maxLength={120} autoFocus />
          </div>
          <div>
            <label htmlFor="project-desc" className="mb-1 block text-sm font-medium">Description <span className="font-normal text-fg-3">(optional)</span></label>
            <textarea id="project-desc" className="field w-full resize-y" rows={2} value={description}
              onChange={(e) => setDescription(e.target.value)} maxLength={1000} />
          </div>
          <InlineError error={create.error} />
          <div className="flex justify-end gap-2 pt-1">
            <Button onClick={() => setOpen(false)}>Cancel</Button>
            <Button type="submit" variant="primary" loading={create.isPending} disabled={!name.trim()}>Create project</Button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
