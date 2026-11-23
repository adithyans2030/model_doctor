"""Run the whole ModelDoctor pipeline headlessly on a CSV.

    upload → target → Data Doctor → baselines → explanations → experiments → diagnosis

The full API runs in-process against a throwaway database, so this exercises
exactly the code the server runs. It writes JSON reports for DVC metrics, CI
checks or offline review.

Usage:
    python scripts/run_pipeline.py --csv ../sample_data/churn_demo.csv --target churned --out ../reports
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.main import create_app  # noqa: E402

API = "/api/v1"


def _ok(r, expected=200):
    if r.status_code != expected:
        raise SystemExit(f"{r.request.method} {r.request.url} -> {r.status_code}: {r.text[:500]}")
    return r.json()


def _job(client, job):
    job = _ok(client.get(f"{API}/jobs/{job['id']}"))
    if job["status"] != "completed":
        raise SystemExit(f"job failed: {job['error']}")
    return job


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="defaults to the standard algorithms for the target's task")
    p.add_argument("--budget", choices=["quick", "standard"], default="standard")
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        app = create_app(Settings(_env_file=None, data_dir=Path(tmp)))
        Base.metadata.create_all(app.state.engine)
        with TestClient(app) as c:
            project = _ok(c.post(f"{API}/projects", json={"name": args.csv.stem}), 201)
            with open(args.csv, "rb") as f:
                ds = _ok(c.post(f"{API}/projects/{project['id']}/datasets",
                                files={"file": (args.csv.name, f, "text/csv")}), 201)
            target = _ok(c.post(f"{API}/datasets/{ds['id']}/target", json={"column": args.target}))
            print(f"[1/5] {ds['n_rows']} rows, task {target['dataset']['task_type']}")

            data_report = _ok(c.get(f"{API}/datasets/{ds['id']}/data-report"))
            print(f"[2/5] Data Doctor: score {data_report['score']['value']}, {data_report['summary']['severity_counts']}")

            training = _job(c, _ok(c.post(f"{API}/datasets/{ds['id']}/train",
                                          json={"algorithms": args.algorithms} if args.algorithms else {}), 202))
            best_id = training["result"]["best_model_id"]
            print(f"[3/5] Baselines: best {training['result']['leaderboard'][0]['name']} "
                  f"{training['result']['primary_metric']} {training['result']['leaderboard'][0]['primary']}")

            explanations = _ok(c.get(f"{API}/models/{best_id}/explanations"))
            print(f"[4/5] Explanations: {len(explanations['findings'])} prediction findings")

            experiment = _job(c, _ok(c.post(f"{API}/models/{best_id}/experiments", json={"budget": args.budget}), 202))
            res = experiment["result"]
            final_id = res["best_model_id"] or best_id
            diagnosis = _ok(c.get(f"{API}/datasets/{ds['id']}/diagnosis", params={"model_id": final_id}))
            print(f"[5/5] Experiments: {len(res['trials'])} trials, best {res['best']['description']}")
        app.state.engine.dispose()  # release the SQLite file (Windows can't delete open files)

    args.out.mkdir(parents=True, exist_ok=True)
    best_trial = res["best"]["metrics"]
    metrics = {
        "data_quality_score": data_report["score"]["value"],
        "task": training["result"]["task"],
        "primary_metric": training["result"]["primary_metric"],
        "baseline_primary": training["result"]["leaderboard"][0]["primary"],
        "leak_free_baseline_primary": res["baseline_primary"],
        "optimised_primary": best_trial["primary"],
        **({"optimised_minority_recall": best_trial["minority_recall"]} if "minority_recall" in best_trial
           else {"optimised_rmse": best_trial["rmse"]}),
        "health_score": diagnosis["health"]["overall"],
        "critical_findings": diagnosis["severity_counts"]["critical"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2, default=str))
    (args.out / "data_report.json").write_text(json.dumps(data_report, indent=2, default=str))
    (args.out / "experiment.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n".join(diagnosis["narrative"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
