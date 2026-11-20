"""Saving and loading model bundles.

Security: joblib/pickle executes code on load, so ModelDoctor only ever loads
bundles it wrote itself. The path is derived from the model id, never from
user input, and user-uploaded model files are not accepted at all.
"""

from pathlib import Path

import joblib
import sklearn

BUNDLE_VERSION = 1


def artifact_name(model_id: str) -> str:
    return f"{model_id}.joblib"


def save_bundle(models_dir: Path, model_id: str, *, pipeline, config: dict, spec: dict,
                classes: list[str], positive_index: int | None) -> str:
    models_dir.mkdir(parents=True, exist_ok=True)
    name = artifact_name(model_id)
    joblib.dump(
        {
            "bundle_version": BUNDLE_VERSION,
            "sklearn_version": sklearn.__version__,
            "pipeline": pipeline,
            "config": config,
            "spec": spec,
            "classes": classes,
            "positive_index": positive_index,
        },
        models_dir / name,
        compress=3,
    )
    return name


def load_bundle(models_dir: Path, artifact: str) -> dict:
    path = models_dir / Path(artifact).name  # never follow directories from stored values
    return joblib.load(path)


def delete_bundle(models_dir: Path, artifact: str) -> None:
    (models_dir / Path(artifact).name).unlink(missing_ok=True)
