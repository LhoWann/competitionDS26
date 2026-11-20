"""Colab CLI job, step 4 (with run_job.sh --ckpt): uploads each finished run's fold checkpoints to a private Kaggle
dataset `<user>/dataslayer-ckpt-<run>` (files fold<k>.pt + config.yaml + cv*.json), creating it or adding a version.
The final Kaggle notebook predicts from these datasets; the checkpoints are too large for `colab download`."""
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ["KAGGLE_CONFIG_DIR"] = "/content"
user = json.loads(Path("/content/kaggle.json").read_text())["username"]
for run in sorted(p.parent for p in Path("/content/outputs").glob("*/config.yaml")):
    ckpts = sorted(run.glob("fold*/best.pt"))
    if not ckpts:
        continue
    stage = Path(f"/content/ckpt_upload/{run.name}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for c in ckpts:
        os.link(c, stage / f"{c.parent.name}.pt")
    for extra in ("config.yaml", "cv.json", "cv_flip.json"):
        if (run / extra).exists():
            shutil.copy(run / extra, stage / extra)
    slug = f"dataslayer-ckpt-{run.name.replace('_', '-')}"[:50]
    meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta))
    exists = subprocess.run(["kaggle", "datasets", "status", meta["id"]], capture_output=True).returncode == 0
    if exists:  # a version replaces every file: keep the folds saved by earlier jobs (e.g. folds 0-2 + now 3-4)
        old = stage.parent / f"{run.name}_old"
        subprocess.run(["kaggle", "datasets", "download", meta["id"], "-p", str(old), "--unzip", "-q"], check=True)
        for f in old.glob("fold*.pt"):
            if not (stage / f.name).exists():
                f.rename(stage / f.name)
    cmd = ["kaggle", "datasets", "version", "-m", "fold checkpoints"] if exists else ["kaggle", "datasets", "create"]
    subprocess.run([*cmd, "-p", str(stage), "-q"], check=True)
    print(f"SAVED {run.name} -> {meta['id']} ({len(list(stage.glob('fold*.pt')))} folds)", flush=True)
print("ALL SAVED", flush=True)