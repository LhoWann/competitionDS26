"""Colab CLI job, step 2 with `START=vm_rescore.py bash colab/run_job.sh <configs>`: re-scores finished video runs with
every frame of each clip (FAQ 66/70: data.eval_passes=2, person boxes from the stride-1 YOLO11s cache) instead of
training. Per config the fold checkpoints come from the run's private Kaggle dataset (vm_save.py); the detached queue
writes outputs/<run>_<$TAG, default a30>/ (config.yaml, fold + pooled OOF, cv*.json, test probabilities); $EXTRA_SET
is applied after the defaults (e.g. TAG=t30 EXTRA_SET="data.num_frames=30 data.eval_passes=1": one 30-frame pass).
No-op if the queue already runs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/content/src")
from core.config import load_config  # noqa: E402

RUNS = os.environ["RUNS"].split()
QUEUE, LOG = "/content/run_queue.sh", "/content/outputs/train.log"
SET = f"data.eval_passes=2 data.crop_features=yolo11s-pose_360x640_s1 {os.environ.get('EXTRA_SET', '')}".strip()
TAG = os.environ.get("TAG") or "a30"
user = json.loads(Path("/content/kaggle.json").read_text())["username"]

if subprocess.run(["pgrep", "-f", QUEUE], capture_output=True).returncode == 0:
    print("queue already running")
else:
    steps = []
    for config in RUNS:
        run = yaml.safe_load(Path(f"/content/{config}").read_text())["name"]
        out, dl = Path(f"/content/outputs/{run}_{TAG}"), f"/content/ckpt/{run}"
        o = f"name={run}_{TAG} {SET} data.num_workers={os.cpu_count()}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(load_config(f"/content/{config}", o.split()).to_dict()))
        slug = f"dataslayer-ckpt-{run.replace('_', '-')}"[:50]
        steps += [f"echo '=== {config}: checkpoints ==='", f"kaggle datasets download {user}/{slug} -p {dl} --unzip -q",
                  f"for k in 0 1 2 3 4; do mkdir -p {out}/fold$k && mv {dl}/fold$k.pt {out}/fold$k/best.pt; done",
                  f"echo '=== {config}: oof ==='", f"python -m cli.predict --config {config} --set {o} --oof",
                  f"echo '=== {config}: test ==='", f"python -m cli.predict --config {config} --set {o}",
                  f"rm -rf {dl}"]
    with open(QUEUE, "w") as f:
        f.write("set -e\n" + "\n".join(steps) + "\n")
    env = os.environ | {"PYTHONPATH": "/content/src", "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "30",
                        "KAGGLE_CONFIG_DIR": "/content"}
    subprocess.Popen(f"bash {QUEUE} >> {LOG} 2>&1; echo \"=== queue exit $? ===\" >> {LOG}", shell=True,
                     cwd="/content", env=env, start_new_session=True)
    print("queue started:", RUNS)