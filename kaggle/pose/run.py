"""Kaggle GPU kernel: frame cache, YOLO-pose keypoint cache and the window-label pose head, 5 folds + test
(`configs/<CONFIG>.yaml`). Inputs: competition data and the `dataslayer-src` dataset (src/, configs/).
Outputs in /kaggle/working/outputs/<run name> (oof.csv, test_probs.csv, cv.json, fold*/best.pt).
Push: `kaggle kernels push -p kaggle/pose` (the laptop runs the same steps, README section 7)."""
import os
import subprocess
from pathlib import Path

import yaml

CONFIG = "feat_pose"

comp = next(Path("/kaggle/input").rglob("sample_submission.csv")).parent
code = next(Path("/kaggle/input").rglob("src/core/config.py")).parents[2]
cfg = f"{code}/configs/{CONFIG}.yaml"
paths = f"paths.data_dir={comp} paths.cache_dir=/tmp/cache paths.output_dir=/kaggle/working/outputs"
data = yaml.safe_load(Path(cfg).read_text())["data"]  # label tables are repo-relative: point them into the dataset
over = " ".join([paths] + [f"data.{k}={code}/{data[k]}" for k in ("onsets", "neutral_secs") if data.get(k)])
env = os.environ | {"PYTHONPATH": f"{code}/src", "TQDM_MININTERVAL": "60"}


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env, cwd="/kaggle/working")


sh("pip -q install ultralytics")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"python -m cli.prepare --config {cfg} --set {paths} --workers {os.cpu_count()}")
sh(f"python -m cli.extract --config {cfg} --set {paths} --workers {os.cpu_count()}")
sh(f"python -m cli.train --config {cfg} --set {over}")
sh(f"python -m cli.predict --config {cfg} --set {over}")
sh("ls -la /kaggle/working/outputs/*")