"""Colab CLI job, step 1 (idempotent): code from /content/up, competition data from the Kaggle API, frame cache, YOLO11s
person-box cache. Run by colab/run_job.sh through `colab exec`."""
import os
import subprocess
from pathlib import Path

UP = "/content/up"
os.environ["KAGGLE_CONFIG_DIR"] = f"{UP}/secrets"
os.environ["PYTHONPATH"] = "/content/src"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/content")


sh(f"rm -rf src configs && tar -xzf {UP}/code.tgz")
sh('pip -q install timm accelerate kaggle "transformers>=4.53"')
if not Path("/content/data/sample_submission.csv").exists():
    if not Path("/content/data-slayer-4.zip").exists():
        sh("kaggle competitions download -c data-slayer-4 -p /content")
    sh("rm -rf data data_tmp && unzip -q -o data-slayer-4.zip -d data_tmp && mv data_tmp data && rm data-slayer-4.zip")
if not Path("/content/cache/videos.csv").exists():
    sh(f"python -m cli.prepare --config {os.environ['RUNS'].split()[0]} --workers {os.cpu_count()}")
if not Path("/content/cache/features/yolo11s-pose_360x640_s2/test").exists():
    sh(f"mkdir -p cache/features && tar -xzf {UP}/pose_cache.tgz -C cache/features")
sh("ls data && ls cache/frames | wc -l && ls cache/features/yolo11s-pose_360x640_s2/train | wc -l")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
print("SETUP DONE", flush=True)