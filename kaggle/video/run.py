"""Kaggle 2 x T4 kernel: frame cache, YOLO11s person boxes (dataslayer-assets pose_cache.tgz) and one video model on
the folds in FOLDS + test (`configs/<CONFIG>.yaml`; split runs are merged afterwards). T4 has no bf16 -> fp16; the
per-GPU batch is half the config batch, so the effective batch on 2 GPUs matches the A100 recipe. Outputs incl.
fold*/best.pt in /kaggle/working/outputs/<run name>. Push: `kaggle kernels push -p kaggle/video`."""
import os
import subprocess
import tarfile
from pathlib import Path

import yaml

CONFIG = "colab_vjepa2_lora16mlp_e4_win"
FOLDS = "[0,1,2]"

comp = next(Path("/kaggle/input").rglob("sample_submission.csv")).parent
code = next(Path("/kaggle/input").rglob("src/core/config.py")).parents[2]
cfg = f"{code}/configs/{CONFIG}.yaml"
spec = yaml.safe_load(Path(cfg).read_text())
paths = f"paths.data_dir={comp} paths.cache_dir=/tmp/cache paths.output_dir=/kaggle/working/outputs"
over = (f"{paths} data.onsets={code}/{spec['data']['onsets']} data.num_workers={os.cpu_count()} "
        f"train.mixed_precision=fp16 train.batch_size={spec['train']['batch_size'] // 2} train.folds={FOLDS}")
env = os.environ | {"PYTHONPATH": f"{code}/src", "TQDM_MININTERVAL": "60"}
launch = "accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env, cwd="/kaggle/working")


sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"python -m cli.prepare --config {cfg} --set {paths} --workers {os.cpu_count()}")
feat = Path("/tmp/cache/features")  # noqa: S108 - single-user Kaggle VM
feat.mkdir(parents=True, exist_ok=True)
with tarfile.open(next(Path("/kaggle/input").rglob("pose_cache.tgz"))) as t:
    t.extractall(feat, filter="data")
sh(f"{launch} -m cli.train --config {cfg} --set {over}")
sh(f"{launch} -m cli.predict --config {cfg} --set {over}")
sh("ls -la /kaggle/working/outputs/*")