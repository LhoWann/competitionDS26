"""Colab CLI job, step 2: starts the train + predict queue of the configs in $RUNS detached from the kernel (no-op if it
already runs). Each GPU gets the largest micro-batch that divides the config batch and fits in 85% of its memory, with
grad_accum making up the rest, so the effective batch (and the measured recipe) is the same on L4 and A100."""
import os
import subprocess
from pathlib import Path

import torch
import yaml

RUNS = os.environ["RUNS"].split()
MEM_GB = {"vjepa2": (0.43, 2.1), "videomae": (0.6, 1.5)}  # measured training memory: (GB per clip, fixed GB)
QUEUE, LOG = "/content/run_queue.sh", "/content/outputs/train.log"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30


def overrides(config: str) -> str:
    spec = yaml.safe_load(Path(f"/content/{config}").read_text())
    batch = spec["train"]["batch_size"]
    per_clip, fixed = MEM_GB[spec["model"]["kind"]]
    micro = max(d for d in range(1, batch + 1) if batch % d == 0 and fixed + d * per_clip <= 0.85 * gpu_gb)
    extra = f"data.num_workers={os.cpu_count()}"
    return extra + (f" train.batch_size={micro} train.grad_accum={batch // micro}" if micro < batch else "")


if subprocess.run(["pgrep", "-f", QUEUE], capture_output=True).returncode == 0:
    print("queue already running")
else:
    steps = []
    for config in RUNS:
        o = overrides(config)
        steps += [f"echo '=== {config}: train ==='", f"python -m cli.train --config {config} --set {o}",
                  f"echo '=== {config}: predict ==='", f"python -m cli.predict --config {config} --set {o}"]
    with open(QUEUE, "w") as f:
        f.write("set -e\n" + "\n".join(steps) + "\n")
    os.makedirs("/content/outputs", exist_ok=True)
    env = os.environ | {"PYTHONPATH": "/content/src", "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "30"}
    subprocess.Popen(f"bash {QUEUE} >> {LOG} 2>&1; echo \"=== queue exit $? ===\" >> {LOG}", shell=True,
                     cwd="/content", env=env, start_new_session=True)
    print("queue started:", [overrides(c) for c in RUNS])