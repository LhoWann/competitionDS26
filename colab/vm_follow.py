"""Colab CLI job, step 3: keeps the kernel busy (Colab keeps a VM alive while its kernel works) and prints stage and
epoch lines until the queue exits, then packs the outputs without checkpoints into /content/results.tgz. Safe to rerun
after a dropped connection."""
import re
import subprocess
import time
from pathlib import Path

QUEUE, LOG = "/content/run_queue.sh", Path("/content/outputs/train.log")
KEEP = re.compile(r"^(===|resume|fold \d|CV|\s*\d+\s+\d\.\d{4}\s|.*Error|Traceback)")
seen = 0
while True:
    lines = LOG.read_text(errors="replace").replace("\r", "\n").splitlines()
    for line in lines[seen:]:
        if KEEP.match(line):
            print(line[:220], flush=True)
    seen = len(lines)
    if subprocess.run(["pgrep", "-f", QUEUE], capture_output=True).returncode != 0:
        break
    time.sleep(60)
exits = re.findall(r"=== queue exit (\d+) ===", LOG.read_text(errors="replace"))
subprocess.run("tar --exclude=best.pt -czf /content/results.tgz -C /content outputs", shell=True, check=True)
print(f"QUEUE EXIT {exits[-1] if exits else '?'}", flush=True)