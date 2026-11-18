#!/bin/bash
# Trains + predicts configs on a Colab GPU through the official Colab CLI (run inside WSL, see README section 7):
#   bash colab/run_job.sh [--ckpt] configs/colab_vjepa2_win.yaml configs/colab_videomae_win.yaml
# Session `ds` (A100, L4 fallback) -> upload code, YOLO11s box cache, kaggle.json -> vm_setup / vm_start / vm_follow
# (each retried only after a dropped websocket; all three are idempotent) -> outputs/<run>/ extracted into the repo
# (with --ckpt also fold*/best.pt) -> VM stopped. Logs: ~/colab_runs/<time>/runner.log.
export PATH=$HOME/.local/bin:$PATH
C="colab --auth=adc"
REPO=$(cd "$(dirname "$0")/.." && pwd)
CKPT=0
[ "$1" = "--ckpt" ] && { CKPT=1; shift; }
RUNS="$*"
[ -z "$RUNS" ] && { echo "usage: run_job.sh [--ckpt] config.yaml ..."; exit 2; }
R=$HOME/colab_runs/$(date +%Y%m%d_%H%M)
mkdir -p $R
log() { echo "=== $* $(date) ===" | tee -a $R/runner.log; }

step() {
  for attempt in 1 2 3 4 5 6; do
    log "STEP $1 attempt $attempt"
    $C exec -s ds --timeout 0 --env "RUNS=$RUNS" --env "CKPT=$CKPT" -f $REPO/colab/$1 > $R/$1.log 2>&1
    cat $R/$1.log >> $R/runner.log
    grep -q "$2" $R/$1.log && return 0
    grep -q "Connection was lost" $R/$1.log || { log "STEP $1 FAILED"; return 1; }
    sleep 60
  done
  log "STEP $1 RETRIES EXHAUSTED"; return 1
}

stop_vm() { $C stop -s ds >> $R/runner.log 2>&1; log "VM STOPPED"; }

if ! $C status -s ds 2>/dev/null | grep -q "Hardware"; then
  got=""
  for gpu in A100 A100 A100 L4; do
    if $C new -s ds --gpu $gpu >> $R/runner.log 2>&1; then got=$gpu; break; fi
    sleep 30
  done
  [ -z "$got" ] && { log "NO GPU"; exit 1; }
  log "GOT $got"
fi
tar --exclude=__pycache__ -czf $R/code.tgz -C $REPO src configs
echo 'import os; os.makedirs("/content/up/secrets", exist_ok=True)' | $C exec -s ds >> $R/runner.log 2>&1
$C upload -s ds $R/code.tgz /content/up/code.tgz >> $R/runner.log 2>&1
$C upload -s ds $REPO/cache/pose_cache.tgz /content/up/pose_cache.tgz >> $R/runner.log 2>&1
$C upload -s ds $REPO/secrets/kaggle.json /content/up/secrets/kaggle.json >> $R/runner.log 2>&1
log "UPLOADED $RUNS"
step vm_setup.py "SETUP DONE" || { stop_vm; exit 1; }
step vm_start.py "queue" || { stop_vm; exit 1; }
step vm_follow.py "QUEUE EXIT" || exit 1
grep -q "QUEUE EXIT 0" $R/vm_follow.py.log && log "QUEUE OK" || log "QUEUE FAILED"
$C download -s ds /content/results.tgz $R/results.tgz >> $R/runner.log 2>&1 && tar -xzf $R/results.tgz -C $REPO &&
  log "RESULTS IN $REPO/outputs"
if [ $CKPT = 1 ]; then
  $C download -s ds /content/ckpt.tar $R/ckpt.tar >> $R/runner.log 2>&1 && tar -xf $R/ckpt.tar -C $REPO &&
    rm $R/ckpt.tar && log "CHECKPOINTS IN $REPO/outputs"
fi
stop_vm