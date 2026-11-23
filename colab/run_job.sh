#!/bin/bash
# Trains + predicts configs on a Colab GPU through the official Colab CLI (run inside WSL, see README section 7):
#   bash colab/run_job.sh [--ckpt] configs/colab_vjepa2_win.yaml configs/colab_videomae_win.yaml
# Session $COLAB_SESSION (default ds; A100, L4 fallback) -> upload code, box cache, kaggle.json -> vm_setup /
# vm_start / vm_follow (each retried only after a dropped websocket; all idempotent) -> outputs/<run>/ in the repo
# (with --ckpt the fold checkpoints go to private Kaggle datasets, vm_save.py) -> VM stopped.
# Logs: ~/colab_runs/<time>/runner.log. START=vm_rescore.py replaces training by re-scoring saved checkpoints.
export PATH=$HOME/.local/bin:$PATH
C="colab --auth=adc"
S=${COLAB_SESSION:-ds}  # session name; one CLI job per machine at a time (the CLI's local state is shared)
REPO=$(cd "$(dirname "$0")/.." && pwd)
CKPT=0
[ "$1" = "--ckpt" ] && { CKPT=1; shift; }
RUNS="$*"
[ -z "$RUNS" ] && { echo "usage: run_job.sh [--ckpt] config.yaml ..."; exit 2; }
R=$HOME/colab_runs/$(date +%Y%m%d_%H%M)
mkdir -p $R
log() { echo "=== $* $(date) ===" | tee -a $R/runner.log; }

# A client whose websocket dropped can hang instead of exiting: every exec runs under `timeout` (seconds in $3).
# Dropped websockets and transient DNS / connection errors to the VM are retried (all steps are idempotent).
NET_ERR="Connection was lost|NameResolutionError|Max retries exceeded"
step() {
  for attempt in 1 2 3 4 5 6; do
    log "STEP $1 attempt $attempt"
    timeout $3 $C exec -s $S --timeout 0 --env "RUNS=$RUNS" --env "CKPT=$CKPT"       --env "EXTRA_SET=$EXTRA_SET" --env "TAG=$TAG" -f $REPO/colab/$1 > $R/$1.log 2>&1
    cat $R/$1.log >> $R/runner.log
    grep -q "$2" $R/$1.log && return 0
    grep -q -E "$NET_ERR" $R/$1.log || { log "STEP $1 FAILED"; return 1; }
    sleep 60
  done
  log "STEP $1 RETRIES EXHAUSTED"; return 1
}

stop_vm() { $C stop -s $S >> $R/runner.log 2>&1; log "VM STOPPED"; }

got=""
if ! $C status -s $S 2>/dev/null | grep -q "Hardware"; then
  for gpu in A100 A100 A100 L4; do
    if $C new -s $S --gpu $gpu >> $R/runner.log 2>&1; then got=$gpu; break; fi
    sleep 30
  done
  [ -z "$got" ] && { log "NO GPU"; exit 1; }
  log "GOT $got"
fi
# The first exec after `colab new` sometimes drops: warm the kernel up until a trivial exec answers.
for attempt in 1 2 3 4 5; do
  echo 'print("KERNEL OK")' | timeout 120 $C exec -s $S > $R/warmup.log 2>&1
  grep -q "KERNEL OK" $R/warmup.log && break
  log "WARMUP attempt $attempt failed"; sleep 20
done
# Never stop a VM this run did not create: its detached training queue may still be running.
grep -q "KERNEL OK" $R/warmup.log || { log "KERNEL NOT REACHABLE"; [ -n "$got" ] && stop_vm; exit 1; }
tar --exclude=__pycache__ -czf $R/code.tgz -C $REPO src configs
for f in $R/code.tgz $REPO/cache/pose_cache.tgz $REPO/secrets/kaggle.json; do
  ok=""
  for attempt in 1 2 3 4; do
    timeout 600 $C upload -s $S $f /content/$(basename $f) >> $R/runner.log 2>&1 && { ok=1; break; }
    log "UPLOAD $f attempt $attempt failed"; sleep 30
  done
  [ -n "$ok" ] || { log "UPLOAD $f FAILED"; stop_vm; exit 1; }
done
log "UPLOADED $RUNS"
step vm_setup.py "SETUP DONE" 3600 || { stop_vm; exit 1; }
step ${START:-vm_start.py} "queue" 600 || { stop_vm; exit 1; }
step vm_follow.py "QUEUE EXIT" 43200 || exit 1
grep -q "QUEUE EXIT 0" $R/vm_follow.py.log && log "QUEUE OK" || log "QUEUE FAILED"
$C download -s $S /content/results.tgz $R/results.tgz >> $R/runner.log 2>&1 && tar -xzf $R/results.tgz -C $REPO &&
  log "RESULTS IN $REPO/outputs"
if [ $CKPT = 1 ]; then
  # Checkpoints exist only on the VM: if the upload fails, leave the VM running for a manual rescue.
  step vm_save.py "ALL SAVED" 14400 || { log "CKPT SAVE FAILED: VM LEFT RUNNING"; exit 1; }
  log "CHECKPOINTS SAVED TO KAGGLE"
fi
stop_vm