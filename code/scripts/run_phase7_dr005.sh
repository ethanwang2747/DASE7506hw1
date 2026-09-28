#!/usr/bin/env bash
# Local hedge probe: dropout 0.05 on the S3-champion recipe (S7 @28800, wd0.05, muon0.03, ema0.9995,
# old teacher pair, cosine). Motivation: wd 0.1->0.05 = -0.024 and dr 0.1->0.2 = +0.030 both say the
# S7/long-plan regime is over-regularised; dr 0.1->0.05 is the untested cell on the same axis.
# Paired seeds 17/42; idempotent; detached-friendly; no wait-loop (GPU is free).
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase7
LOG=$P/commands.log
TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"

mkdir -p "$P"
log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_and_score() {   # <name> <seed>
  local name="$1" seed="$2"
  local dir="$P/$name"
  if [ -f "$dir/metrics.json" ]; then
    log "already trained $name"
  else
    mkdir -p "$dir"
    local tcmd="$PY train.py --implementation student --config configs/student_s7_dr005.json --steps 28800 --seed $seed --eval-every 1200 --save-every 1200 --optimizer muon --muon-lr 0.03 --ema-decay 0.9995 --weight-decay 0.05 --distill-checkpoint $TEACHERS --distill-T 4.0 --distill-alpha 0.7 --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
    log "$tcmd"
    if ! eval "$tcmd" >"$P/$name.train.out" 2>&1; then
      log "!! training failed for $name"; return 1
    fi
    log "trained $name"
  fi
  if [ -f "$dir/validation_cpu_fp32.json" ]; then log "already scored $name"; return 0; fi
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4"
  log "$ecmd"
  if eval "$ecmd" >"$P/$name.eval.out" 2>&1; then log "finished $name"; else log "!! scoring failed for $name"; fi
}

log "=== phase7 dr005 probe started (pid $$) ==="
train_and_score s7x-dr005-s17 17 &
train_and_score s7x-dr005-s42 42 &
wait
log "=== phase7 dr005 probe complete ==="
