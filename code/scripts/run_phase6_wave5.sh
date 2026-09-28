#!/usr/bin/env bash
# Phase 6, wave 5 — sweet-spot hyper-parameter matrix (detached, idempotent, logged).
#
# Every arm is 7200 steps of S5 (w288 d6 h4) + dropout, Muon, EMA 0.999, distilling from the phase-5
# teacher pair (the only pair that ever helped), T=4 alpha=0.7, seed 17, --eval-every 600
# --save-every 600 (which now also leaves per-point snapshots for scripts/select_best_checkpoint.py).
# Reference: e2-s17 = 1.4752 with the same recipe and dropout 0.1.
#
#   stage 0  wait for wave 4 (its script and every train.py gone), same guard as wave 4
#   stage A  hp-dr02  (dropout 0.2)  hp-wd005 (weight decay 0.05)
#   stage B  hp-lr003 (Muon lr 0.03) hp-dr03  (dropout 0.3)
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"
DISTILL="--distill-checkpoint $TEACHERS --distill-T 4.0 --distill-alpha 0.7"
WAIT_LIMIT=$((6 * 3600))

log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_and_score() {   # <name> <config> <extra...>
  local name="$1" cfg="$2"; shift 2
  local dir="$P/$name"
  if [ -f "$dir/metrics.json" ]; then
    log "already trained $name"
  else
    mkdir -p "$dir"
    local tcmd="$PY train.py --implementation student --config $cfg --steps 7200 --seed 17 --eval-every 600 --save-every 600 --optimizer muon --muon-lr 0.05 --ema-decay 0.999 $DISTILL $* --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
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

log "=== phase6 wave5 started (pid $$); waiting for wave 4 to clear the GPU ==="
waited=0
while pgrep -f run_phase6_wave4.sh >/dev/null 2>&1 || pgrep -f "train.py --implementation" >/dev/null 2>&1; do
  if [ "$waited" -ge "$WAIT_LIMIT" ]; then
    log "!! gave up waiting for wave 4 after ${WAIT_LIMIT}s; not starting wave 5"
    exit 1
  fi
  if [ $((waited % 600)) -eq 0 ]; then
    log "wave5 still waiting (${waited}s): wave4 script=$([ "$(pgrep -f run_phase6_wave4.sh >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down) trainings=$([ "$(pgrep -f 'train.py --implementation' >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down)"
  fi
  sleep 60
  waited=$((waited + 60))
done
log "GPU quiet after ${waited}s of waiting; starting wave 5"

# stage A: regularization first (the memory-wall question), then weight decay
train_and_score hp-dr02 configs/student_s5_dr02.json &
train_and_score hp-wd005 configs/student_s5_dr01.json --weight-decay 0.05 &
wait

# stage B: the lower-priority axes
train_and_score hp-lr003 configs/student_s5_dr01.json --muon-lr 0.03 &
train_and_score hp-dr03 configs/student_s5_dr03.json &
wait
log "=== phase6 wave5 complete ==="
