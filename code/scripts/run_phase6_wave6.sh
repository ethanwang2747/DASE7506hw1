#!/usr/bin/env bash
# Phase 6, wave 6 — the two highest-expected-value arms (detached, idempotent, logged).
#
#   A  e2x-s17 / e2x-s42   the missing cell of the teacher x steps 2x2: the phase-5 teacher pair at
#                          14400 steps (extrapolated ~1.460, the most confident missing number)
#   B  s7x-dr02-s17/s42    the never-tested "big + heavily regularised + long" corner: S7 (w336 d6 h6,
#                          ~8.9 M) with dropout 0.2 at 14400 steps
#
# Both groups: old teacher pair, T=4 alpha=0.7, Muon 0.05, EMA 0.9995, --eval-every 1200
# --save-every 1200 (per-point snapshots kept), scored afterwards on CPU fp32 validation.
# Waits for wave 5 to clear the GPU first (same guard as waves 4 and 5).
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"
DISTILL="--distill-checkpoint $TEACHERS --distill-T 4.0 --distill-alpha 0.7"
WAIT_LIMIT=$((8 * 3600))

log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_and_score() {   # <name> <config> <seed> <extra...>
  local name="$1" cfg="$2" seed="$3"; shift 3
  local dir="$P/$name"
  if [ -f "$dir/metrics.json" ]; then
    log "already trained $name"
  else
    mkdir -p "$dir"
    local tcmd="$PY train.py --implementation student --config $cfg --steps 14400 --seed $seed --eval-every 1200 --save-every 1200 --optimizer muon --muon-lr 0.05 --ema-decay 0.9995 $DISTILL $* --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
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

log "=== phase6 wave6 started (pid $$); waiting for wave 5 to clear the GPU ==="
waited=0
while pgrep -f run_phase6_wave5.sh >/dev/null 2>&1 || pgrep -f "train.py --implementation" >/dev/null 2>&1; do
  if [ "$waited" -ge "$WAIT_LIMIT" ]; then
    log "!! gave up waiting for wave 5 after ${WAIT_LIMIT}s; not starting wave 6"
    exit 1
  fi
  if [ $((waited % 600)) -eq 0 ]; then
    log "wave6 still waiting (${waited}s): wave5 script=$([ "$(pgrep -f run_phase6_wave5.sh >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down) trainings=$([ "$(pgrep -f 'train.py --implementation' >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down)"
  fi
  sleep 60
  waited=$((waited + 60))
done
log "GPU quiet after ${waited}s of waiting; starting wave 6"

# A: the missing teacher x steps cell — two seeds in parallel (~2.5 GB each)
train_and_score e2x-s17 configs/student_s5_dr01.json 17 &
train_and_score e2x-s42 configs/student_s5_dr01.json 42 &
wait

# B: big + dropout 0.2 + long (~3.2 GB each)
train_and_score s7x-dr02-s17 configs/student_s7_dr02.json 17 &
train_and_score s7x-dr02-s42 configs/student_s7_dr02.json 42 &
wait

log "=== phase6 wave6 complete ==="
