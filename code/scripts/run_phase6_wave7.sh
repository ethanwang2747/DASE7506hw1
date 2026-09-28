#!/usr/bin/env bash
# Phase 6, wave 7 — the combination arm (detached, idempotent, logged).
#
# The two best levers had never been combined: S7 (w336 d6 h6, 8.9 M) gave 1.4552 at wd 0.1, and
# lowering weight decay to 0.05 gave 1.4578 on S5 (vs 1.4752 at wd 0.1). This wave runs S7 at wd 0.05,
# the same sweet-spot recipe otherwise (7200 steps, old teacher pair, T=4 alpha=0.7, Muon 0.05,
# EMA 0.999), seeds 17 and 42 in parallel, then scores each on CPU fp32 validation and re-times the
# checkpoints with the machine idle.
#
# Waits for wave 6.5 first, so the two S7-sized groups never share the 8 GB card.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"
DISTILL="--distill-checkpoint $TEACHERS --distill-T 4.0 --distill-alpha 0.7"
WAIT_LIMIT=$((8 * 3600))

log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_one() {   # <name> <seed>
  local name="$1" seed="$2" dir="$P/$1"
  if [ -f "$dir/metrics.json" ]; then log "already trained $name"; return 0; fi
  mkdir -p "$dir"
  local tcmd="$PY train.py --implementation student --config configs/student_s7_dr01.json --steps 7200 --seed $seed --eval-every 600 --save-every 600 --optimizer muon --muon-lr 0.05 --weight-decay 0.05 --ema-decay 0.999 $DISTILL --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
  log "$tcmd"
  if eval "$tcmd" >"$P/$name.train.out" 2>&1; then log "trained $name"; else log "!! training failed for $name"; return 1; fi
}

score() {   # <name> [output]
  local name="$1" out="${2:-}" dir="$P/$1"
  [ -f "$dir/metrics.json" ] || { log "!! not scoring $name: training incomplete"; return 1; }
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4 ${out:+--output $out}"
  log "$ecmd"
  if eval "$ecmd" >"$P/$name.eval.out" 2>&1; then log "finished $name"; else log "!! scoring failed for $name"; fi
}

log "=== phase6 wave7 started (pid $$); waiting for wave 6.5 to clear the GPU ==="
waited=0
while pgrep -f run_phase6_wave65.sh >/dev/null 2>&1 || pgrep -f "train.py --implementation" >/dev/null 2>&1; do
  if [ "$waited" -ge "$WAIT_LIMIT" ]; then
    log "!! gave up waiting for wave 6.5 after ${WAIT_LIMIT}s; not starting wave 7"
    exit 1
  fi
  if [ $((waited % 600)) -eq 0 ]; then
    log "wave7 still waiting (${waited}s): wave6.5 script=$([ "$(pgrep -f run_phase6_wave65.sh >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down) trainings=$([ "$(pgrep -f 'train.py --implementation' >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down)"
  fi
  sleep 60
  waited=$((waited + 60))
done
log "GPU quiet after ${waited}s of waiting; starting wave 7"

train_one s7w-s17 17 &
train_one s7w-s42 42 &
wait

score s7w-s17
score s7w-s42

for name in s7w-s17 s7w-s42; do
  if [ -f "$P/$name/checkpoint.pt" ]; then
    log "clean re-timing of $name (machine otherwise idle)"
    score "$name" "$P/$name.clean-timing.json"
  fi
done
log "=== phase6 wave7 complete ==="
