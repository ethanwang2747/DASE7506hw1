#!/usr/bin/env bash
# Phase 6 wave 2, remainder — designed to run detached from the agent session.
#
#   stage 1  wait (poll) for the two in-flight runs to finish training, then score each on CPU fp32
#            teacher-deep-ema (7200 steps) and e3l-s42 (14400 steps, no distillation)
#   stage 2  run the distilled pair e2l-s17 / e2l-s42 in parallel and score each
#
# Everything is appended to runs/phase6/commands.log. Each step is idempotent: a run whose
# metrics.json exists is not retrained, and a run whose validation_cpu_fp32.json exists is not
# rescored, so the script can be re-run safely after an interruption.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
S5=configs/student_s5_dr01.json
TEACHERS="runs/phase6/teacher-s17-ema/checkpoint.pt,runs/phase6/teacher-big2-ema/checkpoint.pt,runs/phase6/teacher-deep-ema/checkpoint.pt"
WAIT_LIMIT=$((110 * 60))   # seconds to wait for the in-flight runs before giving up on them

log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_running() { pgrep -f "run-dir $P/$1( |$)" >/dev/null 2>&1; }

wait_for_training() {
  local name="$1" dir="$P/$1" waited=0
  if [ -f "$dir/metrics.json" ]; then log "already trained $name"; return 0; fi
  log "waiting for $name to finish training (metrics.json)"
  while [ ! -f "$dir/metrics.json" ]; do
    if [ "$waited" -ge "$WAIT_LIMIT" ]; then
      log "!! gave up waiting for $name after ${WAIT_LIMIT}s (still running: $([ "$(train_running "$name" && echo yes || echo no)" = yes ] && echo yes || echo no))"
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
  log "$name finished training (waited ${waited}s)"
  return 0
}

score() {
  local name="$1" dir="$P/$1"
  if [ ! -f "$dir/metrics.json" ]; then
    log "!! not scoring $name: training did not complete (no metrics.json)"; return 1
  fi
  if [ -f "$dir/validation_cpu_fp32.json" ]; then log "already scored $name"; return 0; fi
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4"
  log "$ecmd"
  if eval "$ecmd" >"$P/$name.eval.out" 2>&1; then
    log "finished $name"
  else
    log "!! scoring failed for $name"
  fi
}

train_e2l() {
  local seed="$1" name="e2l-s$1" dir="$P/e2l-s$1"
  if [ -f "$dir/metrics.json" ]; then log "already trained $name"; score "$name"; return 0; fi
  if train_running "$name"; then log "$name is already training; waiting for it"; wait_for_training "$name" || return 1; score "$name"; return 0; fi
  mkdir -p "$dir"   # claim the run directory before launching
  local tcmd="$PY train.py --implementation student --config $S5 --steps 14400 --seed $seed --eval-every 600 --save-every 600 --optimizer muon --muon-lr 0.05 --ema-decay 0.9995 --distill-checkpoint $TEACHERS --distill-T 4.0 --distill-alpha 0.7 --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
  log "$tcmd"
  if ! eval "$tcmd" >"$P/$name.train.out" 2>&1; then
    log "!! training failed for $name"; return 1
  fi
  log "trained $name"
  score "$name"
}

log "=== phase6 wave2 remainder started (pid $$) ==="

# stage 1: the two runs that were already in flight
wait_for_training teacher-deep-ema; deep_ok=$?
wait_for_training e3l-s42;          e3l42_ok=$?
score teacher-deep-ema
score e3l-s42

if [ "$deep_ok" -ne 0 ] || [ "$e3l42_ok" -ne 0 ]; then
  log "!! one of the in-flight runs did not complete; NOT starting the distilled pair (rerun this script after inspecting)"
  exit 1
fi

# stage 2: the distilled pair, in parallel (~2.5 GB each: student + three teacher forwards)
train_e2l 17 &
train_e2l 42 &
wait
log "=== phase6 wave2 remainder complete ==="
