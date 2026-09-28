#!/usr/bin/env bash
# Phase 3, task 5: two teacher runs (t1, 4800 steps, seeds 17 and 42), CUDA training, CPU fp32 scoring.
# CPU-only training was measured at ~7 h per seed, so the teachers train on the GPU like the sweep.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase3
LOG=$P/commands.log

run_one() {
  local name="$1" seed="$2" device="$3" threads="$4"
  local dir="$P/$name"
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[$(date -Is)] !! $dir is not empty, skipping $name" >>"$LOG"; return 1
  fi
  local tcmd="$PY train.py --implementation student --config configs/teacher_t1.json --device $device --threads $threads --precision fp32 --steps 4800 --seed $seed --batch-size 32 --eval-every 600 --run-dir $dir --optimizer muon --muon-lr 0.02"
  echo "[$(date -Is)] $tcmd" >>"$LOG"
  if ! eval "$tcmd" >"$P/$name.train.out" 2>&1; then
    echo "[$(date -Is)] !! training failed for $name" >>"$LOG"; return 1
  fi
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4"
  echo "[$(date -Is)] $ecmd" >>"$LOG"
  if ! eval "$ecmd" >"$P/$name.eval.out" 2>&1; then
    echo "[$(date -Is)] !! scoring failed for $name" >>"$LOG"; return 1
  fi
  echo "[$(date -Is)] finished $name" >>"$LOG"
}

run_one "$@"
echo "[$(date -Is)] task5 $1 complete" >>"$LOG"
