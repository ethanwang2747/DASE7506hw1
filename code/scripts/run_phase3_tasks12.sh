#!/usr/bin/env bash
# Phase 3: seed recheck (task 1) and S4 size probe (task 2).
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase3
LOG=$P/commands.log
mkdir -p "$P" runs/phase3/diagnostics

run_one() {
  local name="$1" cfg="$2" steps="$3" seed="$4"; shift 4
  local dir="$P/$name"
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[$(date -Is)] !! $dir is not empty, skipping $name" >>"$LOG"; return 1
  fi
  local tcmd="$PY train.py --implementation student --config $cfg --device cuda --precision fp32 --steps $steps --seed $seed --batch-size 32 --run-dir $dir $*"
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

run_one s2-seed42 configs/student_s2.json 1200 42 --optimizer muon --muon-lr 0.05
run_one s2-seed7  configs/student_s2.json 1200 7  --optimizer muon --muon-lr 0.05
run_one s4-probe  configs/student_s4.json 1200 17 --optimizer muon --muon-lr 0.05
echo "[$(date -Is)] task1+task2 complete" >>"$LOG"
