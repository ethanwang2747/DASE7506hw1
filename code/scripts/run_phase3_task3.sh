#!/usr/bin/env bash
# Phase 3, task 3: step-count / dropout sweep on the promoted S4 size (configs/student_s4*.json).
# Training runs two at a time on the GPU (2 x ~2.5 GB fits the 8 GB laptop card), then each finished
# checkpoint is scored on the CPU, one at a time, so the CPU evaluations never contend with each other.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase3
LOG=$P/commands.log

train_one() {
  local name="$1" cfg="$2" steps="$3" eval_every="$4"; shift 4
  local dir="$P/$name"
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[$(date -Is)] !! $dir is not empty, skipping $name" >>"$LOG"; return 1
  fi
  local tcmd="$PY train.py --implementation student --config $cfg --device cuda --precision fp32 --steps $steps --seed 17 --batch-size 32 --eval-every $eval_every --run-dir $dir $*"
  echo "[$(date -Is)] $tcmd" >>"$LOG"
  if eval "$tcmd" >"$P/$name.train.out" 2>&1; then
    echo "[$(date -Is)] trained $name" >>"$LOG"
  else
    echo "[$(date -Is)] !! training failed for $name" >>"$LOG"
  fi
}

score_one() {
  local name="$1"
  local dir="$P/$name"
  if [ ! -f "$dir/checkpoint.pt" ]; then
    echo "[$(date -Is)] !! no checkpoint for $name, skipping scoring" >>"$LOG"; return 1
  fi
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4"
  echo "[$(date -Is)] $ecmd" >>"$LOG"
  if eval "$ecmd" >"$P/$name.eval.out" 2>&1; then
    echo "[$(date -Is)] finished $name" >>"$LOG"
  else
    echo "[$(date -Is)] !! scoring failed for $name" >>"$LOG"
  fi
}

train_one n3600-dr0  configs/student_s4.json      3600 300 --optimizer muon --muon-lr 0.05 &
train_one n3600-dr01 configs/student_s4_dr01.json 3600 300 --optimizer muon --muon-lr 0.05 &
wait
score_one n3600-dr0
score_one n3600-dr01
train_one n4800-dr01 configs/student_s4_dr01.json 4800 300 --optimizer muon --muon-lr 0.05 &
train_one n7200-dr01 configs/student_s4_dr01.json 7200 600 --optimizer muon --muon-lr 0.05 &
wait
score_one n4800-dr01
score_one n7200-dr01
echo "[$(date -Is)] task3 complete" >>"$LOG"
