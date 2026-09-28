#!/usr/bin/env bash
# Phase 6, wave 1: 1200-step screening (untied head, w320 d6 h5, Muon 0.08) and the two upgraded
# teachers. The screen runs three-up (each ~2 GB, the card holds ~6 GB of them); the 18-20 M teachers
# then run strictly one at a time, because two of them plus their optimiser states do not fit 8 GB.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
mkdir -p "$P"

train_only() {
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
    echo "[$(date -Is)] !! training failed for $name" >>"$LOG"; return 1
  fi
}

score() {
  local name="$1" out="${2:-}" dir="$P/$1"
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4 ${out:+--output $out}"
  echo "[$(date -Is)] $ecmd" >>"$LOG"
  if eval "$ecmd" >"$P/$name.eval.out" 2>&1; then
    echo "[$(date -Is)] finished $name" >>"$LOG"
  else
    echo "[$(date -Is)] !! scoring failed for $name" >>"$LOG"
  fi
}

# stage 1: the three screening trainings together
train_only s5-untie configs/student_s5_untie.json 1200 0 --optimizer muon --muon-lr 0.05 &
train_only s6-probe configs/student_s6.json       1200 0 --optimizer muon --muon-lr 0.05 &
train_only s5-lr008 configs/student_s5.json       1200 0 --optimizer muon --muon-lr 0.08 &
wait

# stage 1 scoring: the w320 probe is scored while the machine is otherwise idle, and scored a second
# time into an explicit clean-timing file, because its membership in the next round hinges on it
score s5-untie
score s5-lr008
score s6-probe
score s6-probe "$P/s6-probe.clean-timing.json"

# stage 2/3: teachers, one at a time
train_only teacher-big2-s17 configs/teacher_t2.json 6000 1200 --optimizer muon --muon-lr 0.02
score teacher-big2-s17
train_only teacher-deep-s17 configs/teacher_t3.json 7200 1200 --optimizer muon --muon-lr 0.02
score teacher-deep-s17
echo "[$(date -Is)] phase6 wave 1 complete" >>"$LOG"
