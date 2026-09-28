#!/usr/bin/env bash
# Phase 5 final matrix: one heterogeneous big teacher, then the e1/e2/e3 student arms at N=7200.
# Two GPU jobs at a time (a 20 M teacher plus a student already reserves ~6 GB of the 8 GB card).
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase5
LOG=$P/commands.log
T_S17=runs/phase3/teacher-s17/checkpoint.pt
T_BIG=$P/teacher-big-s17/checkpoint.pt
mkdir -p "$P"

run_job() {
  local name="$1"; shift
  local dir="$P/$name"
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[$(date -Is)] !! $dir is not empty, skipping $name" >>"$LOG"; return 1
  fi
  local tcmd="$PY train.py $* --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
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

S5=configs/student_s5_dr01.json
MUON="--optimizer muon --muon-lr 0.05"
EMA="--ema-decay 0.999"

# stage 1: the big teacher, sharing the card with the first no-distillation arm
run_job teacher-big-s17 --implementation student --config configs/teacher_t2.json --steps 7200 \
  --seed 17 --eval-every 1200 --optimizer muon --muon-lr 0.02 &
run_job e3-s17 --implementation student --config $S5 --steps 7200 --seed 17 --eval-every 600 \
  $MUON $EMA &
wait

# stage 2: second no-distillation arm plus the two baseline-recipe controls
run_job e3-s42 --implementation student --config $S5 --steps 7200 --seed 42 --eval-every 600 \
  $MUON $EMA &
run_job e1-s17 --implementation model --steps 7200 --seed 17 --eval-every 1200 &
run_job e1-s42 --implementation model --steps 7200 --seed 42 --eval-every 1200 &
wait

# stage 3: the distilled arms, from the phase-3 15 M teacher and the fresh 20 M one
run_job e2-s17 --implementation student --config $S5 --steps 7200 --seed 17 --eval-every 600 \
  $MUON $EMA --distill-checkpoint "$T_S17,$T_BIG" --distill-T 4.0 --distill-alpha 0.7 &
run_job e2-s42 --implementation student --config $S5 --steps 7200 --seed 42 --eval-every 600 \
  $MUON $EMA --distill-checkpoint "$T_S17,$T_BIG" --distill-T 4.0 --distill-alpha 0.7 &
wait
echo "[$(date -Is)] phase5 matrix complete" >>"$LOG"
