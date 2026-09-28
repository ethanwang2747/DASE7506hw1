#!/usr/bin/env bash
# Phase 6, wave 2: EMA-retrained teachers, then the N=14400 final matrix (e1l / e2l / e3l).
# Batches keep the GPU busy without exceeding 8 GB: each batch is one 15-20 M teacher plus one
# student, or the two distilled students together. The 14400-step runs checkpoint every 600 steps.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
T1=$P/teacher-s17-ema/checkpoint.pt
T2=$P/teacher-big2-ema/checkpoint.pt
T3=$P/teacher-deep-ema/checkpoint.pt
DISTILL="--distill-checkpoint $T1,$T2,$T3 --distill-T 4.0 --distill-alpha 0.7"
S5=configs/student_s5_dr01.json
MUON="--optimizer muon --muon-lr 0.05"
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

# batch A: the small teacher alongside the (tiny) baseline control for the 14400-step matrix
run_job teacher-s17-ema --implementation student --config configs/teacher_t1.json --steps 4800 \
  --seed 17 --eval-every 600 --optimizer muon --muon-lr 0.02 --ema-decay 0.999 &
run_job e1l-s17 --implementation model --steps 14400 --seed 17 --eval-every 600 &
wait

# batch B: the 20 M teacher alongside the first no-distillation 14400-step arm
run_job teacher-big2-ema --implementation student --config configs/teacher_t2.json --steps 6000 \
  --seed 17 --eval-every 600 --optimizer muon --muon-lr 0.02 --ema-decay 0.999 &
run_job e3l-s17 --implementation student --config $S5 --steps 14400 --seed 17 --eval-every 600 \
  --save-every 600 $MUON --ema-decay 0.9995 &
wait

# batch C: the deep teacher alongside the second no-distillation arm
run_job teacher-deep-ema --implementation student --config configs/teacher_t3.json --steps 7200 \
  --seed 17 --eval-every 600 --optimizer muon --muon-lr 0.02 --ema-decay 0.9995 &
run_job e3l-s42 --implementation student --config $S5 --steps 14400 --seed 42 --eval-every 600 \
  --save-every 600 $MUON --ema-decay 0.9995 &
wait

# batch D: the two distilled arms, with all three EMA teachers riding along
run_job e2l-s17 --implementation student --config $S5 --steps 14400 --seed 17 --eval-every 600 \
  --save-every 600 $MUON --ema-decay 0.9995 $DISTILL &
run_job e2l-s42 --implementation student --config $S5 --steps 14400 --seed 42 --eval-every 600 \
  --save-every 600 $MUON --ema-decay 0.9995 $DISTILL &
wait
echo "[$(date -Is)] phase6 wave 2 complete" >>"$LOG"
