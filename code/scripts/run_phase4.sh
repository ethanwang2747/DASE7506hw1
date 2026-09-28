#!/usr/bin/env bash
# Phase 4 experiments: distillation T/alpha grid, dropout at N=7200, w288d6 size probe, EMA check.
# Two jobs run at a time on the GPU (a training job plus its teacher fits comfortably; three do not),
# and each job scores itself on the CPU as soon as its training ends.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase4
LOG=$P/commands.log
T1=runs/phase3/teacher-s17/checkpoint.pt
T2=runs/phase3/teacher-s42/checkpoint.pt
mkdir -p "$P"

run_job() {
  local name="$1" cfg="$2" steps="$3" eval_every="$4"; shift 4
  local dir="$P/$name"
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[$(date -Is)] !! $dir is not empty, skipping $name" >>"$LOG"; return 1
  fi
  local tcmd="$PY train.py --implementation student --config $cfg --device cuda --precision fp32 --steps $steps --seed 17 --batch-size 32 --eval-every $eval_every --run-dir $dir $*"
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

# batch 1: the long dropout control, paired with the first distillation arm
run_job n7200-dr0   configs/student_s4.json 7200 600 --optimizer muon --muon-lr 0.05 &
run_job d-t2a07     configs/student_s4.json 1200 0 --optimizer muon --muon-lr 0.05 --distill-checkpoint "$T1" --distill-T 2.0 --distill-alpha 0.7 &
wait

# batch 2: the remaining single-teacher arms
run_job d-t4a07     configs/student_s4.json 1200 0 --optimizer muon --muon-lr 0.05 --distill-checkpoint "$T1" --distill-T 4.0 --distill-alpha 0.7 &
run_job d-t2a09     configs/student_s4.json 1200 0 --optimizer muon --muon-lr 0.05 --distill-checkpoint "$T1" --distill-T 2.0 --distill-alpha 0.9 &
wait

# batch 3: dual-teacher arm and the EMA check
run_job d-t2a07-dual configs/student_s4.json 1200 0 --optimizer muon --muon-lr 0.05 --distill-checkpoint "$T1,$T2" --distill-T 2.0 --distill-alpha 0.7 &
run_job ema-0.999   configs/student_s4.json 1200 0 --optimizer muon --muon-lr 0.05 --ema-decay 0.999 &
wait

# batch 4: the size probe runs alone so its CPU scoring is unobstructed
run_job s5-probe    configs/student_s5.json 1200 0 --optimizer muon --muon-lr 0.05
echo "[$(date -Is)] phase4 experiments complete" >>"$LOG"
