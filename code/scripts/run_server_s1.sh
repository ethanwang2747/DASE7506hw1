#!/bin/bash
# MP1 服务器第一波（S1）：冠军延长 + lr003确认 + WSD单点A/B
# 5090 32GB，4 个蒸馏 run 全并行。幂等：有 metrics.json 跳过训练，有 validation json 跳过评测。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name cfg steps seed extra...
  local name=$1 cfg=$2 steps=$3 seed=$4; shift 4
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config "$cfg" --device cuda \
      --precision fp32 --steps "$steps" --seed "$seed" --eval-every 1200 --save-every 1200 \
      --optimizer muon --muon-lr 0.05 --ema-decay 0.9995 --weight-decay 0.1 \
      --distill-checkpoint "$TEACHERS" --distill-T 4.0 --distill-alpha 0.7 \
      --run-dir "runs/server/$name" "$@" > "runs/server/$name.train.out" 2>&1 \
      && log "trained $name" || log "!! TRAIN FAILED $name"
  fi
  if [ -f "runs/server/$name/metrics.json" ] && [ ! -f "runs/server/$name/validation_cpu_fp32.json" ]; then
    env -u PYTHONPATH $PY evaluate.py --checkpoint "runs/server/$name/checkpoint.pt" \
      --device cpu --precision fp32 --split validation \
      --output "runs/server/$name/validation_cpu_fp32.json" > "runs/server/$name.eval.out" 2>&1 \
      && log "scored $name" || log "!! EVAL FAILED $name"
  fi
}

log "=== server wave S1 started (pid $$) ==="
run_one s7x-dr01-s17 configs/student_s7_dr01.json 14400 17 &
run_one s7x-dr01-s42 configs/student_s7_dr01.json 14400 42 &
run_one hp-lr003-s42 configs/student_s5_dr01.json 7200 42 --muon-lr 0.03 --ema-decay 0.999 --eval-every 600 --save-every 600 &
run_one wsd-e2x-s17 configs/student_s5_dr01.json 14400 17 --schedule wsd &
wait
log "=== server wave S1 complete ==="
for f in s7x-dr01-s17 s7x-dr01-s42 hp-lr003-s42 wsd-e2x-s17; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
