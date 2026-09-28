#!/bin/bash
# MP1 服务器第四波（S4）：叠加 S3 两条获胜轴 + 顺「正则过度」证据再放松一格
# 臂1 s4-ba2x：born-again 二代教师对（s2-combo 双 seed）+ 28800 步 ×2 seed —— 两轴组合
# 臂2 s4-wd002：s3-x28800 配方 + wd 0.02（单变量 vs s3-x28800 的 wd0.05）×2 seed
# 共同：S7(student_s7_dr01) + muon0.03 + ema0.9995 + cosine + T4α0.7。幂等。5090 4 路并行。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS_OLD="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"
TEACHERS_BA="runs/server/s2-combo-s17/checkpoint.pt,runs/server/s2-combo-s42/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name steps seed teachers wd
  local name=$1 steps=$2 seed=$3 teachers=$4 wd=$5
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config configs/student_s7_dr01.json \
      --device cuda --precision fp32 --steps "$steps" --seed "$seed" \
      --eval-every 1200 --save-every 1200 \
      --optimizer muon --muon-lr 0.03 --ema-decay 0.9995 --weight-decay "$wd" \
      --distill-checkpoint "$teachers" --distill-T 4.0 --distill-alpha 0.7 \
      --run-dir "runs/server/$name" > "runs/server/$name.train.out" 2>&1 \
      && log "trained $name" || log "!! TRAIN FAILED $name"
  fi
  if [ -f "runs/server/$name/metrics.json" ] && [ ! -f "runs/server/$name/validation_cpu_fp32.json" ]; then
    env -u PYTHONPATH $PY evaluate.py --checkpoint "runs/server/$name/checkpoint.pt" \
      --device cpu --precision fp32 --split validation \
      --output "runs/server/$name/validation_cpu_fp32.json" > "runs/server/$name.eval.out" 2>&1 \
      && log "scored $name" || log "!! EVAL FAILED $name"
  fi
}

log "=== server wave S4 started (pid $$) ==="
run_one s4-ba2x-s17 28800 17 "$TEACHERS_BA" 0.05 &
run_one s4-ba2x-s42 28800 42 "$TEACHERS_BA" 0.05 &
run_one s4-wd002-s17 28800 17 "$TEACHERS_OLD" 0.02 &
run_one s4-wd002-s42 28800 42 "$TEACHERS_OLD" 0.02 &
wait
log "=== server wave S4 complete ==="
for f in s4-ba2x-s17 s4-ba2x-s42 s4-wd002-s17 s4-wd002-s42; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
