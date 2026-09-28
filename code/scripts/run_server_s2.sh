#!/bin/bash
# MP1 服务器第二波（S2）：最终组合格。S1 裁决后：调度=cosine（WSD 已否决）、EMA 0.9995、旧教师对 T4α0.7。
# 臂1 s2-combo：S7 + wd0.05 + muon0.03（两单变量增益叠加）×2 seed —— 最终冲刺候选
# 臂2 s2-wd005：S7 + wd0.05 + muon0.05（只动 wd）×2 seed —— 拆开 lr 归因，服务冻结选型
# 幂等：有 metrics.json 跳过训练，有 validation json 跳过评测。5090 32GB，4 路并行（S1 实测峰值 12.8GB）。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name seed muon_lr
  local name=$1 seed=$2 muon_lr=$3
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config configs/student_s7_dr01.json \
      --device cuda --precision fp32 --steps 14400 --seed "$seed" \
      --eval-every 1200 --save-every 1200 \
      --optimizer muon --muon-lr "$muon_lr" --ema-decay 0.9995 --weight-decay 0.05 \
      --distill-checkpoint "$TEACHERS" --distill-T 4.0 --distill-alpha 0.7 \
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

log "=== server wave S2 started (pid $$, schedule=cosine) ==="
run_one s2-combo-s17 17 0.03 &
run_one s2-combo-s42 42 0.03 &
run_one s2-wd005-s17 17 0.05 &
run_one s2-wd005-s42 42 0.05 &
wait
log "=== server wave S2 complete ==="
for f in s2-combo-s17 s2-combo-s42 s2-wd005-s17 s2-wd005-s42; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
