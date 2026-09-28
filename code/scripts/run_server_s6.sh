#!/bin/bash
# MP1 服务器第六波（S6）：最后一格 = gen3 教师对 + dropout 0.05 @28800 ×2 seed
# 依据：dr005 在 x28800 配方上配对 −0.0031（本地 phase7）；gen3 教师 −0.0020；组合未知。
# 共同：student_s7_dr005 + muon0.03 + wd0.05 + ema0.9995 + cosine + T4α0.7。幂等。5090 2 路并行。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS_G3="runs/server/s4-ba2x-s17/checkpoint.pt,runs/server/s4-ba2x-s42/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name seed
  local name=$1 seed=$2
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config configs/student_s7_dr005.json \
      --device cuda --precision fp32 --steps 28800 --seed "$seed" \
      --eval-every 1200 --save-every 1200 \
      --optimizer muon --muon-lr 0.03 --ema-decay 0.9995 --weight-decay 0.05 \
      --distill-checkpoint "$TEACHERS_G3" --distill-T 4.0 --distill-alpha 0.7 \
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

log "=== server wave S6 started (pid $$) ==="
run_one s6-gen3dr005-s17 17 &
run_one s6-gen3dr005-s42 42 &
wait
log "=== server wave S6 complete ==="
for f in s6-gen3dr005-s17 s6-gen3dr005-s42; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
