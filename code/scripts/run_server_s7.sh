#!/bin/bash
# MP1 服务器第七波（S7）：正则/步长轴探底（都是 vs S6 冠军的干净单变量）
# 臂1 s7-dr002：gen3 教师 + dropout 0.02（dr 梯子 0.2/0.1/0.05 之后找底）×2 seed
# 臂2 s7-lr002：gen3 教师 + dropout 0.05 + muon 0.02（lr 0.05→0.03 之后的下一格）×2 seed
# 共同：S7 + wd0.05 + ema0.9995 + cosine + 28800 步 + T4α0.7。幂等。5090 4 路并行。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS_G3="runs/server/s4-ba2x-s17/checkpoint.pt,runs/server/s4-ba2x-s42/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name seed cfg muon_lr
  local name=$1 seed=$2 cfg=$3 muon_lr=$4
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config "$cfg" \
      --device cuda --precision fp32 --steps 28800 --seed "$seed" \
      --eval-every 1200 --save-every 1200 \
      --optimizer muon --muon-lr "$muon_lr" --ema-decay 0.9995 --weight-decay 0.05 \
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

log "=== server wave S7 started (pid $$) ==="
run_one s7-dr002-s17 17 configs/student_s7_dr002.json 0.03 &
run_one s7-dr002-s42 42 configs/student_s7_dr002.json 0.03 &
run_one s7-lr002-s17 17 configs/student_s7_dr005.json 0.02 &
run_one s7-lr002-s42 42 configs/student_s7_dr005.json 0.02 &
wait
log "=== server wave S7 complete ==="
for f in s7-dr002-s17 s7-dr002-s42 s7-lr002-s17 s7-lr002-s42; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
