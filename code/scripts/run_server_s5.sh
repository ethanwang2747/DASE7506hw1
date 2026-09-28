#!/bin/bash
# MP1 服务器第五波（S5）：第三代 born-again ± 时长轴
# 臂1 s5-gen3：教师 = s4-ba2x 双 seed 对（1.4164/1.4167），28800 步 —— 隔离 gen3
# 臂2 s5-gen3x：同教师 + 43200 步 —— 叠加时长（ema 0.9995 horizon 2000/43200=4.6% 合规）
# 共同：S7(student_s7_dr01) + muon0.03 + wd0.05 + ema0.9995 + cosine + T4α0.7。幂等。5090 4 路并行。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS_G3="runs/server/s4-ba2x-s17/checkpoint.pt,runs/server/s4-ba2x-s42/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name steps seed
  local name=$1 steps=$2 seed=$3
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config configs/student_s7_dr01.json \
      --device cuda --precision fp32 --steps "$steps" --seed "$seed" \
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

log "=== server wave S5 started (pid $$) ==="
run_one s5-gen3-s17 28800 17 &
run_one s5-gen3-s42 28800 42 &
run_one s5-gen3x-s17 43200 17 &
run_one s5-gen3x-s42 43200 42 &
wait
log "=== server wave S5 complete ==="
for f in s5-gen3-s17 s5-gen3-s42 s5-gen3x-s17 s5-gen3x-s42; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
