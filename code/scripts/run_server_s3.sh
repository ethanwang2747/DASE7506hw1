#!/bin/bash
# MP1 服务器第三波（S3）：在 S2 冠军配方（S7+wd0.05+muon0.03+ema0.9995+cosine+蒸馏）上打两条轴
# 臂1 s3-x28800：时长翻倍 14400→28800（ema 0.9995 horizon 2000=6.9% 合规）×2 seed，教师不变（隔离时长）
# 臂2 s3-ba2：gen-2 born-again——用 S2 组合双 seed 冠军对（1.4283/1.4289，同架构天然多样）做教师 @14400 ×2 seed
# 幂等：有 metrics.json 跳过训练，有 validation json 跳过评测。5090 32GB，4 路并行。
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY=.venv/bin/python
mkdir -p runs/server
LOG=runs/server/commands.log
TEACHERS_OLD="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"
TEACHERS_BA="runs/server/s2-combo-s17/checkpoint.pt,runs/server/s2-combo-s42/checkpoint.pt"

log(){ echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "$LOG"; }

run_one(){ # name steps seed teachers
  local name=$1 steps=$2 seed=$3 teachers=$4
  if [ -f "runs/server/$name/metrics.json" ]; then log "already trained $name";
  else
    mkdir -p "runs/server/$name"
    env -u PYTHONPATH $PY train.py --implementation student --config configs/student_s7_dr01.json \
      --device cuda --precision fp32 --steps "$steps" --seed "$seed" \
      --eval-every 1200 --save-every 1200 \
      --optimizer muon --muon-lr 0.03 --ema-decay 0.9995 --weight-decay 0.05 \
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

log "=== server wave S3 started (pid $$) ==="
run_one s3-x28800-s17 28800 17 "$TEACHERS_OLD" &
run_one s3-x28800-s42 28800 42 "$TEACHERS_OLD" &
run_one s3-ba2-s17 14400 17 "$TEACHERS_BA" &
run_one s3-ba2-s42 14400 42 "$TEACHERS_BA" &
wait
log "=== server wave S3 complete ==="
for f in s3-x28800-s17 s3-x28800-s42 s3-ba2-s17 s3-ba2-s42; do
  [ -f "runs/server/$f/validation_cpu_fp32.json" ] && $PY -c "import json;print('$f val=%.4f'%json.load(open('runs/server/$f/validation_cpu_fp32.json'))['bpb'])" | tee -a "$LOG"
done
