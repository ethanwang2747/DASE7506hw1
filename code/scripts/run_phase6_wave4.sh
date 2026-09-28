#!/usr/bin/env bash
# Phase 6, wave 4 — detached (setsid nohup), idempotent, fully logged.
#
# Wave 3 is still running when this starts, so stage 0 polls until both the wave-3 script and every
# train.py have exited; only then does it touch the GPU. The distillation teachers are the phase-5
# pair (teacher-s17 + teacher-big-s17) — the pair that actually worked — so the variables isolated
# here are capacity (S7), sampling (epoch shuffle) and the MTP weight.
#
#   stage A  s7t-s17 / s7t-s42   w336 d6 h6 (student_s7_dr01) + distill, seeds 17/42
#   stage B  s5-epoch-s17        S5 + distill + --epoch-shuffle
#            g2-mtpw01-s17       S5 + distill + mtp_k=1 + --mtp-weight 0.1 (the weight wave 3 got wrong)
#   stage C  uncontended re-timing of the S7 checkpoints
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
OLD_TEACHERS="runs/phase3/teacher-s17/checkpoint.pt,runs/phase5/teacher-big-s17/checkpoint.pt"
DISTILL="--distill-checkpoint $OLD_TEACHERS --distill-T 4.0 --distill-alpha 0.7"
MUON="--optimizer muon --muon-lr 0.05"
WAIT_LIMIT=$((6 * 3600))

log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_and_score() {   # train_and_score <name> <config> <steps> <eval_every> <seed> <extra...>
  local name="$1" cfg="$2" steps="$3" eval_every="$4" seed="$5"; shift 5
  local dir="$P/$name"
  if [ -f "$dir/metrics.json" ]; then
    log "already trained $name"
  else
    mkdir -p "$dir"
    local tcmd="$PY train.py --implementation student --config $cfg --steps $steps --seed $seed --eval-every $eval_every --save-every $eval_every $MUON $* --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
    log "$tcmd"
    if ! eval "$tcmd" >"$P/$name.train.out" 2>&1; then
      log "!! training failed for $name"; return 1
    fi
    log "trained $name"
  fi
  if [ -f "$dir/validation_cpu_fp32.json" ]; then log "already scored $name"; return 0; fi
  local ecmd="$PY evaluate.py --checkpoint $dir/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4"
  log "$ecmd"
  if eval "$ecmd" >"$P/$name.eval.out" 2>&1; then log "finished $name"; else log "!! scoring failed for $name"; fi
}

log "=== phase6 wave4 started (pid $$); waiting for wave 3 to clear the GPU ==="

# ---------------- stage 0: wait for wave 3 ----------------
waited=0
while pgrep -f run_phase6_wave3.sh >/dev/null 2>&1 || pgrep -f "train.py --implementation" >/dev/null 2>&1; do
  if [ "$waited" -ge "$WAIT_LIMIT" ]; then
    log "!! gave up waiting for wave 3 after ${WAIT_LIMIT}s; not starting wave 4"
    exit 1
  fi
  if [ $((waited % 600)) -eq 0 ]; then
    log "wave4 still waiting (${waited}s): wave3 script=$([ "$(pgrep -f run_phase6_wave3.sh >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down) trainings=$([ "$(pgrep -f 'train.py --implementation' >/dev/null 2>&1 && echo up || echo down)" = up ] && echo up || echo down)"
  fi
  sleep 60
  waited=$((waited + 60))
done
log "GPU quiet after ${waited}s of waiting; starting wave 4"

# ---------------- stage A: S7 capacity at the sweet spot ----------------
train_and_score s7t-s17 configs/student_s7_dr01.json 7200 600 17 $DISTILL --ema-decay 0.999 &
train_and_score s7t-s42 configs/student_s7_dr01.json 7200 600 42 $DISTILL --ema-decay 0.999 &
wait

# ---------------- stage B: sampling and MTP-weight questions ----------------
train_and_score s5-epoch-s17 configs/student_s5_dr01.json 7200 600 17 $DISTILL --ema-decay 0.999 --epoch-shuffle &
train_and_score g2-mtpw01-s17 configs/student_s5_dr01_mtp.json 7200 600 17 $DISTILL --ema-decay 0.999 --mtp-weight 0.1 &
wait

# ---------------- stage C: uncontended timing of the S7 checkpoints ----------------
for name in s7t-s17 s7t-s42; do
  if [ -f "$P/$name/checkpoint.pt" ]; then
    log "clean re-timing of $name (machine otherwise idle)"
    ecmd="$PY evaluate.py --checkpoint $P/$name/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4 --output $P/$name.clean-timing.json"
    log "$ecmd"
    eval "$ecmd" >"$P/$name.clean-timing.out" 2>&1 && log "clean timing written for $name" || log "!! clean timing failed for $name"
  fi
done
log "=== phase6 wave4 complete ==="
