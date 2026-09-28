#!/usr/bin/env bash
# Phase 6, wave 3 — detached from the agent session (setsid nohup), idempotent, fully logged.
#
#   stage A  MTP screen at 1200 steps (mtp-w03 / mtp-w01, dr0) and the promotion gate
#   stage B  e2t: the 7200-step sweet spot re-run with the three EMA teachers (two seeds)
#   stage C  g2: born-again — the same recipe distilling from the two e2t students plus teacher-big2-ema
#   stage C2 (only if the MTP gate passed) g2-mtp: the same two arms with --mtp-weight 0.3
#   stage D  s6-distill: the w320 capacity re-check at the 7200-step sweet spot, alone, then re-timed
#
# Two GPU jobs at a time at most (each distilled run is ~2.6 GB of the 8 GB card).
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY='env -u PYTHONPATH .venv/bin/python'
P=runs/phase6
LOG=$P/commands.log
T3="runs/phase6/teacher-s17-ema/checkpoint.pt,runs/phase6/teacher-big2-ema/checkpoint.pt,runs/phase6/teacher-deep-ema/checkpoint.pt"
DISTILL="--distill-checkpoint $T3 --distill-T 4.0 --distill-alpha 0.7"
MUON="--optimizer muon --muon-lr 0.05"
MTP_REFERENCE=1.6473      # runs/phase4/s5-probe, dr0, 1200 steps
MTP_THRESHOLD=1.6323      # reference - 0.015

log() { echo "[$(date -Is)] $*" >>"$LOG"; }

train_and_score() {   # train_and_score <name> <config> <steps> <eval_every> <extra...>
  local name="$1" cfg="$2" steps="$3" eval_every="$4"; shift 4
  local dir="$P/$name"
  if [ -f "$dir/metrics.json" ]; then
    log "already trained $name"
  else
    mkdir -p "$dir"
    local tcmd="$PY train.py --implementation student --config $cfg --steps $steps --seed ${SEED:-17} --eval-every $eval_every --save-every $eval_every $MUON $* --device cuda --precision fp32 --batch-size 32 --run-dir $dir"
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

val_of() { $PY -c "import json;print(json.load(open('$P/$1/validation_cpu_fp32.json'))['bpb'])" 2>/dev/null; }

log "=== phase6 wave3 started (pid $$) ==="

# ---------------- stage A: MTP screen ----------------
SEED=17 train_and_score mtp-w03 configs/student_s5_mtp.json 1200 0 --mtp-weight 0.3 &
SEED=17 train_and_score mtp-w01 configs/student_s5_mtp.json 1200 0 --mtp-weight 0.1 &
wait

MTP_PROMOTED=0
for arm in mtp-w03 mtp-w01; do
  value=$(val_of "$arm")
  if [ -n "$value" ]; then
    pass=$($PY -c "print(1 if $value <= $MTP_THRESHOLD else 0)")
    log "MTP gate: $arm val $value vs reference $MTP_REFERENCE (threshold $MTP_THRESHOLD) -> $([ "$pass" = 1 ] && echo PROMOTE || echo reject)"
    [ "$pass" = 1 ] && MTP_PROMOTED=1
  else
    log "MTP gate: $arm has no score, cannot judge"
  fi
done
log "MTP gate result: promoted=$MTP_PROMOTED"
echo "$MTP_PROMOTED" >"$P/.mtp_promoted"

# ---------------- stage B: e2t, the 7200-step sweet spot ----------------
SEED=17 train_and_score e2t-s17 configs/student_s5_dr01.json 7200 600 $DISTILL --ema-decay 0.999 &
SEED=42 train_and_score e2t-s42 configs/student_s5_dr01.json 7200 600 $DISTILL --ema-decay 0.999 &
wait

# ---------------- stage C: g2 born-again ----------------
G2_TEACHERS="runs/phase6/e2t-s17/checkpoint.pt,runs/phase6/e2t-s42/checkpoint.pt,runs/phase6/teacher-big2-ema/checkpoint.pt"
if [ ! -f "$P/e2t-s17/checkpoint.pt" ] || [ ! -f "$P/e2t-s42/checkpoint.pt" ]; then
  log "!! e2t checkpoints missing; skipping the born-again stage"
else
  G2_DISTILL="--distill-checkpoint $G2_TEACHERS --distill-T 4.0 --distill-alpha 0.7"
  SEED=17 train_and_score g2-s17 configs/student_s5_dr01.json 7200 600 $G2_DISTILL --ema-decay 0.999 &
  SEED=42 train_and_score g2-s42 configs/student_s5_dr01.json 7200 600 $G2_DISTILL --ema-decay 0.999 &
  wait

  if [ "$MTP_PROMOTED" = 1 ]; then
    log "MTP promoted: running the born-again + MTP sister arms"
    SEED=17 train_and_score g2-mtp-s17 configs/student_s5_dr01_mtp.json 7200 600 $G2_DISTILL --ema-decay 0.999 --mtp-weight 0.3 &
    SEED=42 train_and_score g2-mtp-s42 configs/student_s5_dr01_mtp.json 7200 600 $G2_DISTILL --ema-decay 0.999 --mtp-weight 0.3 &
    wait
  else
    log "MTP not promoted: sister arms skipped"
  fi
fi

# ---------------- stage D: capacity re-check on the w320 model, alone ----------------
SEED=17 train_and_score s6-distill-s17 configs/student_s6_dr01.json 7200 600 $DISTILL --ema-decay 0.999

if [ -f "$P/s6-distill-s17/metrics.json" ]; then
  log "clean re-timing of the w320 checkpoint (machine otherwise idle)"
  ecmd="$PY evaluate.py --checkpoint $P/s6-distill-s17/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4 --output $P/s6-distill-s17.clean-timing.json"
  log "$ecmd"
  eval "$ecmd" >"$P/s6-distill-s17.clean-timing.out" 2>&1 && log "clean timing written" || log "!! clean timing failed"
fi
log "=== phase6 wave3 complete ==="
