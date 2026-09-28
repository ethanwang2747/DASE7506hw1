#!/usr/bin/env bash
# MP1 freeze-day measurement runbook (local 3070L). Usage:
#   bash scripts/freeze_check.sh <candidate_checkpoint.pt>
# Runs, in order, with the machine otherwise idle (NO concurrent training/eval!):
#   1. protected-file integrity vs the original zip (sha256)
#   2. contract tests (5/5)
#   3. clean CPU timing: baseline x3 and candidate x3 alternating, validation split
#      (test split is NOT touched here; that single run happens after the freeze decision)
#   4. peak RSS of one candidate validation eval under /usr/bin/time -v
#   5. asset bytes + parameter count
# Appends everything to runs/final/freeze-check-<timestamp>.log
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$1"
PY='env -u PYTHONPATH .venv/bin/python'
mkdir -p runs/final
LOG=runs/final/freeze-check-$(date +%Y%m%d-%H%M%S).log

log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "freeze_check candidate=$CKPT"
[ -f "$CKPT" ] || { log "!! checkpoint missing"; exit 1; }

# 1. protected files vs original zip
log "== 1. protected-file integrity"
$PY - <<'EOF' 2>&1 | tee -a "$LOG"
import hashlib, zipfile, pathlib, sys, os
z = zipfile.ZipFile('/home/yucheng/ris/homework/7506/homework1/MP1_student_starter.zip')
zmap = {}
for i in z.infolist():
    parts = pathlib.PurePosixPath(i.filename).parts
    if 'code' in parts:
        rel = str(pathlib.PurePosixPath(*parts[parts.index('code')+1:]))
        if rel: zmap[rel] = i.filename
prot = ['model.py','common.py','evaluate.py','configs/baseline.json','tests/test_contract.py']
prot += [f'data/{n}' for n in sorted(os.listdir('data')) if os.path.isfile(f'data/{n}')]
bad = 0
for rel in prot:
    local = pathlib.Path(rel)
    if not local.exists(): print('MISSING LOCAL', rel); bad += 1; continue
    if rel not in zmap: print('NOT IN ZIP', rel); bad += 1; continue
    ok = hashlib.sha256(local.read_bytes()).hexdigest() == hashlib.sha256(z.read(zmap[rel])).hexdigest()
    print(('OK ' if ok else 'DIFF '), rel)
    bad += 0 if ok else 1
print('protected check:', 'ALL MATCH' if bad==0 else f'{bad} PROBLEMS')
sys.exit(1 if bad else 0)
EOF

# 2. contract tests
log "== 2. contract tests"
$PY -m unittest discover -s tests -v 2>&1 | tail -3 | tee -a "$LOG"

# 3. clean timing: alternate baseline/candidate, 3 each, validation split
log "== 3. clean timing (validation, CPU fp32, threads 4)"
for i in 1 2 3; do
  $PY evaluate.py --checkpoint runs/e0-baseline/checkpoint.pt --device cpu --precision fp32 --split validation --threads 4 --output /tmp/freeze-e0-$i.json >/dev/null 2>&1
  $PY evaluate.py --checkpoint "$CKPT" --device cpu --precision fp32 --split validation --threads 4 --output /tmp/freeze-cand-$i.json >/dev/null 2>&1
  $PY -c "
import json
e=json.load(open('/tmp/freeze-e0-$i.json')); c=json.load(open('/tmp/freeze-cand-$i.json'))
print('round $i: baseline %.2fs bpb=%.4f | candidate %.2fs bpb=%.6f | ratio %.3f'%(e['seconds'],e['bpb'],c['seconds'],c['bpb'],c['seconds']/e['seconds']))" | tee -a "$LOG"
done

# 4. peak RSS
log "== 4. peak RSS (candidate, validation)"
/usr/bin/time -v $PY evaluate.py --checkpoint "$CKPT" --device cpu --precision fp32 --split validation --threads 4 --output /tmp/freeze-cand-rss.json 2>&1 | grep -E 'Maximum resident|Elapsed' | tee -a "$LOG"

# 5. asset
log "== 5. asset"
stat -c 'checkpoint bytes: %s' "$CKPT" | tee -a "$LOG"
$PY -c "
import torch
ck=torch.load('$CKPT',map_location='cpu',weights_only=True)
print('params:', sum(v.numel() for v in ck['model'].values()))
print('protocol:', ck.get('protocol'), 'seed:', ck.get('seed'), 'train_tokens:', ck.get('train_tokens'))
print('config:', ck['config'])" 2>&1 | tee -a "$LOG"

log "freeze_check done -> $LOG"
