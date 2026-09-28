# Checkpoint bundle — expected contents

What ships with this submission, and how to verify it. Frozen 2026-09-22; every sha256 and size
below is measured on the shipped files (see `report/report.md` §1 for the matching score table).

## 1. Contents

| file | what it is | size limit | sha256 |
|---|---|---|---|
| `checkpoint.pt` | the frozen model: `{protocol, implementation, config, model, seed, train_tokens}` written by `train.py` (run `s6-gen3dr005-s42`) | ≤ 64 MiB uncompressed | `f6f1f30aa54c3888ca2e8303a9ff00a45882d8c04cfe595543089a2a2c9832fb` |
| `config.json` | a copy of the config the checkpoint was built with (`configs/student_s7_dr005.json`; also embedded in the checkpoint) | — | `22682feb88767f80a646331a331ebd4f15a2d542f10c3d8900b84f19ea01afe4` |
| `RUN_LOG.csv` | the filled row of `code/RUN_LOG_TEMPLATE.csv` for the frozen run (lineage, steps, wall clock) | — | — |
| `train_command.txt` | the exact command line that produced `checkpoint.pt` | — | — |
| `score_test.json` | output of the single post-freeze test-split scoring run (test BPB **1.426939**) | — | `49d19850a33dfa32940e98dfca0ad89e196303e94836f3c9caf7b9818516f17f` |

Not shipped: teacher checkpoints and intermediate snapshots. They are reproducible artifacts, and
each student checkpoint that used them records the teachers' paths and sha256 in its `metrics.json`.
The frozen run's teacher chain: `s4-ba2x-s17/s42` (sha256 `1962e8b1…` / `39041b9c…`) ← `s2-combo-s17/s42`
← `teacher-s17` + `teacher-big-s17`.

## 2. Verification (any machine, CPU only)

```bash
cd code
python -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# 1. the model loads with the frozen implementation and scores as reported
python evaluate.py --checkpoint <bundle>/checkpoint.pt --device cpu --precision fp32 --split test

# 2. the checkpoint is self-describing: implementation sha256 and config are inside
#    (note: with tie=true the state_dict stores the tied embedding/head twice —
#    summing ck["model"] counts ~9.60 M tensors; the unique parameter count is 8,907,696,
#    which is what metrics.json and the report use; the 64 MiB limit applies to bytes either way)
python - <<'PY'
import torch, json, hashlib
ck = torch.load("<bundle>/checkpoint.pt", map_location="cpu", weights_only=True)
print(ck["protocol"], ck["implementation"], json.dumps(ck["config"]))
print("params", sum(v.numel() for v in ck["model"].values()))
print("student.py sha256", hashlib.sha256(open("student.py","rb").read()).hexdigest())
PY
```

## 3. Resource limits for the frozen model

| limit | how it is measured | value |
|---|---|---|
| CPU scoring time ≤ 5× the baseline | `evaluate.py --split test --device cpu --precision fp32` on the frozen model vs the course baseline on the same machine | frozen model **28.72 s** vs baseline **6.11 s** (same-day pair) = **4.70×**; against the older 6.55 s anchor = 4.39×. Both pass — the frozen model's raw result is shipped in `score_test.json`, the schedule and the measured fields in `RUN_LOG.csv`, and the per-run data behind them in `report/evidence.csv` |
| peak evaluation memory ≤ 4 GiB | validation-split scoring under `/usr/bin/time -v`, read “Maximum resident set size” | **2,220,248 kB ≈ 2.12 GiB** (freeze-day, uncontended) |
| uncompressed asset ≤ 64 MiB | `stat -c %s checkpoint.pt` | **35,652,341 B ≈ 34.0 MiB** (8,907,696 unique FP32 parameters; the tied embedding/head is stored twice in the file, 9,595,824 tensor elements) |

## 4. Fallback

If the frozen candidate turns out to be unusable (e.g. it exceeds the scoring budget on the grader's
machine), the pre-registered fallback is the run recorded in `report/report.md` §1's footnote — its
checkpoint is kept in the same layout, so only the paths in sections 1–3 change.
