# DASE7506 MP1 — submission

Reproducible package for the DASE7506 MP1 small-language-model challenge: a from-scratch transformer
trained (and distilled) on WikiText-2 under a fixed BPE-2048 tokenizer, scored by the course's
untouched evaluator (`code/evaluate.py`, protocol `7506-mp1-wt2-v2`).

Code repository: https://github.com/ethanwang2747/DASE7506hw1

* `code/` — the frozen training/evaluation code. `student.py` and `train.py` are ours; `model.py`,
  `common.py`, `evaluate.py`, `tests/`, `configs/baseline.json` and `data/` are the course files,
  byte-identical to the distributed package.
* The written report ships in two languages, each ≤ 10 pages and each with the figures from
  `report/figs/`: Chinese as `report/report.md` + `report/report.pdf`, English as
  `report/report-en.md` + `report/report-en.pdf`.
* `CHECKPOINTS.md` — what the shipped checkpoint bundle contains and how to verify it.

## 1. Install

Python 3.12 is required. Everything below runs from `code/`.

```bash
cd code
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1

# CPU-only (this is the device the score is measured on)
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
# or an NVIDIA GPU for training (the score is still measured on CPU)
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt
python -m unittest discover -s tests -v     # 5/5 contract tests, CPU only
```

Data and the tokenizer are inside `code/data/`; no download, API key or pretrained weights are needed.

## 2. Train and evaluate

The exact command line of the **frozen** recipe (run `s6-gen3dr005-s42`, frozen 2026-09-22):

```bash
# final recipe — train (teachers: the s4-ba2x pair from the previous born-again generation)
python train.py --implementation student --config configs/student_s7_dr005.json \
  --steps 28800 --seed 42 --batch-size 32 --device cuda --precision fp32 \
  --optimizer muon --muon-lr 0.03 --weight-decay 0.05 --ema-decay 0.9995 \
  --eval-every 1200 --save-every 1200 \
  --distill-checkpoint <teacher A>,<teacher B> --distill-T 4.0 --distill-alpha 0.7 \
  --run-dir runs/s6-gen3dr005-s42

# final recipe — score (CPU fp32, the graded setting); test is scored ONCE, after freezing
# the frozen checkpoint ships at the bundle root, i.e. one level above code/
python evaluate.py --checkpoint ../checkpoint.pt \
  --device cpu --precision fp32 --split test
```

A run writes `metrics.json` (recipe, curves, EMA/raw values, provenance) and `checkpoint.pt`
(plus `checkpoint-step<N>.pt` snapshots when `--save-every` is on). Every training run needs a new
empty `--run-dir`.

## 3. The three resource limits

All three are measured on the **frozen** model, on this machine, with the course evaluator. The
measurement procedure is fixed here so the numbers cannot be produced by a different method later.

1. **Evaluation CPU time ≤ 5× the baseline.** Score the course baseline and the frozen model on the
   same machine (`--split test --device cpu --precision fp32`); the ratio of the two wall clocks must
   be ≤ 5. Measured 2026-09-22 on this machine: baseline **6.11 s**, frozen model **28.72 s** ⇒
   ratio **4.70×** (against the older 6.55 s anchor: 4.39×). Pass.
2. **Peak evaluation memory ≤ 4 GiB.** Run the same scoring under `/usr/bin/time -v` and read
   “Maximum resident set size”. `common.device_metrics()` returns 0 on CPU, so this must be measured
   from the outside. Measured: **2,220,248 kB ≈ 2.12 GiB** (uncontended, freeze-day). Pass.
3. **Uncompressed inference asset ≤ 64 MiB.** `stat -c %s` on the shipped checkpoint (FP32 storage
   is 4 bytes per parameter ⇒ ≈16.78 M parameters). Measured: **35,652,341 B ≈ 34.0 MiB**
   (8,907,696 unique parameters; tied embedding stored twice ⇒ 9,595,824 tensor elements). Pass.

## 4. Reproduction path (summary)

1. Install as above; run the contract tests.
2. Baseline: `python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/e0-baseline`
   then score it on the test split — this reproduces the 5× budget anchor.
3. Train the frozen recipe (section 2). Distillation teachers are not shipped as assets; the
   checkpoints used are recorded with path + sha256 inside each run's `metrics.json`, and the
   teacher recipe is in the report (§7, cost disclosure).
4. Score with `evaluate.py` on **validation** while iterating; the **test** split is scored once,
   after freezing.
5. `code/scripts/select_best_checkpoint.py <run-dir>` ranks every checkpoint in a run directory by
   validation BPB (CPU fp32) and copies the winner to `checkpoint-best.pt` — useful when
   `--save-every` was on and a run’s curve turned back up.

## 5. AI-assisted work — disclosure

Per the course guide, substantive AI assistance is disclosed here:

* **Implementation and experiment orchestration.** [Kimi Code](https://github.com/MoonshotAI/kimi-cli)
  (an agentic coding CLI) was used to write most of `student.py` and the additions to `train.py`
  (Muon, distillation, EMA, multi-token-prediction heads, `--epoch-shuffle`, `--schedule wsd`,
  per-point checkpoint snapshots), to author the `scripts/` runners and `diagnostics/position_nll.py`,
  and to schedule and supervise the training runs described in the report. All generated code was
  reviewed, executed and verified in this repository: the contract tests, the bit-identity smoke test
  of the default training path, and the checkpoint-compatibility checks are run after every change.
* **Strategy review.** Separate AI reviews were used to critique the experiment plan (which levers to
  keep or drop, how to control noise, the freeze discipline). Their recommendations were accepted or
  rejected in writing in the project's work log; two examples of rejections are "calibrate the
  val→test mapping with the test split before freezing" (rejected: that is test-driven selection) and
  "scale up regularisation for long plans" (tested and rejected: dropout 0.2 at 14,400 steps lost by
  +0.030 paired, while relaxing weight decay 0.1→0.05 won −0.024 — the regime was over-regularised).
* **What the AI did not do.** No part of the score comes from a pretrained model or an external
  service at evaluation time: the shipped model is trained from random initialisation by `train.py`
  inside this package, and scoring runs fully offline on CPU.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia
contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when
redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text.
Dataset hashes are in `code/data/manifest.json`. These dataset notices do not assign a new license to
the surrounding classroom code.

---

## 中文摘要

本仓库是 DASE7506 MP1 的提交包：`code/` 是冻结后的训练与评测代码（`student.py` 与 `train.py`
为自研部分，其余课程文件与原始分发包逐位一致），`report/report.md` 为报告（≤10 页，
图表在 `report/figs/`），`CHECKPOINTS.md` 说明随包提交的 checkpoint 内容与校验方式。
三条资源限制（评测 CPU 时间 ≤ 基线 5×、峰值内存 ≤ 4 GiB、未压缩资产 ≤ 64 MiB）的测量流程见
第 3 节，最终数值为 28.72 s / 6.11 s = 4.70×、2.12 GiB、34.0 MiB（逐 run 数据见
`report/evidence.csv`）。AI 辅助情况见第 5 节披露。数据许可见第 6 节。
