#!/usr/bin/env python3
"""Build report/evidence.csv: one row per training run, read from the local run tree.

Usage: python3 make_evidence_csv.py [CODE_ROOT]

CODE_ROOT is the directory that contains `runs/` (default: ../../MP1_student_starter/code
relative to this file; override with argv[1] or $MP1_CODE_ROOT). Every `runs/**/metrics.json`
found there becomes one row, so the CSV is a superset of the runs cited by the report's tables
and each cited number can be looked up by `run_id` / `run_dir`.

Values come from the actual keys of each run's `metrics.json`
(`config`, `seed`, `parameters`, `train_tokens`, `train_seconds`, `validation.bpb`,
`history[-1].step`, `ema.{selected,raw_bpb,ema_bpb}`, `recipe.distill.teachers`,
`implementation_sha256`, `device_name`). Two columns are derived rather than stored:

* `batch_size` = `train_tokens / (steps * config.context)`, written only when that division is
  exact; otherwise `NA`.
* `test_bpb` is read from the run's own post-freeze scoring file when one exists next to it
  (`score_test.json`, `test_cpu_fp32.json`, `test_cpu_fp32_recheck.json`); otherwise `NA`.
  `metrics.json` itself never records a test-split score.

Any field genuinely absent for a run is written as `NA`; nothing is inferred. `val_bpb_raw` /
`val_bpb_ema` are the run's final validation on the training device, and `val_bpb_selected` is
the value selection kept (`ema` when `ema.selected == "ema"`, otherwise the raw value). Where
the run directory also holds `validation_cpu_fp32.json`, that file is the CPU-fp32 re-score of
the *selected* weights and its `bpb` is what the report quotes; it is flagged in `notes`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CODE_ROOT = HERE / "../../MP1_student_starter/code"
CODE_ROOT = Path(
    os.environ.get("MP1_CODE_ROOT")
    or (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else DEFAULT_CODE_ROOT)
).resolve()
OUT = HERE / "evidence.csv"

CONFIG_KEYS = (
    ("vocab", "vocab"),
    ("width", "w"),
    ("heads", "h"),
    ("depth", "d"),
    ("context", "ctx"),
    ("norm", "norm"),
    ("mlp", "mlp"),
    ("qk_norm", "qk_norm"),
    ("tie", "tie"),
    ("bias", "bias"),
    ("dropout", "dropout"),
)
TEST_FILES = ("score_test.json", "test_cpu_fp32.json", "test_cpu_fp32_recheck.json")


def na(value):
    return "NA" if value is None else value


def compact_config(cfg):
    if not isinstance(cfg, dict):
        return "NA"
    parts = []
    for key, label in CONFIG_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        if isinstance(val, bool):
            parts.append(f"{label}={int(val)}")
        else:
            parts.append(f"{label}={val}")
    return "|".join(parts) if parts else "NA"


def teachers(recipe):
    if not isinstance(recipe, dict):
        return "NA"
    distill = recipe.get("distill")
    if not isinstance(distill, dict):
        return "NA"
    names = []
    for teacher in distill.get("teachers") or []:
        if isinstance(teacher, dict) and teacher.get("checkpoint"):
            names.append(str(teacher["checkpoint"]))
    return ",".join(names) if names else "NA"


def test_bpb(run_dir, checkpoint_sha256):
    for name in TEST_FILES:
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "bpb" in data:
            return data["bpb"], name
    # the frozen run's single post-freeze test scoring ships at the bundle root; it is ours
    # only when it names this run's checkpoint
    bundle = HERE.parent / "score_test.json"
    if checkpoint_sha256 and bundle.is_file():
        try:
            data = json.loads(bundle.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("checkpoint_sha256") == checkpoint_sha256 and "bpb" in data:
            return data["bpb"], "../score_test.json"
    return None, None


def main():
    if not (CODE_ROOT / "runs").is_dir():
        raise SystemExit(f"no runs/ directory under {CODE_ROOT}")

    rows = []
    for path in sorted(CODE_ROOT.glob("runs/**/metrics.json")):
        run_dir = path.parent
        rel = run_dir.relative_to(CODE_ROOT).as_posix()
        data = json.loads(path.read_text())
        config = data.get("config") or {}
        history = data.get("history") or []
        steps = max((h.get("step", 0) for h in history), default=None)

        batch_size = None
        context = config.get("context")
        if steps and context and isinstance(data.get("train_tokens"), int):
            if data["train_tokens"] % (steps * context) == 0:
                batch_size = data["train_tokens"] // (steps * context)

        raw = (data.get("validation") or {}).get("bpb")
        ema = data.get("ema") or {}
        selected = ema.get("selected")
        if selected == "ema":
            picked = ema.get("ema_bpb")
        elif selected == "raw":
            picked = ema.get("raw_bpb", raw)
        else:
            picked = raw

        tbpb, tsrc = test_bpb(run_dir, data.get("checkpoint_sha256"))
        notes = []
        if not data.get("ema"):
            notes.append("no-ema-block")
        if (run_dir / "validation_cpu_fp32.json").is_file():
            notes.append("cpu_fp32_rescore")
        if tsrc:
            notes.append(f"test_source={tsrc}")
        if data.get("device_name"):
            notes.append(f"device={data['device_name']}")
        if data.get("preparation_seconds") is not None:
            notes.append(f"prep_seconds={data['preparation_seconds']}")

        rows.append(
            {
                "run_id": run_dir.name,
                "run_dir": rel,
                "config": compact_config(config),
                "seed": na(data.get("seed")),
                "steps": na(steps),
                "batch_size": na(batch_size),
                "parameters": na(data.get("parameters")),
                "processed_targets": na(data.get("train_tokens")),
                "val_bpb_raw": na(raw),
                "val_bpb_ema": na(ema.get("ema_bpb")),
                "val_bpb_selected": na(picked),
                "ema_selected": na(selected),
                "distill_teachers": teachers(data.get("recipe")),
                "test_bpb": na(tbpb),
                "notes": "; ".join(notes),
            }
        )

    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows) from {CODE_ROOT}")


if __name__ == "__main__":
    main()
