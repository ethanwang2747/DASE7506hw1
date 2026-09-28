#!/usr/bin/env python3
"""Generate the DASE7506 MP1 report figures (read-only over code/runs).

Usage: <project>/.venv/bin/python make_figs.py [CODE_ROOT] [--only NAME[,NAME...]]

CODE_ROOT defaults to ../../../MP1_student_starter/code relative to this file
(override with argv[1] or $MP1_CODE_ROOT). --only regenerates just the named
figures, e.g. --only assembly_ladder. Writes PNGs next to this script;
nothing else is written.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
_only: list[str] = []
_positional: list[str] = []
_argv = sys.argv[1:]
_i = 0
while _i < len(_argv):
    if _argv[_i] == "--only" and _i + 1 < len(_argv):
        _only = [x for x in _argv[_i + 1].replace(",", " ").split() if x]
        _i += 2
        continue
    _positional.append(_argv[_i])
    _i += 1
CODE_ROOT = Path(
    os.environ.get("MP1_CODE_ROOT")
    or (_positional[0] if _positional else HERE / "../../../MP1_student_starter/code")
).resolve()
RUNS = CODE_ROOT / "runs"
DPI = 150
FIGW = 6.5

C_GREY = "#7a7a7a"
C_BLUE = "#1f5fa9"
C_RED = "#c0392b"
C_ORANGE = "#d98014"
C_GREEN = "#2e7d32"

# Report §6.1 quotes the sd of the 1,200-step S4 repeats as 0.0088, computed from unrounded BPBs;
# the 4-decimal values in results.csv recompute to 0.00874. The figure label follows the report.
REPORT_SD_1200 = 0.0088

def note(msg: str) -> None:
    print(msg)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def save(fig, name: str) -> None:
    from matplotlib.axis import Axis
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    r = canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    w, h = fig.get_size_inches()
    over = []
    for art in list(fig.texts) + [c for ax in fig.axes for c in ax.get_children()]:
        if isinstance(art, (Line2D, Axis)):
            continue
        try:
            bb = art.get_window_extent(r).transformed(inv)
        except Exception:
            continue
        if bb.x1 > w - 0.005 or bb.x0 < 0.005 or bb.y1 > h - 0.005 or bb.y0 < 0.005:
            over.append(f"{type(art).__name__} x[{bb.x0:.2f},{bb.x1:.2f}] y[{bb.y0:.2f},{bb.y1:.2f}]")
    if over:
        note(f"WARNING {name}: {len(over)} artists outside the {w:.2f}x{h:.2f} in canvas -> " + "; ".join(over))
    out = HERE / name
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    note(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {w:.2f} x {h:.2f} in)")


def metrics(run: str) -> dict:
    return json.loads((RUNS / run / "metrics.json").read_text())


def ema_bpb(run: str) -> float:
    return metrics(run)["ema"]["ema_bpb"]


def raw_bpb(run: str) -> float:
    return metrics(run)["ema"]["raw_bpb"]


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- figure 1
def fig_position_nll() -> None:
    d = RUNS / "phase3" / "diagnostics"
    e0 = read_csv(d / "e0-baseline-checkpoint-validation_cpu_fp32.position-nll.csv")
    st = read_csv(d / "n7200-dr01-checkpoint-validation_cpu_fp32.position-nll.csv")
    x0 = np.array([int(r["position"]) for r in e0])
    y0 = np.array([float(r["mean_nll"]) for r in e0])
    x1 = np.array([int(r["position"]) for r in st])
    y1 = np.array([float(r["mean_nll"]) for r in st])
    n_tok = float(e0[0]["count"])
    note(
        f"fig1 rows: E0={len(e0)} student={len(st)} positions "
        f"{x0.min()}..{x0.max()}, ~{n_tok:.0f} targets/position"
    )

    fig, ax = plt.subplots(figsize=(FIGW, 3.3))
    ax.axvspan(-0.5, 15.5, color="#c8c8c8", alpha=0.35, lw=0, zorder=0)
    ax.plot(x0, y0, color=C_GREY, lw=1.6, label="E0 baseline (2.071)")
    ax.plot(x1, y1, color=C_BLUE, lw=1.8, label="n7200-dr0.1 student (1.551)")
    ax.annotate(
        "first 16 positions\n(cold-start region)",
        xy=(7.5, 4.55),
        xytext=(38, 4.72),
        fontsize=9,
        color="#4d4d4d",
        arrowprops=dict(arrowstyle="->", color="#4d4d4d", lw=0.9),
    )
    f0, f1 = y0[:16].mean(), y1[:16].mean()
    r0, r1 = y0[16:].mean(), y1[16:].mean()
    ax.text(
        0.985,
        0.94,
        "mean NLL (nats)\n"
        f"front-16:  E0 {f0:.2f} -> student {f1:.2f}\n"
        f"rest:      E0 {r0:.2f} -> student {r1:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        linespacing=1.5,
        bbox=dict(fc="white", ec="#bbbbbb", lw=0.7, boxstyle="round,pad=0.35"),
    )
    ax.set_xlim(-1, 256)
    ax.set_ylim(2.9, 5.05)
    ax.set_xlabel("Position in the 256-token window")
    ax.set_ylabel("Mean NLL (nats)")
    ax.set_title("Per-position mean NLL, CPU fp32 validation (1,472 windows)")
    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#bbbbbb",
        framealpha=0.95,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.19)
    save(fig, "position_nll.png")


# ---------------------------------------------------------------- figure 2
def fig_val_curves_phase5() -> None:
    rows = read_csv(RUNS / "phase5" / "validation_curves.csv")
    series: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        series.setdefault(r["run"], []).append((int(r["step"]), float(r["bpb"])))
    for k in series:
        series[k].sort()

    teachers = {}
    for run, label in (
        ("phase3/teacher-s17", "teacher-s17 (15.0 M, 4.8k)"),
        ("phase5/teacher-big-s17", "teacher-big-s17 (20.1 M, 7.2k)"),
    ):
        hist = metrics(run)["validation_history"]
        teachers[label] = (np.array([h["step"] for h in hist]), np.array([h["bpb"] for h in hist]))

    note(
        "fig2 rows: validation_curves.csv "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(series.items()))
        + "; teacher histories: "
        + ", ".join(f"{k}={len(v[0])}" for k, v in teachers.items())
    )

    fig, ax = plt.subplots(figsize=(FIGW, 4.0))
    recipes = [
        ("e1", "e1 baseline (AdamW)", C_GREY),
        ("e3", "e3 = S5+Muon+dr0.1+EMA (no distill)", C_BLUE),
        ("e2", "e2 = e3 + 2-teacher distill", C_RED),
    ]
    for key, label, color in recipes:
        for seed, alpha, lw in ((17, 1.0, 1.8), (42, 0.42, 1.5)):
            pts = series[f"{key}-s{seed}"]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                alpha=alpha,
                lw=lw,
                marker="o" if seed == 17 else "s",
                ms=3.2,
                label=label if seed == 17 else None,
            )
    for (label, (xs, ys)), ls in zip(teachers.items(), ("--", "-.")):
        ax.plot(xs, ys, ls, color="#444444", lw=1.4, alpha=0.9, label=label)

    ax.set_xlim(400, 7600)
    ax.set_ylim(1.44, 2.06)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation BPB (raw weights)")
    ax.set_title("Phase 5: 7,200-step recipes, two seeds each")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="black", lw=1.8, label="seed 17"),
        Line2D([], [], color="black", lw=1.5, alpha=0.42, label="seed 42"),
    ]
    labels += ["seed 17", "seed 42"]
    ax.legend(handles, labels, loc="upper right", frameon=False, ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.27)
    fig.text(
        0.01,
        0.02,
        "Intermediate points are the periodic on-device validation on raw weights; the EMA-selected final\n"
        "values are lower (e2 1.475/1.470, e3 1.517/1.519). Teachers are shown as reference curves.",
        fontsize=9,
        color="#4d4d4d",
    )
    save(fig, "val_curves_phase5.png")


# ---------------------------------------------------------------- figure 3
def fig_distill_ablation() -> None:
    pairs = {
        "e3 (no distill)": {"s17": ema_bpb("phase5/e3-s17"), "s42": ema_bpb("phase5/e3-s42")},
        "e2 (2-teacher distill)": {"s17": ema_bpb("phase5/e2-s17"), "s42": ema_bpb("phase5/e2-s42")},
    }
    raw = {
        "e3 (no distill)": {"s17": raw_bpb("phase5/e3-s17"), "s42": raw_bpb("phase5/e3-s42")},
        "e2 (2-teacher distill)": {"s17": raw_bpb("phase5/e2-s17"), "s42": raw_bpb("phase5/e2-s42")},
    }
    keys = list(pairs)
    xs = np.array([0.0, 1.0])
    ema_d = [pairs[keys[1]][s] - pairs[keys[0]][s] for s in ("s17", "s42")]
    raw_d = [raw[keys[1]][s] - raw[keys[0]][s] for s in ("s17", "s42")]
    ema_mean = (pairs[keys[1]]["s17"] - pairs[keys[0]]["s17"] + pairs[keys[1]]["s42"] - pairs[keys[0]]["s42"]) / 2
    raw_mean = (raw[keys[1]]["s17"] - raw[keys[0]]["s17"] + raw[keys[1]]["s42"] - raw[keys[0]]["s42"]) / 2
    note(
        "fig3: EMA-selected e3-s17=%.4f e3-s42=%.4f e2-s17=%.4f e2-s42=%.4f -> paired mean %+.4f"
        % (pairs[keys[0]]["s17"], pairs[keys[0]]["s42"], pairs[keys[1]]["s17"], pairs[keys[1]]["s42"], ema_mean)
    )
    note(
        "fig3: raw       e3-s17=%.4f e3-s42=%.4f e2-s17=%.4f e2-s42=%.4f -> paired mean %+.4f"
        % (raw[keys[0]]["s17"], raw[keys[0]]["s42"], raw[keys[1]]["s17"], raw[keys[1]]["s42"], raw_mean)
    )

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.6), sharex=True)
    tick_labels = ["e3\n(no distill)", "e2\n(2-teacher distill)"]
    for ax, data, deltas, mean, title in (
        (axes[0], raw, raw_d, raw_mean, "(a) Raw weights"),
        (axes[1], pairs, ema_d, ema_mean, "(b) EMA-selected (headline)"),
    ):
        seed_style = {"s17": (C_RED, "o", 1.8, 12), "s42": (C_BLUE, "s", 1.5, -21)}
        for seed, (color, marker, lw, dy) in seed_style.items():
            ys = [data[k][seed] for k in keys]
            ax.plot(xs, ys, color=color, lw=lw, marker=marker, ms=6)
            for xi, yi in zip(xs, ys):
                ax.annotate(
                    f"{yi:.4f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, dy),
                    ha="center",
                    fontsize=9,
                    color=color,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.35, 1.35)
        lo = min(v for vals_ in data.values() for v in vals_.values())
        hi = max(v for vals_ in data.values() for v in vals_.values())
        pad = (hi - lo) * 0.26
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
        ax.set_xlabel("Recipe")
        ax.text(
            0.02,
            0.04,
            f"seed 17: {deltas[0]:+.4f}\nseed 42: {deltas[1]:+.4f}\npaired mean: {mean:+.4f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            linespacing=1.45,
            bbox=dict(fc="white", ec="#bbbbbb", lw=0.7, boxstyle="round,pad=0.3"),
        )
    axes[0].set_ylabel("Validation BPB")
    fig.legend(
        handles=[
            Line2D([], [], color=C_RED, marker="o", lw=1.8, label="seed 17"),
            Line2D([], [], color=C_BLUE, marker="s", lw=1.5, label="seed 42"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.87, bottom=0.22, wspace=0.34)
    fig.suptitle("Distillation ablation: paired e2 - e3, 7,200 steps, CPU fp32 validation", fontsize=10, y=0.99)
    save(fig, "distill_ablation.png")


# ---------------------------------------------------------------- figure 4
def fig_noise_floor() -> None:
    p3 = {
        r["run"]: float(r["validation_bpb"])
        for r in read_csv(RUNS / "phase3" / "results.csv")
        if r["validation_bpb"]
    }
    p4 = {
        r["run"]: float(r["validation_bpb"])
        for r in read_csv(RUNS / "phase4" / "results.csv")
        if r["validation_bpb"]
    }
    # Same recipe (w256 d6, Muon 0.05, dropout 0.0, 1200 steps, seed 17), independent processes.
    core = [
        ("s4-probe (phase 3)", p3["s4-probe"]),
        ("rep1", p4["rep1"]),
        ("rep2", p4["rep2"]),
        ("rep3", p4["rep3"]),
    ]
    # Same S4 recipe in the two EMA runs' raw arms (pooled n=6 in the report).
    pooled_extra = [("ema-0.999 raw arm", raw_bpb("phase4/ema-0.999")), ("ema-0.99 raw arm", raw_bpb("phase4/ema-0.99"))]

    # Four 7,200-step runs: two dropout-0.0 + two dropout-0.1 arms of the same S4 recipe.
    run7200 = [
        ("n7200-dr0", p4["n7200-dr0"], 0.0),
        ("n7200-dr0-rep", p4["n7200-dr0-rep"], 0.0),
        ("n7200-dr01 (phase 3)", p3["n7200-dr01"], 0.1),
        ("n7200-dr01-rep", p4["n7200-dr01-rep"], 0.1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 3.7))

    ax = axes[0]
    vals = np.array([v for _, v in core])
    m, sd, rng = vals.mean(), vals.std(ddof=1), vals.max() - vals.min()
    ax.axhspan(m - sd, m + sd, color=C_BLUE, alpha=0.13, lw=0)
    ax.axhline(m, color=C_BLUE, ls="--", lw=1.2)
    ax.plot(np.arange(len(core)), vals, "o", color=C_BLUE, ms=6)
    for i, (label, v) in enumerate(core):
        ax.annotate(
            f"{v:.4f}",
            (i, v),
            textcoords="offset points",
            xytext=(8, 4) if i == 0 else (0, 8),
            ha="left" if i == 0 else "center",
            fontsize=9,
            color=C_BLUE,
        )
    xextra = np.arange(len(core), len(core) + len(pooled_extra))
    pooled = np.append(vals, [v for _, v in pooled_extra])
    ax.plot(xextra, [v for _, v in pooled_extra], "D", mfc="none", mec="#b07a2a", ms=5.5)
    ax.set_ylim(min(vals.min(), pooled.min()) - 0.004, max(vals.max(), m + 1.7 * sd) + 0.002)
    ax.annotate(
        f"mean {m:.4f}",
        xy=(0.99, m),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=C_BLUE,
    )
    ax.text(
        0.5,
        0.03,
        f"sd {REPORT_SD_1200:.4f}, range {rng:.4f} (n=4)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
    )
    ax.set_xticks(np.arange(len(core) + len(pooled_extra)))
    ax.set_xticklabels(
        ["s4-probe", "rep1", "rep2", "rep3", "ema.999 raw", "ema.99 raw"], rotation=25, ha="right"
    )
    ax.set_xlim(-0.6, len(core) + len(pooled_extra) - 0.4)
    ax.set_title("(a) 1,200-step S4 repeats\n(n=4, +2 pooled)")
    ax.set_xlabel("Independent process, same seed 17")
    ax.set_ylabel("Validation BPB")

    ax = axes[1]
    vals7 = np.array([v for _, v, _ in run7200])
    m7, sd7, rng7 = vals7.mean(), vals7.std(ddof=1), vals7.max() - vals7.min()
    ax.axhspan(m7 - sd7, m7 + sd7, color=C_GREEN, alpha=0.13, lw=0)
    ax.axhline(m7, color=C_GREEN, ls="--", lw=1.2)
    for i, (label, v, dr) in enumerate(run7200):
        color = C_GREEN if dr == 0.1 else C_ORANGE
        ax.plot(i, v, "o", color=color, ms=6)
        ax.annotate(
            f"{v:.4f}",
            (i, v),
            textcoords="offset points",
            xytext=(8, 4) if i == 0 else (0, 8),
            ha="left" if i == 0 else "center",
            fontsize=9,
            color=color,
        )
    ax.plot([], [], "o", color=C_GREEN, label="dropout 0.1")
    ax.plot([], [], "o", color=C_ORANGE, label="dropout 0.0")
    ax.set_ylim(1.545, 1.573)
    ax.text(
        0.5,
        0.975,
        f"sd {sd7:.4f}, range {rng7:.4f} (n=4)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )
    ax.annotate(
        f"mean {m7:.4f}",
        xy=(0.99, m7),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=C_GREEN,
    )
    ax.set_xticks(np.arange(len(run7200)))
    ax.set_xticklabels([r[0] for r in run7200], rotation=25, ha="right")
    ax.set_xlim(-0.6, len(run7200) - 0.4)
    ax.set_title("(b) 7,200-step repeats\n(n=4, both dropout arms)")
    ax.set_xlabel("Run (same recipe, seed 17)")
    ax.set_ylabel("Validation BPB")
    ax.legend(loc="lower right", frameon=False)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.82, bottom=0.26, wspace=0.32)
    fig.suptitle("Run-to-run noise floor: identical recipes re-run in independent processes", fontsize=10, y=0.97)
    note(
        f"fig4 panel a: n=4 {np.round(vals, 4).tolist()} mean {m:.4f} sd_recomputed {sd:.5f} "
        f"(label uses report value {REPORT_SD_1200:.4f}) range {rng:.4f}; "
        f"pooled n=6 sd {np.append(vals, [v for _, v in pooled_extra]).std(ddof=1):.5f}"
    )
    note(
        f"fig4 panel b: n=4 {np.round(vals7, 4).tolist()} mean {m7:.4f} sd {sd7:.5f} range {rng7:.4f}"
    )
    save(fig, "noise_floor.png")


# ---------------------------------------------------------------- figure 5
def fig_assembly_ladder() -> None:
    # EMA-selected CPU fp32 validation, two seeds per recipe, in assembly order.
    recipes = [
        ("e2", "phase5/e2-s17", "phase5/e2-s42"),
        ("s7t", "phase6/s7t-s17", "phase6/s7t-s42"),
        ("s7x", "server/s7x-dr01-s17", "server/s7x-dr01-s42"),
        ("wd005", "server/s2-wd005-s17", "server/s2-wd005-s42"),
        ("combo", "server/s2-combo-s17", "server/s2-combo-s42"),
        ("x28800", "server/s3-x28800-s17", "server/s3-x28800-s42"),
        ("ba2", "server/s3-ba2-s17", "server/s3-ba2-s42"),
        ("ba2x", "server/s4-ba2x-s17", "server/s4-ba2x-s42"),
        ("gen3", "server/s5-gen3-s17", "server/s5-gen3-s42"),
        ("gen3dr005", "server/s6-gen3dr005-s17", "server/s6-gen3dr005-s42"),
    ]
    # Expected EMA-selected values, as supplied for this figure (cross-checked against metrics.json).
    expected = [
        (1.475163, 1.470156),
        (1.471864, 1.455224),
        (1.461959, 1.453082),
        (1.434193, 1.433366),
        (1.428289, 1.428926),
        (1.424291, 1.423180),
        (1.423840, 1.424576),
        (1.416682, 1.416350),
        (1.413448, 1.415547),
        (1.408663, 1.407431),
    ]
    # Baseline of each step: index of the generation it is paired against; [i, j] = mean of both.
    # Indices: 0 e2, 1 s7t, 2 s7x, 3 wd005, 4 combo, 5 x28800, 6 ba2, 7 ba2x, 8 gen3, 9 gen3dr005.
    # s3-ba2 keeps s2-combo's 14,400-step schedule and swaps only the teacher pair, and s4-ba2x moves
    # both axes at once (born-again teachers + 28,800 steps), so its baseline is the two s3 generations.
    baselines = [0, 1, 2, 3, 4, 4, [5, 6], 7, 8]

    vals = []
    for label, a, b in recipes:
        va, vb = ema_bpb(a), ema_bpb(b)
        vals.append((label, va, vb, (va + vb) / 2, abs(va - vb) / 2))
    means = np.array([v[3] for v in vals])
    errs = np.array([v[4] for v in vals])
    seeds17 = np.array([v[1] for v in vals])
    seeds42 = np.array([v[2] for v in vals])
    for (label, va, vb, _, _), (e17, e42) in zip(vals, expected):
        if round(va, 6) != e17 or round(vb, 6) != e42:
            note(f"fig5 MISMATCH {label}: metrics.json {va:.6f}/{vb:.6f} vs expected {e17}/{e42}")

    diffs = []
    for step, base in enumerate(baselines, start=1):
        bases = base if isinstance(base, list) else [base]
        d17 = seeds17[step] - float(np.mean([seeds17[b] for b in bases]))
        d42 = seeds42[step] - float(np.mean([seeds42[b] for b in bases]))
        diffs.append((d17, d42, (d17 + d42) / 2, isinstance(base, list)))

    note("fig5 x labels: " + " ".join(v[0] for v in vals))
    note("fig5 seeds17: " + " ".join(f"{v:.6f}" for v in seeds17))
    note("fig5 seeds42: " + " ".join(f"{v:.6f}" for v in seeds42))
    note(
        "fig5 paired mean diffs (new - baseline): "
        + " ".join(f"{v[2]:+.4f}{'**' if v[3] else ''}" for v in diffs)
    )
    for label, (d17, d42, dm, multi) in zip([v[0] for v in vals[1:]], diffs):
        note(f"fig5   {label:10s} vs s17 {d17:+.4f} / s42 {d42:+.4f} -> mean {dm:+.4f}{' (two-gen baseline)' if multi else ''}")

    xs = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(FIGW, 4.3))
    ax.plot(xs, means, "-", color=C_BLUE, lw=1.8, zorder=2)
    ax.errorbar(xs, means, yerr=errs, fmt="o", color=C_BLUE, ms=7, capsize=4, lw=1.4, zorder=3,
                label="2-seed mean (error bar: seed range)")
    ax.plot(xs - 0.09, seeds17, "o", mfc="none", mec=C_RED, ms=4.5, zorder=4, label="seed 17")
    ax.plot(xs + 0.09, seeds42, "s", mfc="none", mec=C_GREEN, ms=4.5, zorder=4, label="seed 42")
    for i, (label, va, vb, mean, _) in enumerate(vals):
        ax.annotate(f"{mean:.4f}", (i, mean), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=9, color=C_BLUE)
    for i, (d17, d42, dm, multi) in enumerate(diffs):
        ax.annotate(
            f"$\\Delta$ {dm:+.4f}{'**' if multi else ''}",
            ((i + i + 1) / 2, (means[i] + means[i + 1]) / 2),
            textcoords="offset points",
            xytext=(0, -25),
            ha="center",
            fontsize=9,
            color="#333333",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
        )
    ax.annotate(
        "** s4-ba2x: $\\Delta$ vs the mean\nof the two s3 generations",
        xy=(7, means[7]),
        xytext=(4.5, 1.4525),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9, shrinkA=2, shrinkB=4),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([v[0] for v in vals])
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_ylim(1.40, 1.49)
    ax.set_xlabel("Recipe generation (one knob changed per step, two seeds each)")
    ax.set_ylabel("Validation BPB (EMA-selected, CPU fp32)")
    ax.set_title("Assembly ladder of the final recipe (S5.5) — final")
    ax.legend(loc="upper right", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.36)
    fig.text(
        0.01,
        0.02,
        "Segment labels: mean of the two seeds' paired differences (new generation - predecessor).\n"
        "* s3-ba2 keeps the 14,400-step schedule of s2-combo and swaps only the teacher pair, so its\n"
        "  baseline is s2-combo; vs s3-x28800 the paired mean is +0.0009, inside the 0.0074 repeat sd.\n"
        "** s4-ba2x moves two axes at once (born-again teachers + 28,800 steps): its Δ is measured\n"
        "   against the mean of the two s3 generations; vs s3-ba2 -0.0072/-0.0082, vs s3-x28800\n"
        "   -0.0076/-0.0068.\n"
        "Final frozen champion: s6-gen3dr005-s42, val 1.407431, test 1.426939.",
        fontsize=9,
        color="#4d4d4d",
    )
    save(fig, "assembly_ladder.png")


FIGURES = {
    "position_nll": ("position_nll.png", fig_position_nll),
    "val_curves_phase5": ("val_curves_phase5.png", fig_val_curves_phase5),
    "distill_ablation": ("distill_ablation.png", fig_distill_ablation),
    "noise_floor": ("noise_floor.png", fig_noise_floor),
    "assembly_ladder": ("assembly_ladder.png", fig_assembly_ladder),
}


def main() -> None:
    setup_style()
    note(f"CODE_ROOT = {CODE_ROOT}")
    names = _only or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    for name in names:
        FIGURES[name][1]()
    note(f"done: {len(names)} figure(s)")


if __name__ == "__main__":
    main()
