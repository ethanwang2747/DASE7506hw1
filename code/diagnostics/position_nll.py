"""Per-position next-token NLL diagnostic for a trained checkpoint.

Splits a split into exactly the same non-overlapping 256-token windows the scorer uses, forwards
each window once and averages the cross-entropy of every position across windows. The evaluator is
untouched: this script only imports common.load_data / common.make_model / common.windows and
reimplements the accumulation, so `sum(position_nll * count)` reproduces the scorer's total NLL.

    env -u PYTHONPATH .venv/bin/python diagnostics/position_nll.py \
        --checkpoint runs/e0-baseline/checkpoint.pt --output-dir runs/phase3/diagnostics
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import PROTOCOL, autocast, device_metrics, load_data, make_model, setup, sha, windows


def position_statistics(model, tokens, device, precision, batch_size, context):
    """Accumulate per-position NLL and target counts over the scorer's windows."""
    previous_mode = model.training
    model.eval()
    nll = torch.zeros(context, dtype=torch.float64)
    count = torch.zeros(context, dtype=torch.long)
    nll_total, targets, windows_seen = 0., 0, 0
    with torch.no_grad():
        for x, y in windows(tokens, batch_size, context):
            x, y = x.to(device), y.to(device)
            with autocast(device, precision):
                logp = model.predict_log_probs(x).float()
            if logp.shape != (*x.shape, 2048) or not torch.isfinite(logp).all():
                raise ValueError('Return finite log probabilities of shape [batch, time, 2048].')
            if torch.logsumexp(logp, dim=-1).abs().max().item() > 1e-3:
                raise ValueError('The output is not a normalized probability distribution.')
            valid = y != -100
            losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1) * valid
            nll += losses.double().sum(0).cpu()
            count += valid.sum(0).cpu()
            nll_total += losses.double().sum().item()
            targets += int(valid.sum().item())
            windows_seen += int(x.shape[0])
    model.train(previous_mode)
    return nll, count, nll_total, targets, windows_seen


def summarize(nll, count, nll_total, targets, byte_count, windows_seen, head):
    mean = torch.where(count > 0, nll / count.clamp_min(1), torch.zeros_like(nll))
    head = min(head, len(mean) // 2)
    front_mask = torch.zeros(len(mean), dtype=torch.bool)
    front_mask[:head] = True
    front = nll[front_mask].sum().item() / count[front_mask].sum().item()
    tail = nll[-head:].sum().item() / count[-head:].sum().item()
    rest = nll[~front_mask].sum().item() / count[~front_mask].sum().item()
    first32 = nll[:2 * head].sum().item() / count[:2 * head].sum().item()
    summary = {
        'positions': len(mean),
        'windows': windows_seen,
        'targets': targets,
        'head_positions': head,
        f'front{head}_mean_nll_nats': front,
        f'last{head}_mean_nll_nats': tail,
        f'front{head}_over_last{head}_ratio': front / tail,
        f'rest_of_window_mean_nll_nats': rest,
        f'front{head}_over_rest_ratio': front / rest,
        f'first{2 * head}_mean_nll_nats': first32,
        'overall_mean_nll_nats': nll_total / targets,
        'bpb': nll_total / math.log(2) / byte_count,
        'front_share_of_total_nll': nll[front_mask].sum().item() / nll_total,
        'front_token_share': count[front_mask].sum().item() / targets,
        'cold_start_excess_share_of_total': ((front - tail) * head / len(mean)) / (nll_total / targets),
        'position0_mean_nll_nats': mean[0].item(),
        'position255_mean_nll_nats': mean[-1].item(),
    }
    return mean, summary


def write_csv(path, mean, count):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['position', 'mean_nll', 'count'])
        for position in range(len(mean)):
            writer.writerow([position, f'{mean[position].item():.6f}', int(count[position].item())])


def curves_from_directory(directory, suffix, skip_stem):
    """Curves already stored as CSVs in `directory`, matching `suffix` (e.g. `-validation_cpu_fp32`)."""
    curves = []
    for path in sorted(directory.glob(f'*{suffix}.position-nll.csv')):
        stem = path.name.replace('.position-nll.csv', '')
        if stem == skip_stem:
            continue
        table = list(csv.DictReader(path.open()))
        if not table:
            continue
        label = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        if label.endswith('-checkpoint'):
            label = label[:-len('-checkpoint')]
        curves.append((label, [int(row['position']) for row in table],
                       [float(row['mean_nll']) for row in table]))
    return curves


def write_plot(path, series):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is unavailable; skipping the plot', file=sys.stderr)
        return None
    window = 4
    figure, axes = plt.subplots(figsize=(10, 5))
    for index, (label, positions, values) in enumerate(series):
        axes.plot(positions, values, linewidth=1.3, label=label, color=f'C{index}')
        if len(values) >= window:
            smooth = [sum(values[start:start + window]) / window
                      for start in range(len(values) - window + 1)]
            axes.plot(positions[window - 1:], smooth, linewidth=2.2, alpha=.6,
                      color=f'C{index}', linestyle='--')
    axes.set_xlabel('position in the 256-token window')
    axes.set_ylabel('mean next-token NLL (nats)')
    axes.set_title('Per-position loss on the validation split (dashed: 4-token moving average)')
    axes.grid(alpha=.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--head', type=int, default=16, help='Positions compared as front vs back.')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--label', help='Curve label; defaults to the run directory name.')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()
    device, precision = setup(args.device, args.precision, args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    tokens, byte_count = load_data()[args.split]
    context = model.context
    nll, count, nll_total, targets, windows_seen = position_statistics(
        model, tokens, device, precision, args.batch_size, context)
    mean, summary = summarize(nll, count, nll_total, targets, byte_count, windows_seen, args.head)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.checkpoint.parent.name}-{args.checkpoint.stem}-{args.split}_{device.type}_{precision}'
    csv_path = args.output_dir / f'{stem}.position-nll.csv'
    json_path = args.output_dir / f'{stem}.summary.json'
    write_csv(csv_path, mean, count)
    summary.update(protocol=PROTOCOL, split=args.split, precision=precision,
                   checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                   implementation_sha256=implementation_sha, evaluator_sha256=sha(
                       Path(__file__).resolve().parent.parent / 'evaluate.py'),
                   csv=str(csv_path), **device_metrics(device))
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    plot_path = None
    if not args.no_plot:
        label = args.label or args.checkpoint.parent.name
        plot_path = write_plot(args.output_dir / f'{stem}.png',
                               curves_from_directory(args.output_dir, f'-{args.split}_{device.type}_{precision}',
                                                     stem)
                               + [(label, list(range(len(mean))), mean.tolist())])
    print(json.dumps(summary | ({} if plot_path is None else {'plot': str(plot_path)}), indent=2))


if __name__ == '__main__':
    main()
