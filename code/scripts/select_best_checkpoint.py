"""Score every checkpoint in a run directory and keep the best one.

    env -u PYTHONPATH .venv/bin/python scripts/select_best_checkpoint.py runs/phase6/hp-dr02

Every `checkpoint*.pt` in the directory (excluding the output itself) is scored with the untouched
evaluator, CPU fp32, on the validation split — the same measurement the run was judged on, so no test
data is involved and the frozen-scoring discipline holds. The winner is copied to
`<run-dir>/checkpoint-best.pt`; the originals are left exactly as they are.

This exists because `--save-every` overwrites the rolling `checkpoint.pt`, and a run whose validation
curve turns back up (the 18-20 M teachers did) would otherwise only be able to pay for the rebound:
with the per-point snapshots the pre-rebound weights can still be picked afterwards.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import PROTOCOL, load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--pattern', default='checkpoint*.pt')
    p.add_argument('--output-name', default='checkpoint-best.pt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    device, precision = setup(args.device, args.precision, args.threads)
    data = load_data()
    candidates = sorted(path for path in args.run_dir.glob(args.pattern)
                        if path.name != args.output_name and path.is_file())
    if not candidates:
        p.error(f'no checkpoints matching {args.pattern} in {args.run_dir}')
    results = []
    for path in candidates:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['protocol'] != PROTOCOL:
            raise ValueError(f'{path} belongs to a different course protocol.')
        model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
        model.load_state_dict(checkpoint['model'])
        scored = score(model, *data['validation'], device, precision)
        results.append((scored['bpb'], path, checkpoint['train_tokens'] // (32 * 256), scored['seconds']))
    results.sort(key=lambda row: row[0])
    print(f"{'checkpoint':<40}{'step':>8}{'val BPB':>10}{'seconds':>9}")
    for bpb, path, step, seconds in results:
        print(f'{path.name:<40}{step:>8}{bpb:>10.4f}{seconds:>9.2f}')
    best = args.run_dir / args.output_name
    shutil.copyfile(results[0][1], best)
    print(f'\nbest: {results[0][1].name} (val BPB {results[0][0]:.4f}) -> copied to {best}')


if __name__ == '__main__':
    main()
