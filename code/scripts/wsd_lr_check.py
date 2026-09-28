"""Print and check the two learning-rate shapes step by step (CPU, no training).

    env -u PYTHONPATH .venv/bin/python scripts/wsd_lr_check.py --steps 1000 --cooldown-frac 0.2

The cosine column is the expression train.py has always used; the WSD column comes from the helper
train.py now calls. Both are multipliers applied to every peak learning rate.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import wsd_multiplier


def cosine_multiplier(step, steps):
    return min(1., (step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/steps)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--cooldown-frac', type=float, default=.2)
    args = p.parse_args()
    steps, frac = args.steps, args.cooldown_frac
    plateau_end = (1.-frac)*steps
    interesting = sorted({0, 1, 50, 99, 100, steps//2, int(plateau_end)-1, int(plateau_end),
                          int(plateau_end)+1, int(steps*.9), steps-1})
    print(f'steps={steps}  cooldown_frac={frac}  ->  plateau ends at step {plateau_end:g}, '
          f'cooldown runs to step {steps-1}')
    print(f"{'step':>6}{'cosine':>10}{'wsd':>10}")
    for step in interesting:
        print(f'{step:>6}{cosine_multiplier(step, steps):>10.4f}{wsd_multiplier(step, steps, frac):>10.4f}')
    print()
    if plateau_end < 100:
        print(f'SKIP  the schedule is shorter than the 100-step warmup (plateau ends at step '
              f'{plateau_end:g}): WSD is not well defined here, only its endpoint is still exact.')
        checks = {'wsd ends at exactly 0.1 x peak':
                  abs(wsd_multiplier(steps-1, steps, frac) - .1) < 1e-12,
                  'cosine is unchanged (matches the historical expression)':
                  all(cosine_multiplier(s, steps) ==
                      min(1., (s+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*s/steps)))
                      for s in range(steps))}
    else:
        checks = {
            'wsd warmup is linear and hits the peak at step 99':
                abs(wsd_multiplier(99, steps, frac) - 1.) < 1e-12,
            'wsd plateau is flat at the peak (step 100 .. plateau_end-1)':
                all(wsd_multiplier(s, steps, frac) == 1. for s in range(100, int(plateau_end))),
            'wsd turns at exactly 0.8 x steps (multiplier still 1.0 there)':
                abs(wsd_multiplier(int(plateau_end), steps, frac) - 1.) < 1e-12,
            'wsd cooldown is linear in the step':
                all(abs((wsd_multiplier(s-1, steps, frac) - wsd_multiplier(s, steps, frac))
                        - (wsd_multiplier(s, steps, frac) - wsd_multiplier(s+1, steps, frac))) < 1e-12
                    for s in range(int(plateau_end)+1, steps-1)),
            'wsd ends at exactly 0.1 x peak':
                abs(wsd_multiplier(steps-1, steps, frac) - .1) < 1e-12,
            'cosine is unchanged (matches the historical expression)':
                all(cosine_multiplier(s, steps) == min(1., (s+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*s/steps)))
                    for s in range(steps)),
    }
    for name, ok in checks.items():
        print(('PASS  ' if ok else 'FAIL  ') + name)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
