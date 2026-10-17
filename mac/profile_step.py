"""Time each phase of one DUST update on a real batch (after a warm-up update that pays for compilation).

    python mac/profile_step.py [--population 256] [--no-compile]
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dust  # noqa: E402
from model import GPTConfig  # noqa: E402


class NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def timed(estimator, name, totals, device):
    method = getattr(estimator, name)

    def wrapper(*args, **kwargs):
        sync(device)
        start = time.perf_counter()
        out = method(*args, **kwargs)
        sync(device)
        key = name if name != 'direct_errors' else f'direct_errors[{args[0][0].split(".")[-1] if "." in args[0][0] else args[0][0][:2]}]'
        totals[key] += time.perf_counter() - start
        return out
    setattr(estimator, name, wrapper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', type=int, default=256)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--config', help='a mac/bench.py config name (FastDUST variants)')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'train_1M.pt')
    args = parser.parse_args()
    device = torch.device('mps')
    rows = torch.load(args.data, weights_only=True)[:8].long().to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    model = dust.make_model(GPTConfig(), device, 42)
    if not args.no_compile:
        dust.compile_model(model)
    if args.config:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bench
        import fastdust
        _, options, settings = bench.CONFIGS[args.config]
        estimator = fastdust.FastDUST(model, settings, options, seed=42)
    else:
        estimator = dust.DUST(model, dust.recipe('1M', args.population), seed=42)
    sync(device)
    start = time.perf_counter()
    estimator.step(x, y, NoStep())
    sync(device)
    print(f'warm-up update (includes compilation): {time.perf_counter() - start:.1f}s', flush=True)
    totals = defaultdict(float)
    for name in ('capture', 'direct_errors', 'head_error', 'local_errors', 'scalar_gradients', 'reassemble',
                 'closed_form_local_errors', 'attention_backward', 'mix_errors', 'last_block_error'):
        if hasattr(estimator, name):
            timed(estimator, name, totals, device)
    sync(device)
    start = time.perf_counter()
    estimator.step(x, y, NoStep())
    sync(device)
    total = time.perf_counter() - start
    print(f'update: {total:.1f}s')
    for key, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f'   {key:28s} {seconds:7.2f}s  {100 * seconds / total:5.1f}%')
    print(f'   {"(other)":28s} {total - sum(totals.values()):7.2f}s')


if __name__ == '__main__':
    main()
