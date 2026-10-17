"""Peak transient memory of one update (above the resident model) for inference, backprop, backprop with
per-block activation checkpointing, and DUST variants, on one batch of 8 x 2048 tokens.

    python mac/memory_probe.py --out ../runs/bench/memory.json
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
import dust  # noqa: E402
import fastdust  # noqa: E402
from model import GPTConfig  # noqa: E402
import backprop  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / 'data' / 'val.pt'


class Peak:
    def __enter__(self):
        torch.mps.synchronize()
        torch.mps.empty_cache()
        self.base = torch.mps.current_allocated_memory()
        self.peak, self.running = self.base, True
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def poll(self):
        while self.running:
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(0.0002)

    def __exit__(self, *exc):
        torch.mps.synchronize()
        self.running = False
        self.thread.join()
        self.gb = (self.peak - self.base) / 1e9


class NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('mps')
    rows = torch.load(DATA, weights_only=True)[:8].long().to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    results = {}

    model = dust.make_model(GPTConfig(), device, 42)
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    results['model weights (resident)'] = weights_gb
    for _ in range(2):
        with Peak() as m, torch.no_grad(), torch.autocast('mps', dtype=torch.bfloat16):
            model(x, y)
    results['inference forward'] = m.gb

    for label, ckpt in (('backprop (SGD step)', False), ('backprop + per-block checkpointing', True)):
        bp = dust.make_model(GPTConfig(), device, 42).requires_grad_(True)
        opt = backprop.make_optimizer(bp, 0.175, 0.98)
        backprop.train_step(bp, opt, x, y, ckpt)  # allocate optimizer state outside the measurement
        with Peak() as m:
            backprop.train_step(bp, opt, x, y, ckpt)
        results[label] = m.gb
        del bp, opt

    settings = dust.recipe('1M', 256)
    configs = {
        'DUST reference (dust.py)': lambda mdl: dust.DUST(mdl, settings, seed=1),
        'DUST E (fastdust, truncated)': lambda mdl: fastdust.FastDUST(mdl, settings, fastdust.Options(), seed=1),
        'DUST A1 best': lambda mdl: fastdust.FastDUST(mdl, settings, fastdust.Options(
            exact_head=True, derive_hidden=True, derive_attn_out=True, block_output_draws=True,
            derive_embedding=True, closed_form_local=True, loo_baseline=True, exact_last_block=True), seed=1),
        'DUST A1 best+': lambda mdl: fastdust.FastDUST(mdl, settings, fastdust.Options(
            exact_head=True, derive_hidden=True, derive_attn_out=True, block_output_draws=True,
            derive_embedding=True, closed_form_local=True, loo_baseline=True, exact_last_block=True,
            skip_control_variate=True, orthogonal_scalar_es=True), seed=1),
    }
    for label, make in configs.items():
        mdl = dust.make_model(GPTConfig(), device, 42)
        est = make(mdl)
        est.step(x, y, NoStep())
        est.inputs.clear()
        est.outputs.clear()
        for name in ('mix', 'mid', 'o', 've', 'out'):
            if hasattr(est, name):
                setattr(est, name, [])
        with Peak() as m:
            est.step(x, y, NoStep())
        results[label] = m.gb
        est.close()
        del est, mdl
    for k, v in results.items():
        print(f'{k:40s} {v:6.2f} GB')
    args.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
