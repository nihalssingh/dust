"""Exactness tests for fastdust.py on CPU in float32 with a small config.

E options must reproduce dust.DUST draw for draw (same seed, same noise).
    python mac/test_fastdust.py
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dust  # noqa: E402
import fastdust  # noqa: E402
from model import GPTConfig  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / 'data' / 'val.pt'


class NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def setup():
    config = GPTConfig(sequence_len=256, n_embd=128, n_head=8, n_kv_head=8, n_layer=8)
    rows = torch.load(DATA, weights_only=True)[:2, :257].long()
    return config, rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def estimates(make_estimator, config, x, y):
    model = dust.make_model(config, torch.device('cpu'), seed=42)
    estimator = make_estimator(model)
    estimator.step(x, y, NoStep())
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}, estimator


def compare(a, b, label, tol):
    worst, name_worst = 0.0, None
    for name in a:
        rel = ((a[name] - b[name]).norm() / a[name].norm().clamp_min(1e-30)).item()
        if rel > worst:
            worst, name_worst = rel, name
    status = 'PASS' if worst <= tol else 'FAIL'
    print(f'{status} {label}: worst relative difference {worst:.2e} ({name_worst})')
    return worst <= tol


def main():
    torch.manual_seed(0)
    config, x, y = setup()
    settings = dust.recipe('1M', 256)
    ok = True
    reference, _ = estimates(lambda m: dust.DUST(m, settings, seed=42), config, x, y)
    for truncate in (False, True):
        options = fastdust.Options(truncate=truncate)
        fast, _ = estimates(lambda m: fastdust.FastDUST(m, settings, options, seed=42), config, x, y)
        ok &= compare(reference, fast, f'E truncate={truncate} vs dust.DUST (bitwise)', 0.0)

    # The compiled per-chunk local/head kernels, run uncompiled, against the reference.
    def per_chunk(m):
        est = fastdust.FastDUST(m, settings, fastdust.Options(), seed=42)
        est.options.compile = True  # take the per-chunk paths...
        est._local_fn, est._head_chunk_fn = fastdust._local_chunk, fastdust._head_chunk  # ...uncompiled
        return est
    chunked, _ = estimates(per_chunk, config, x, y)
    ok &= compare(reference, chunked, 'E per-chunk local/head kernels vs dust.DUST (bitwise)', 0.0)

    # Query-chunked sliding-window attention against one band-masked SDPA, at lengths that trigger chunking.
    import model as model_lib
    worst = 0.0
    for T, window in ((1024, 512), (1000, 300), (700, 64)):
        q, k, v = (torch.randn(2, T, 4, 16) for _ in range(3))
        model_lib.WINDOW_CHUNK = 0
        ref_out = model_lib.attention(q, k, v, window)
        model_lib.WINDOW_CHUNK = 256
        worst = max(worst, (model_lib.attention(q, k, v, window) - ref_out).abs().max().item())
    print(f"{'PASS' if worst < 1e-5 else 'FAIL'} E chunked window attention vs single mask: max |diff| {worst:.1e}")
    ok &= worst < 1e-5
    print('ALL PASSED' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
