"""Exactness tests for fastdust.py on CPU in float32 with a small config.

E options must reproduce dust.DUST draw for draw (same seed, same noise). A1 options are only checked for
running and, where the answer is known exactly (the exact head), against backprop.
    python mac/test_fastdust.py
"""
from dataclasses import replace
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

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


def cosine(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


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

    # Backprop gradient of the same small model, for the exact-head check.
    model = dust.make_model(config, torch.device('cpu'), seed=42).requires_grad_(True)
    model(x, y).backward()
    true = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    checks = [
        ('A1 exact_head', fastdust.Options(exact_head=True)),
        ('A1 derive_hidden', fastdust.Options(derive_hidden=True)),
        ('A1 derive_attn_out', fastdust.Options(derive_attn_out=True)),
        ('A1 antithetic', fastdust.Options(antithetic=True)),
        ('A1 rademacher', fastdust.Options(rademacher=True)),
    ]
    for label, options in checks:
        grads, _ = estimates(lambda m: fastdust.FastDUST(m, settings, options, seed=42), config, x, y)
        finite = all(torch.isfinite(g).all() for g in grads.values())
        head = cosine(grads['lm_head.weight'], true['lm_head.weight'])
        fc = sum(cosine(grads[f'transformer.h.{i}.mlp.c_fc.weight'], true[f'transformer.h.{i}.mlp.c_fc.weight'])
                 for i in range(8)) / 8
        print(f'     {label}: finite={finite}, cos(head, backprop)={head:.4f}, mean cos(c_fc, backprop)={fc:.4f}')
        ok &= finite
        if options.exact_head:
            ratio = (grads['lm_head.weight'].norm() / true['lm_head.weight'].norm()).item()
            exact = head > 0.9999 and abs(ratio - 0.5) < 1e-3  # batch of 2: shrink (b - 1) / b = 1/2
            print(f'{"PASS" if exact else "FAIL"} exact head: cos {head:.6f}, norm ratio {ratio:.4f} (expect 0.5)')
            ok &= exact
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

    # nrep = 1 has no centering, so its scale is 1 (not (n - 1) / n = 0); antithetic + LOO is rejected.
    one = replace(settings, writers=[2] * 8, hidden=[2] * 8, hubs=[2] * 8, embedding=2, head=16 * 1, nrep=1,
                  local=dict(q=1, v=1, gate=1, k=1, ve=1))
    grads, _ = estimates(lambda m: fastdust.FastDUST(m, one, fastdust.Options(
        loo_baseline=True, block_output_draws=True, exact_head=True, exact_last_block=True), seed=42), config, x, y)
    nonzero = all(grads[f'transformer.h.{i}.mlp.c_proj.weight'].abs().sum() > 0 for i in range(8))
    print(f"{'PASS' if nonzero else 'FAIL'} nrep=1 estimates are not zeroed by the (n - 1) / n scale")
    ok &= nonzero
    try:
        fastdust.FastDUST(dust.make_model(config, torch.device('cpu'), seed=42), settings,
                          fastdust.Options(antithetic=True, loo_baseline=True))
        rejected = False
    except ValueError:
        rejected = True
    print(f"{'PASS' if rejected else 'FAIL'} antithetic + loo_baseline is rejected")
    ok &= rejected
    print('ALL PASSED' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
