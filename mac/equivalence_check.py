"""Run one DUST update on CPU in float32 with a small config and save the loss and gradient estimates.

Run it once against the upstream checkout and once against this port; with the same seed the noise draws are
identical, so the estimates should agree to float32 round-off.
    python mac/equivalence_check.py --repo /path/to/upstream --out upstream.pt
    python mac/equivalence_check.py --repo . --out port.pt
    python mac/equivalence_check.py --compare upstream.pt port.pt
"""
import argparse
import sys
from pathlib import Path

import torch


def run(repo, out, data):
    sys.path.insert(0, str(Path(repo).resolve()))
    import dust
    from model import GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(sequence_len=256, n_embd=128, n_head=8, n_kv_head=8, n_layer=8)
    model = dust.make_model(config, torch.device('cpu'), seed=42)
    rows = torch.load(data, weights_only=True)[:2, :257].long()
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    with torch.no_grad():
        clean = model(x, y).item()
    settings = dust.recipe('1M', 256)
    estimator = dust.DUST(model, settings, seed=42)

    class NoStep:
        def zero_grad(self, set_to_none=True):
            pass

        def step(self):
            pass

    estimator.step(x, y, NoStep())
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    torch.save(dict(clean=clean, grads=grads), out)
    print(f'{repo}: clean loss {clean:.6f}, {len(grads)} gradient estimates -> {out}')


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    print(f"clean loss: {a['clean']:.7f} vs {b['clean']:.7f} (diff {abs(a['clean'] - b['clean']):.2e})")
    worst = 0.0
    for name, ga in a['grads'].items():
        gb = b['grads'][name].to(ga.dtype)
        rel = ((ga - gb).norm() / ga.norm().clamp_min(1e-30)).item()
        cos = torch.nn.functional.cosine_similarity(ga.flatten(), gb.flatten(), dim=0).item()
        worst = max(worst, rel)
        print(f'{name:45s} rel diff {rel:.2e}  cos {cos:.7f}')
    print(f'WORST relative difference: {worst:.2e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo')
    parser.add_argument('--out')
    parser.add_argument('--data', default=str(Path(__file__).resolve().parents[2] / 'data' / 'val.pt'))
    parser.add_argument('--compare', nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.repo, args.out, args.data)
