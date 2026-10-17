"""Benchmark DUST variants: seconds per update, peak memory and gradient quality on one fixed batch.

Quality is the cosine of each parameter's estimate with two float32 backprop targets (see cosine_probe.py):
the full gradient and the same-token gradient that DUST's direct-loss sites converge to.

    python mac/bench.py targets --out ../runs/bench/targets_init.pt [--checkpoint ckpt.pt]
    python mac/bench.py run --config E+N --targets ../runs/bench/targets_init.pt --out ../runs/bench/E+N.json
    python mac/bench.py table ../runs/bench/*.json
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dust  # noqa: E402
import fastdust  # noqa: E402
from fastdust import Options  # noqa: E402
from model import GPTConfig  # noqa: E402
import cosine_probe  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / 'data' / 'val.pt'
TYPES = ['head', 'attention out', 'MLP out', 'MLP in', 'token embedding', 'query', 'key', 'value', 'value gate',
         'value embedding', 'residual scalars']


def scaled(settings, **factors):
    """Multiply recipe streams (writers, hidden, hubs, embedding, head, local) by factors, never rounding a
    nonzero count below one chunk, and keeping head draws a multiple of nrep x the number of slabs."""
    n = settings.nrep
    unit = {'head': n * (GPTConfig().vocab_size // 256)}
    chunk = lambda v, f, u: max(1, round(v * f / u)) * u if v else 0
    changes = {}
    for stream, f in factors.items():
        value = getattr(settings, stream)
        u = unit.get(stream, n)
        if isinstance(value, list):
            changes[stream] = [chunk(v, f, u) for v in value]
        elif isinstance(value, dict):
            changes[stream] = {k: chunk(v, f, u) for k, v in value.items()}
        else:
            changes[stream] = chunk(value, f, u)
    return replace(settings, **changes)


def drop_caches(estimator):
    """Free the previous update's clean-pass caches, so a memory measurement starts from model + optimizer."""
    estimator.inputs.clear()
    estimator.outputs.clear()
    for name in ('mix', 'mid', 'o', 've', 'out'):
        if hasattr(estimator, name):
            setattr(estimator, name, [])
    torch.mps.empty_cache()


BASE = dust.recipe('1M', 256)
CONFIGS = {
    # name: (implementation, options, recipe)
    'reference-eager': ('dust', None, BASE),
    'reference+N': ('dust-compiled', None, BASE),
    'E-eager': ('fast', Options(), BASE),
    'E+N': ('fast', Options(compile=True), BASE),
    'E+N no-truncate': ('fast', Options(compile=True, truncate=False), BASE),
    'A1 exact_head': ('fast', Options(compile=True, exact_head=True), BASE),
    'A1 derive_hidden': ('fast', Options(compile=True, derive_hidden=True), BASE),
    'A1 derive_attn_out': ('fast', Options(compile=True, derive_attn_out=True), BASE),
    'A1 antithetic': ('fast', Options(compile=True, antithetic=True), BASE),
    'A1 rademacher': ('fast', Options(compile=True, rademacher=True), BASE),
    'A1 all': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                antithetic=True), BASE),
    'A1 block_output_draws': ('fast', Options(compile=True, block_output_draws=True), BASE),
    'A1 all+': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                 block_output_draws=True, antithetic=True), BASE),
    'A2 hybrid': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                   block_output_draws=True, antithetic=True, local_attention_backward=True), BASE),
    'A1 anchored': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                     block_output_draws=True, derive_embedding=True, antithetic=True), BASE),
    'A2 anchored hybrid': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                            block_output_draws=True, derive_embedding=True, antithetic=True,
                                            local_attention_backward=True, scalars_from_errors=True), BASE),
    'A1 anchored+cf': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                        block_output_draws=True, derive_embedding=True, antithetic=True,
                                        closed_form_local=True), BASE),
    'A2 anchored+cf+scalars': ('fast', Options(compile=True, exact_head=True, derive_hidden=True,
                                                derive_attn_out=True, block_output_draws=True, derive_embedding=True,
                                                antithetic=True, closed_form_local=True, scalars_from_errors=True),
                               BASE),
    'A1 best': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                 block_output_draws=True, derive_embedding=True, closed_form_local=True,
                                 loo_baseline=True, exact_last_block=True), BASE),
    'A2 best': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                 block_output_draws=True, derive_embedding=True, closed_form_local=True,
                                 loo_baseline=True, exact_last_block=True, scalars_from_errors=True), BASE),
    'A2 best+attn': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                      block_output_draws=True, derive_embedding=True, local_attention_backward=True,
                                      loo_baseline=True, exact_last_block=True, scalars_from_errors=True), BASE),
    'A1 best+': ('fast', Options(compile=True, exact_head=True, derive_hidden=True, derive_attn_out=True,
                                  block_output_draws=True, derive_embedding=True, closed_form_local=True,
                                  loo_baseline=True, exact_last_block=True, skip_control_variate=True,
                                  orthogonal_scalar_es=True), BASE),
    'A2 freeze_attention': ('fast', Options(compile=True, freeze_attention=True), BASE),
    'A2 freeze_scalars': ('fast', Options(compile=True, freeze_scalars=True), BASE),
}


LADDER = ['exact_head', 'derive_hidden', 'derive_attn_out', 'block_output_draws', 'derive_embedding',
          'closed_form_local', 'loo_baseline', 'exact_last_block', 'skip_control_variate', 'orthogonal_scalar_es']
for _i in range(len(LADDER)):  # cumulative: E+N, then each A1 change added in turn (the last rung is 'A1 best')
    CONFIGS[f'L{_i + 1} +{LADDER[_i]}'] = ('fast', Options(compile=True, **{k: True for k in LADDER[:_i + 1]}), BASE)


class NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


class PeakMemory:
    """Polls MPS allocated memory from a thread; returns the peak above the starting level."""

    def __init__(self):
        self.peak, self.running = 0, False

    def __enter__(self):
        torch.mps.synchronize()
        self.base = torch.mps.current_allocated_memory()
        self.peak, self.running = self.base, True
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def poll(self):
        while self.running:
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(0.0005)

    def __exit__(self, *exc):
        torch.mps.synchronize()
        self.running = False
        self.thread.join()


def batch(device):
    rows = torch.load(DATA, weights_only=True)[:8].long().to(device)
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def make_targets(args):
    device = torch.device('mps')
    x, y = batch(device)
    model = dust.make_model(GPTConfig(), device, 42)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    reference = model.float().requires_grad_(True)
    full, _ = cosine_probe.backprop(reference, x, y)
    same, _ = cosine_probe.same_token_gradient(reference, x, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(full={k: v.cpu() for k, v in full.items()}, same={k: v.cpu() for k, v in same.items()},
                    checkpoint=str(args.checkpoint)), args.out)
    print(f'targets -> {args.out}')


def run(args):
    impl, options, settings = CONFIGS[args.config]
    if args.population != 256:
        settings = scaled(settings, writers=args.population / 256, hidden=args.population / 256,
                          hubs=args.population / 256, embedding=args.population / 256,
                          head=args.population / 256, local=args.population / 256)
    if args.scale:
        settings = scaled(settings, **json.loads(args.scale))
    device = torch.device('mps')
    targets = torch.load(args.targets, weights_only=True)
    x, y = batch(device)
    model = dust.make_model(GPTConfig(), device, 42)
    if targets['checkpoint'] not in (None, 'None'):
        model.load_state_dict(torch.load(targets['checkpoint'], map_location=device, weights_only=True))
    if impl == 'dust-compiled' or (options is not None and options.compile):
        dust.compile_model(model)
    make = {
        'dust': lambda seed: dust.DUST(model, settings, seed=seed),
        'dust-compiled': lambda seed: dust.DUST(model, settings, seed=seed),
        'fast': lambda seed: fastdust.FastDUST(model, settings, options, seed=seed),
    }[impl]
    est = make(1000)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    est.step(x, y, NoStep())  # warm-up: compilation
    torch.mps.synchronize()
    warmup = time.perf_counter() - t0
    cells = []
    for seed in args.seeds:  # reuse the warmed-up estimator (new hooks would force recompiles); reseed only
        est.generator.manual_seed(seed + 3)
        drop_caches(est)
        with PeakMemory() as mem:
            t0 = time.perf_counter()
            est.step(x, y, NoStep())
            torch.mps.synchronize()
            seconds = time.perf_counter() - t0
        cos, per_param = {}, {}
        for name, p in model.named_parameters():
            label = cosine_probe.layer_type(name)
            g = p.grad.float().cpu()
            per_param[name] = dict(full=cosine_probe.cosine(g, targets['full'][name]),
                                   same=cosine_probe.cosine(g, targets['same'][name]))
            cos.setdefault(label, []).append(per_param[name])
        cells.append(dict(seed=seed, seconds=seconds, peak_gb=(mem.peak - mem.base) / 1e9, params=per_param, cos={
            k: dict(full=sum(c['full'] for c in v) / len(v), same=sum(c['same'] for c in v) / len(v))
            for k, v in cos.items()}))
        print(f'{args.config} seed {seed}: {seconds:.1f}s, peak +{(mem.peak - mem.base) / 1e9:.2f} GB', flush=True)
    result = dict(config=args.config, impl=impl, options=asdict(options) if options else None,
                  tiers=options.tiers() if options else {}, recipe=asdict(settings), warmup_seconds=warmup,
                  population=args.population, scale=args.scale, targets=str(args.targets), cells=cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')


def table(args):
    rows = [json.loads(Path(p).read_text()) for p in args.results]
    rows.sort(key=lambda r: (list(CONFIGS).index(r['config']) if r['config'] in CONFIGS else 99, r.get('scale') or ''))
    base = next((r for r in rows if r['config'] == args.baseline), rows[0])
    sec = lambda r: sum(c['seconds'] for c in r['cells']) / len(r['cells'])
    mean = lambda r, t, k: sum(c['cos'][t][k] for c in r['cells']) / len(r['cells'])
    short = {'head': 'head', 'attention out': 'attnO', 'MLP out': 'mlpO', 'MLP in': 'mlpI',
             'token embedding': 'emb', 'query': 'q', 'key': 'k', 'value': 'v', 'value gate': 'gate',
             'value embedding': 've', 'residual scalars': 'lam'}
    key = args.target
    print(f'cosine with the {key} target; speedup vs {base["config"]}')
    print(f'{"config":28s} {"s/upd":>6s} {"x":>5s} {"GB":>5s} ' + ' '.join(f'{short[t]:>6s}' for t in TYPES))
    for r in rows:
        label = r['config'] + (f' p{r["population"]}' if r.get('population', 256) != 256 else '') + \
            (f' {r["scale"]}' if r.get('scale') else '')
        gb = sum(c['peak_gb'] for c in r['cells']) / len(r['cells'])
        print(f'{label[:28]:28s} {sec(r):6.1f} {sec(base) / sec(r):5.2f} {gb:5.1f} ' +
              ' '.join(f'{mean(r, t, key):6.3f}' for t in TYPES))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('targets')
    p.add_argument('--checkpoint')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', choices=CONFIGS, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--population', type=int, default=256)
    p.add_argument('--scale', help='JSON dict of recipe stream multipliers, e.g. {"writers": 2}')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('table')
    p.add_argument('results', nargs='+')
    p.add_argument('--baseline', default='reference+N')
    p.add_argument('--target', choices=['full', 'same'], default='full')
    args = parser.parse_args()
    {'targets': make_targets, 'run': run, 'table': table}[args.cmd](args)


if __name__ == '__main__':
    main()
