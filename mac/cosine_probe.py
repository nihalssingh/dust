"""Gradient-fidelity probe: how close is Dust's estimate to backprop, and to which gradient?

On one batch, Dust's per-parameter estimates at several populations are compared with
  full:       the backprop gradient of the mean token loss;
  same-token: for every jittered site y, only d loss_t / d y_t at each token t. This drops what token t's
              activation does to later tokens' losses through attention.
Dust rewards each token's jitter with that token's own loss change (credit decay 0 for every direct-loss
site), so its direct-loss estimates should converge to the same-token gradient, not the full one. The gap
between the two is the cross-token credit the estimator leaves out by design.

The same-token gradient comes from one backward pass through an attention whose forward is unchanged but
whose keys and values receive gradient only from their own token's query (diagonal_attention). For the
attention internals of block l, only blocks after l are made diagonal, which matches Dust's target for
queries exactly and for keys/values up to their credit decay (0.98 per token, here 1).

    python mac/cosine_probe.py --self-test
    python mac/cosine_probe.py --populations 64 256 1024 4096 --out ../runs/cosine_init.json
    python mac/cosine_probe.py --checkpoint ../runs/bp_sgd_10M_s42/best.pt --out ../runs/cosine_bp10M.json
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dust  # noqa: E402
import model as model_lib  # noqa: E402
from model import GPTConfig  # noqa: E402

ORIGINAL_ATTENTION = model_lib.attention
LAYER_TYPES = {
    'wte': 'token embedding', 'attn.c_q': 'query', 'attn.c_k': 'key', 'attn.c_v': 'value',
    'attn.ve_gate': 'value gate', 'value_embeds': 'value embedding', 'attn.c_proj': 'attention out',
    'mlp.c_fc': 'MLP in', 'mlp.c_proj': 'MLP out', 'lm_head': 'head', 'lambdas': 'residual scalars',
}
DIRECT = ('token embedding', 'attention out', 'MLP in', 'MLP out', 'head')


def layer_type(name):
    for key, label in LAYER_TYPES.items():
        if key in name:
            return label
    raise KeyError(name)


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-300)).item()


def diagonal_attention(q, k, v, window):
    """Same output as model.attention, but keys and values get gradient only from their own token's query.

    out_s = sum_r p_sr v_r. The forward uses detached keys and values (so queries get their full gradient),
    plus a zero-valued surrogate whose derivative restores exactly the r = s terms:
        d out_s / d v_s = p_ss,   d out_s / d z_ss = p_ss (v_s - out_s),   z_ss = <q_s, k_s> / sqrt(d).
    """
    out = ORIGINAL_ATTENTION(q, k.detach(), v.detach(), window)
    B, T, H, D = q.shape
    scale = D ** -0.5
    dt = torch.promote_types(q.dtype, torch.float32)
    with torch.no_grad():  # log partition function of every query, in chunks of queries
        qh, kh = q.detach().transpose(1, 2).to(dt), k.detach().transpose(1, 2).to(dt)
        lse = torch.empty(B, H, T, device=q.device, dtype=dt)
        for start in range(0, T, 256):
            stop = min(T, start + 256)
            logits = qh[:, :, start:stop] @ kh.transpose(-1, -2) * scale
            rows = torch.arange(start, stop, device=q.device)[:, None]
            cols = torch.arange(T, device=q.device)[None, :]
            logits = logits.masked_fill(~((rows - cols >= 0) & (rows - cols <= window)), float('-inf'))
            lse[:, :, start:stop] = torch.logsumexp(logits, dim=-1)
        lse = lse.transpose(1, 2).unsqueeze(-1)  # [B, T, H, 1]
    z_live = (q.detach().to(dt) * k.to(dt)).sum(-1, keepdim=True) * scale
    p_live = torch.exp(z_live - lse)
    p_dead = p_live.detach()
    o, v_dead = out.detach().to(dt), v.detach().to(dt)
    live = (o - p_dead * v_dead + p_live * v.to(dt)) / (1 - p_dead + p_live)
    return out + (live - live.detach()).to(out.dtype)


class AttentionSwitch:
    """Replaces model.attention: blocks >= first_diagonal use diagonal_attention, earlier ones the original."""

    def __init__(self, first_diagonal):
        self.first_diagonal, self.calls = first_diagonal, 0

    def __call__(self, q, k, v, window):
        layer, self.calls = self.calls, self.calls + 1
        fn = diagonal_attention if layer >= self.first_diagonal else ORIGINAL_ATTENTION
        return fn(q, k, v, window)


def activation_sites(model):
    """Modules whose outputs Dust jitters with token-loss rewards, plus the attention outputs o@l."""
    sites = {'transformer.wte': model.transformer.wte}
    for i, block in enumerate(model.transformer.h):
        for name in ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'):
            sites[f'transformer.h.{i}.{name}'] = block.get_submodule(name)
    return sites


def backprop(model, x, y, first_diagonal=None, keep_activations=False):
    """Gradient of the mean token loss; with first_diagonal, cross-token paths in blocks >= it are cut."""
    model.zero_grad(set_to_none=True)
    grads_at, hooks = {}, []
    if keep_activations:
        def output_hook(name):
            def hook(module, args, output):
                output.retain_grad()
                grads_at[name] = output
            return hook

        def input_hook(name):
            def hook(module, args):
                inp = args[0]
                inp.retain_grad()
                grads_at[name] = inp
            return hook
        for name, module in activation_sites(model).items():
            hooks.append(module.register_forward_hook(output_hook(name)))
        for i, block in enumerate(model.transformer.h):
            hooks.append(block.attn.c_proj.register_forward_pre_hook(input_hook(f'o@{i}')))
    if first_diagonal is not None:
        model_lib.attention = AttentionSwitch(first_diagonal)
    try:
        model(x, y).backward()
    finally:
        model_lib.attention = ORIGINAL_ATTENTION
        for h in hooks:
            h.remove()
    params = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()}
    acts = {n: t.grad.detach().float().clone() for n, t in grads_at.items()}
    return params, acts


def same_token_gradient(model, x, y):
    """Direct-loss sites from one all-diagonal pass; each block's attention internals from a pass that is
    diagonal only after that block."""
    params, acts = backprop(model, x, y, first_diagonal=0, keep_activations=True)
    for layer in range(model.config.n_layer):
        internals, _ = backprop(model, x, y, first_diagonal=layer + 1)
        for name, g in internals.items():
            if name.startswith(f'transformer.h.{layer}.attn.') and 'c_proj' not in name:
                params[name] = g
            elif name == f'value_embeds.{layer}.weight':
                params[name] = g
    return params, acts


def scaled_recipe(population):
    """The 1M SGD recipe scaled to a population, two draws per forward (nrep=2) from 256 up so memory stays
    bounded; 64 uses the paper's 64-draw allocation (Appendix E) with one draw per forward."""
    base = dust.recipe('1M', 256)
    if population == 64:
        return replace(
            base, writers=[8, 7, 4, 3, 1, 1, 1, 1], hidden=[8, 7, 4, 3, 2, 1, 1, 1], hubs=[1] * 8,
            embedding=3, head=4 * 64, local=dict(q=1, v=5, gate=2, k=3, ve=2), nrep=1,
        )
    f = population // 256
    return replace(
        base, writers=[c * f for c in base.writers], hidden=[c * f for c in base.hidden],
        hubs=[c * f for c in base.hubs], embedding=base.embedding * f, head=4 * population,
        local={k: c * f for k, c in base.local.items()},
    )


class NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def dust_estimate(model, settings, x, y, seed):
    """One DUST update's estimates: parameter gradients and the per-token errors of the direct-loss sites."""
    estimator = dust.DUST(model, settings, seed=seed)
    errors, direct = {}, estimator.direct_errors

    def recording(sites, draws, sigma):
        out = direct(sites, draws, sigma)
        errors.update({s: e.detach().float().clone() for s, e in out.items()})
        return out
    estimator.direct_errors = recording
    try:
        estimator.step(x, y, NoStep())
    finally:
        estimator.close()
    params = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()}
    return params, errors


def per_token_cosine(estimate, target):
    """Mean over tokens of the cosine between the estimated and target error at one token."""
    e, t = estimate.flatten(0, 1).double(), target.flatten(0, 1).double()
    keep = t.norm(dim=-1) > 0
    c = (e * t).sum(-1) / (e.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-300)
    return c[keep].mean().item()


def self_test():
    """The same-token gradient from diagonal_attention matches brute force (one backward per token)."""
    torch.manual_seed(0)
    config = GPTConfig(sequence_len=24, vocab_size=4096, n_layer=4, n_head=2, n_kv_head=2, n_embd=32)
    config.window_pattern = 'SL'
    model = dust.make_model(config, torch.device('cpu'), seed=1).double().requires_grad_(True)
    x = torch.randint(0, 4096, (2, 24))
    y = torch.randint(0, 4096, (2, 24))
    site = 'transformer.h.1.mlp.c_fc'
    params, acts = backprop(model, x, y, first_diagonal=0, keep_activations=True)
    trick = acts[site]
    # Brute force: d loss_t / d y_t for each token t, from one backward of loss_t alone.
    module = model.get_submodule(site)
    brute = torch.zeros_like(trick)

    def keep_output(store):
        def hook(module, args, output):
            output.retain_grad()
            store['y'] = output
        return hook

    def keep_input(store):
        def hook(module, args):
            args[0].retain_grad()
            store['o'] = args[0]
        return hook
    for t in range(24):
        store = {}
        hook = module.register_forward_hook(keep_output(store))
        model.zero_grad(set_to_none=True)
        per_token = model(x, y, loss_reduction='none').view(2, 24)
        (per_token[:, t].sum() / y.numel()).backward()
        hook.remove()
        brute[:, t] = store['y'].grad[:, t]
    err = ((trick - brute).norm() / brute.norm()).item()
    full, full_acts = backprop(model, x, y, keep_activations=True)
    gap = cosine(full_acts[site], brute)
    print(f'same-token activation gradient, trick vs brute force: relative error {err:.2e}')
    print(f'  (full vs same-token cosine at {site}: {gap:.4f}, so the two targets do differ)')
    # Attention internals of block 1: gradient of sum_s <o_s, J_s> where J_s = d loss_s / d o_s at token s.
    o_store = {}
    pre = model.transformer.h[1].attn.c_proj.register_forward_pre_hook(keep_input(o_store))
    J = None
    for t in range(24):
        model.zero_grad(set_to_none=True)
        per_token = model(x, y, loss_reduction='none').view(2, 24)
        (per_token[:, t].sum() / y.numel()).backward()
        row = o_store['o'].grad[:, t].clone()
        J = torch.zeros_like(o_store['o']) if J is None else J
        J[:, t] = row
    pre.remove()
    store = {}
    pre = model.transformer.h[1].attn.c_proj.register_forward_pre_hook(lambda m, a: store.update(o=a[0]))
    model.zero_grad(set_to_none=True)
    model(x, y)
    (store['o'] * J).sum().backward()
    pre.remove()
    brute_k = model.transformer.h[1].attn.c_k.weight.grad.clone()
    internals, _ = backprop(model, x, y, first_diagonal=2)
    err_k = ((internals['transformer.h.1.attn.c_k.weight'] - brute_k).norm() / brute_k.norm()).item()
    print(f'block-1 key gradient against same-token attention-output errors, trick vs brute force: '
          f'relative error {err_k:.2e}')
    print(f"  (full vs same-token cosine for that key gradient: "
          f"{cosine(full['transformer.h.1.attn.c_k.weight'], brute_k):.4f})")
    # model.forward computes the loss from float32 logits, so float32 round-off is the floor here.
    ok = err < 1e-6 and err_k < 1e-6
    print('SELF-TEST', 'PASSED' if ok else 'FAILED')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='state dict to probe (default: initialization)')
    parser.add_argument('--populations', type=int, nargs='+', default=[64, 256, 1024, 4096])
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'val.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=False)
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
    rows = torch.load(args.data, weights_only=True)[:8].long().to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    model = dust.make_model(GPTConfig(), device, args.seed)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    reference = copy.deepcopy(model).float().requires_grad_(True)  # float32 backprop, no autocast

    t0 = time.monotonic()
    full, full_acts = backprop(reference, x, y, keep_activations=True)
    same, same_acts = same_token_gradient(reference, x, y)
    del reference
    print(f'backprop targets in {time.monotonic() - t0:.0f}s', flush=True)
    result = dict(
        checkpoint=str(args.checkpoint) if args.checkpoint else 'init', seed=args.seed,
        batch=f'{args.data} rows 0-7', populations={},
        target_agreement=dict(
            params={n: cosine(full[n], same[n]) for n in full},
            sites={s: cosine(full_acts[s], same_acts[s]) for s in full_acts},
        ),
    )
    if not args.no_compile:
        dust.compile_model(model)
    for population in args.populations:
        settings = scaled_recipe(population)
        t0 = time.monotonic()
        est, errors = dust_estimate(model, settings, x, y, args.seed)
        cell = dict(
            recipe=asdict(settings), seconds=time.monotonic() - t0,
            params={n: dict(full=cosine(est[n], full[n]), same=cosine(est[n], same[n])) for n in est},
            sites={s: dict(full=cosine(e, full_acts[s]), same=cosine(e, same_acts[s]),
                           per_token_same=per_token_cosine(e, same_acts[s]),
                           per_token_full=per_token_cosine(e, full_acts[s]))
                   for s, e in errors.items() if s in full_acts},
        )
        result['populations'][str(population)] = cell
        print(f'population {population}: {cell["seconds"]:.0f}s', flush=True)
        summary = {}
        for n, c in cell['params'].items():
            summary.setdefault(layer_type(n), []).append(c)
        for label in sorted(summary, key=lambda k: list(LAYER_TYPES.values()).index(k)):
            vals = summary[label]
            f = sum(v['full'] for v in vals) / len(vals)
            s = sum(v['same'] for v in vals) / len(vals)
            print(f'   {label:18s} cos(full) {f:7.4f}   cos(same-token) {s:7.4f}', flush=True)
        if args.out:
            args.out.write_text(json.dumps(result, indent=1) + '\n')
    agreement = {}
    for n, c in result['target_agreement']['params'].items():
        agreement.setdefault(layer_type(n), []).append(c)
    print('cos(full, same-token) per layer type, i.e. the ceiling if Dust converged to the same-token gradient:')
    for label, vals in agreement.items():
        print(f'   {label:18s} {sum(vals) / len(vals):7.4f}')
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
