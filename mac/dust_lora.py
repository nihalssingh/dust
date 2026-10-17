"""Fine-tune a Hugging Face causal LM with LoRA adapters, trained by anchored DUST (no backward pass through
the network), by MeZO-style weight-space ES, or by backprop for comparison.

Anchored DUST for a decoder-only model (see fastdust.py for the tiers; this is the A1 "anchored" design):
  - LoRA adapters on each layer's two residual writers, self_attn.o_proj and mlp.down_proj.
  - One clean forward per update caches every layer's output, its MLP input (x_mid) and the adapters' inputs.
  - A draw jitters one layer's output at every token (noise scale relative to that layer's output RMS) and
    reruns only the layers after it. Each token's jitter is rewarded by its own loss change, with a
    leave-one-out baseline over that layer's draws.
  - down_proj's output error is that layer's output error; o_proj's output error is the same error mapped back
    through the layer's MLP with its exact per-token Jacobian (a local vector-Jacobian product, no cross-token
    terms, no backward pass through the network).
  - LoRA gradients are outer products of those errors with the cached adapter inputs.

    python mac/dust_lora.py --method dust --steps 100 --output ../runs/lora/dust
    python mac/dust_lora.py --method backprop --steps 100 --output ../runs/lora/bp
    python mac/dust_lora.py --method mezo --steps 100 --output ../runs/lora/mezo
"""
import argparse
import json
import math
from pathlib import Path
import threading
import time

import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(__file__).resolve().parents[2] / 'data' / 'shakespeare' / 'input.txt'


class LoRALinear(nn.Module):
    """y = W x + scale * B (A x). Standard init: A random, B zero. frozen_b: B a fixed random orthonormal basis
    and A zero (only A trains), so the bottleneck z = A x can be jittered (`jitter`, [batch, tokens, rank])."""

    def __init__(self, base, rank, alpha, frozen_b=False):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.jitter = None
        if frozen_b:
            q, _ = torch.linalg.qr(torch.randn(base.out_features, rank, dtype=torch.float32))
            self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
            self.B = nn.Parameter(q, requires_grad=False)
        else:
            self.A = nn.Parameter(torch.randn(rank, base.in_features, dtype=torch.float32) / math.sqrt(base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, x):
        z = F.linear(x.float(), self.A)
        if self.jitter is not None:
            z = z + self.jitter
        return self.base(x) + (F.linear(z, self.B) * self.scale).to(x.dtype)


def add_lora(model, rank, alpha, layers=None, frozen_b=False):
    """Adapters on o_proj and down_proj of the top `layers` layers (all layers by default)."""
    adapters = {}
    first = 0 if layers is None else len(model.model.layers) - layers
    for i, layer in enumerate(model.model.layers):
        if i < first:
            continue
        layer.self_attn.o_proj = LoRALinear(layer.self_attn.o_proj, rank, alpha, frozen_b)
        layer.mlp.down_proj = LoRALinear(layer.mlp.down_proj, rank, alpha, frozen_b)
        adapters[f'{i}.o'] = layer.self_attn.o_proj
        adapters[f'{i}.down'] = layer.mlp.down_proj
    return adapters


def token_losses(model, hidden, targets, normed=False, chunk=1024):
    """Per-token cross-entropy from the last layer's output (normed=True: transformers' last_hidden_state,
    which already has the final norm applied), in chunks of tokens (the vocabulary is large)."""
    h = (hidden if normed else model.model.norm(hidden)).flatten(0, 1)
    t = targets.flatten()
    out = torch.empty(t.shape[0], device=hidden.device, dtype=torch.float32)
    for s in range(0, t.shape[0], chunk):
        logits = model.lm_head(h[s:s + chunk]).float()
        out[s:s + chunk] = F.cross_entropy(logits, t[s:s + chunk], reduction='none')
    return out.view(targets.shape)


class Capture:
    """Hooks for one clean forward: layer outputs, MLP inputs, adapter inputs and the decoder-layer kwargs."""

    def __init__(self, model):
        self.model, self.on, self.handles = model, False, []
        self.out, self.mid, self.inp, self.kwargs = {}, {}, {}, None
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_hook(self._out(i)))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self._mid(i)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._inp(f'{i}.o')))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._inp(f'{i}.down')))
        self.handles.append(model.model.layers[0].register_forward_pre_hook(self._kw, with_kwargs=True))

    def _out(self, i):
        def hook(module, args, output):
            if self.on:
                self.out[i] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def _mid(self, i):
        def hook(module, args):
            if self.on:
                self.mid[i] = args[0].detach()
        return hook

    def _inp(self, key):
        def hook(module, args):
            if self.on:
                self.inp[key] = args[0].detach()
        return hook

    def _kw(self, module, args, kwargs):
        if self.on:
            self.kwargs = {k: v for k, v in kwargs.items() if k in ('attention_mask', 'position_embeddings',
                                                                       'position_ids', 'cache_position')}

    def clean(self, x, y):
        self.out, self.mid, self.inp = {}, {}, {}  # drop the previous step's caches before allocating new ones
        self.on = True
        try:
            hidden = self.model.model(input_ids=x, use_cache=False).last_hidden_state
        finally:
            self.on = False
        return token_losses(self.model, hidden, y, normed=True)


def repeat_kwargs(kwargs, n):
    def rep(v):
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] > 1:
            return v.repeat(n, *([1] * (v.dim() - 1)))
        if isinstance(v, tuple):
            return tuple(rep(t) for t in v)
        return v
    return {k: rep(v) for k, v in kwargs.items()}


def last_layer_error(model, cap, y):
    """The last layer's output error exactly: d loss_t / d h_t through the final norm, head and cross-entropy,
    which act on each token separately (a local per-token derivative, no backward through the layers)."""
    with torch.enable_grad():
        h = cap.out[len(model.model.layers) - 1].detach().float().requires_grad_(True)
        losses = token_losses(model, h.to(cap.out[0].dtype), y)
        (error,) = torch.autograd.grad(losses.sum(), h)
    return error


@torch.no_grad()
def dust_gradients(model, adapters, cap, x, y, draws_per_layer, rel_sigma, generator, exact_last=True,
                   control_variate=False):
    """Anchored DUST estimates of each layer's output error: independent draws, two per forward, rewarded per
    token with a leave-one-out baseline over the layer's draws; each draw reruns only the later layers."""
    clean = cap.clean(x, y)
    L = len(model.model.layers)
    B, T = y.shape
    kw = repeat_kwargs(cap.kwargs, 2)
    y2 = y.repeat(2, 1)
    errors = {}
    g_last = last_layer_error(model, cap, y) if (exact_last or control_variate) else None
    if exact_last:
        errors[L - 1] = g_last
    adapted = sorted({int(k.split('.')[0]) for k in adapters})
    for l in [l for l in adapted if not (exact_last and l == L - 1)]:
        h = cap.out[l]
        sigma = rel_sigma * h.float().pow(2).mean().sqrt().item()
        weighted = torch.zeros(B, T, h.shape[-1], device=h.device, dtype=torch.float32)
        noise_sum = torch.zeros_like(weighted)
        change_sum = torch.zeros(B, T, device=h.device, dtype=torch.float32)
        for _ in range(draws_per_layer // 2):
            a = torch.randn(2, B, T, h.shape[-1], device=h.device, generator=generator)
            z = (h.unsqueeze(0) + (sigma * a).to(h.dtype)).flatten(0, 1)
            for j in range(l + 1, L):
                z = model.model.layers[j](z, **kw)
                z = z[0] if isinstance(z, tuple) else z
            change = token_losses(model, z, y2).view(2, B, T) - clean.unsqueeze(0)
            if control_variate:  # the skip path's part of each token's loss change, known exactly from g_last
                change = change - sigma * torch.einsum('btd,nbtd->nbt', g_last, a)
            weighted += torch.einsum('nbt,nbtd->btd', change, a)
            noise_sum += a.sum(0)
            change_sum += change.sum(0)
        K = 2 * (draws_per_layer // 2)  # the draws that actually ran
        errors[l] = (weighted - (change_sum / K).unsqueeze(-1) * noise_sum) / ((K - 1) * sigma)
        if control_variate:
            errors[l] = errors[l] + g_last
    return clean, errors


@torch.no_grad()
def bottleneck_gradients(model, adapters, cap, x, y, draws, rel_sigma, generator, control_variate=True):
    """A2 "bottleneck" DUST for frozen-B adapters: every adapter's rank-r bottleneck z = A x is jittered at every
    token, jointly, and each token's jitter is rewarded by its own loss change (leave-one-out baseline over the
    draws). That estimates dL/dz, so dL/dA = sum_t dL/dz_t x_t^T. Noise lives in r dimensions per adapter
    instead of the layer width. With control_variate, each adapter's skip-path term scale B^T g_last (from the
    exact last-layer error) is subtracted from every token's loss change and added back."""
    clean = cap.clean(x, y)
    L = len(model.model.layers)
    B, T = y.shape
    g_last = last_layer_error(model, cap, y) if control_variate else None
    sig, known = {}, {}
    for key, ad in adapters.items():
        layer = int(key.split('.')[0])
        width = cap.out[layer].float().pow(2).mean().sqrt().item()  # residual-stream RMS where y lands
        # per-coordinate jitter in the residual stream of rel_sigma x its RMS in total, split over all adapters
        sig[key] = rel_sigma * width * math.sqrt(ad.B.shape[0] / ad.B.shape[1]) / ad.scale / math.sqrt(len(adapters))
        if control_variate:
            known[key] = ad.scale * g_last @ ad.B  # [B, T, r]
    r = next(iter(adapters.values())).B.shape[1]
    weighted = {k: torch.zeros(B, T, r, device=y.device) for k in adapters}
    noise_sum = {k: torch.zeros(B, T, r, device=y.device) for k in adapters}
    change_sum = torch.zeros(B, T, device=y.device)
    x2, y2 = x.repeat(2, 1), y.repeat(2, 1)
    try:
        for _ in range(draws // 2):
            noise = {k: torch.randn(2, B, T, r, device=y.device, generator=generator) for k in adapters}
            for k, ad in adapters.items():
                ad.jitter = (sig[k] * noise[k]).flatten(0, 1)
            hidden = model.model(input_ids=x2, use_cache=False).last_hidden_state
            change = token_losses(model, hidden, y2, normed=True).view(2, B, T) - clean.unsqueeze(0)
            for k in known:
                change = change - sig[k] * torch.einsum('btr,nbtr->nbt', known[k], noise[k])
            change_sum += change.sum(0)
            for k in adapters:
                weighted[k] += torch.einsum('nbt,nbtr->btr', change, noise[k])
                noise_sum[k] += noise[k].sum(0)
    finally:
        for ad in adapters.values():
            ad.jitter = None
    K = 2 * (draws // 2)
    grads = {}
    for k, ad in adapters.items():
        dz = (weighted[k] - (change_sum / K).unsqueeze(-1) * noise_sum[k]) / ((K - 1) * sig[k])
        if control_variate:
            dz = dz + known[k]
        xin = cap.inp[k].float().flatten(0, 1)
        grads[k] = (torch.zeros_like(ad.B), dz.flatten(0, 1).T @ xin / y.numel())
    return clean, grads


def mid_errors(model, cap, errors):
    """Error at x_mid (o_proj's output) = e_out + (d MLP(norm(x_mid)) / d x_mid)^T e_out, per token."""
    out = {}
    for l, layer in enumerate(model.model.layers):
        if l not in errors:
            continue
        with torch.enable_grad():
            xm = cap.mid[l].detach().float().requires_grad_(True)
            y = layer.mlp(layer.post_attention_layernorm(xm.to(cap.mid[l].dtype)))
            (back,) = torch.autograd.grad(y.float(), xm, grad_outputs=errors[l])
        out[l] = errors[l] + back
    return out


def lora_gradients(adapters, cap, errors, mids, tokens):
    """dL/dB = s sum_t e_t (A x_t)^T, dL/dA = s sum_t (B^T e_t) x_t^T, for the mean token loss."""
    grads = {}
    for key, ad in adapters.items():
        l, kind = key.split('.')
        e = (errors if kind == 'down' else mids)[int(l)].flatten(0, 1)
        xin = cap.inp[key].float().flatten(0, 1)
        ax = xin @ ad.A.T
        grads[key] = (ad.scale * (e.T @ ax) / tokens, ad.scale * ((e @ ad.B).T @ xin) / tokens)
    return grads


def batches(tokens, seq, batch, generator):
    n = (len(tokens) - 1) // seq
    rows = torch.stack([tokens[i * seq:i * seq + seq + 1] for i in range(n)])
    while True:
        order = torch.randperm(n, generator=generator)
        for s in range(0, n - batch + 1, batch):
            yield rows[order[s:s + batch]]


@torch.no_grad()
def evaluate(model, rows, device, batch=8):
    total, count = 0.0, 0
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch].to(device)
        hidden = model.model(input_ids=r[:, :-1], use_cache=False).last_hidden_state
        loss = token_losses(model, hidden, r[:, 1:], normed=True)
        total += loss.sum().item()
        count += loss.numel()
    return total / count


class PeakMemory:
    def __init__(self):
        self.peak, self.running = 0, False

    def __enter__(self):
        self.peak, self.running = torch.mps.current_allocated_memory(), True
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def poll(self):
        while self.running:
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(0.001)

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=['dust', 'dust-bottleneck', 'backprop', 'mezo'], required=True)
    parser.add_argument('--frozen-b', action='store_true',
                        help='adapters with a fixed random orthonormal B and A from zero (implied by dust-bottleneck)')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16)
    parser.add_argument('--layers', type=int, help='adapt only the top N layers (default: all)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--draws-per-layer', type=int, default=2, help='dust: draws per layer (pairs)')
    parser.add_argument('--rel-sigma', type=float, default=0.05, help='dust: noise scale / layer-output RMS')
    parser.add_argument('--no-exact-last', action='store_true', help='dust: estimate the last layer by draws too')
    parser.add_argument('--control-variate', action='store_true',
                        help="dust: subtract each draw's skip-path term from the exact last-layer error (A1)")
    parser.add_argument('--mezo-eps', type=float, default=1e-3)
    parser.add_argument('--mezo-pairs', type=int, default=16, help='mezo: +/- pairs per step')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum (adam: beta1)')
    parser.add_argument('--checkpointing', action='store_true', help='backprop: recompute each layer in backward')
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--eval-seqs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.method in ('dust', 'dust-bottleneck') and (args.draws_per_layer < 2 or args.draws_per_layer % 2):
        parser.error('--draws-per-layer must be even and at least 2 (two draws per forward, leave-one-out baseline)')

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device('mps')
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = torch.tensor(tok(args.data.read_text())['input_ids'], dtype=torch.long)
    split = int(0.9 * len(ids))
    train, held = ids[:split], ids[split:]
    eval_rows = torch.stack([held[i * args.seq:i * args.seq + args.seq + 1]
                             for i in range(min(args.eval_seqs, (len(held) - 1) // args.seq))])
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    model.requires_grad_(False)
    frozen_b = args.frozen_b or args.method == 'dust-bottleneck'
    adapters = add_lora(model, args.rank, args.alpha, args.layers, frozen_b)
    model.to(device)
    params = [p for ad in adapters.values() for p in ((ad.A,) if frozen_b else (ad.A, ad.B))]
    for p in params:  # only backprop needs autograd on the adapters
        p.requires_grad_(args.method == 'backprop')
    if args.method == 'backprop':
        if args.checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.disable_input_require_grads()  # non-reentrant checkpointing does not need it; skips frozen layers
            model.train()  # transformers only checkpoints in training mode (the model has no dropout)
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr, betas=(args.momentum, 0.999))
    else:
        opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    gen = torch.Generator().manual_seed(args.seed)
    noise_gen = torch.Generator(device).manual_seed(args.seed + 1)
    cap = Capture(model) if args.method in ('dust', 'dust-bottleneck') else None
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    history = [dict(step=0, eval_loss=evaluate(model, eval_rows, device), seconds=0.0)]
    print(json.dumps(history[0]), flush=True)
    seconds, peak = 0.0, 0
    stream = batches(train, args.seq, args.batch, gen)
    for step in range(1, args.steps + 1):
        rows = next(stream).to(device)
        x, y = rows[:, :-1], rows[:, 1:]
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with PeakMemory() as mem:
            opt.zero_grad(set_to_none=True)
            if args.method == 'backprop':
                hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
                loss = token_losses(model, hidden, y, normed=True).mean()
                loss.backward()
                train_loss = loss.item()
            elif args.method == 'dust':
                clean, errors = dust_gradients(model, adapters, cap, x, y, args.draws_per_layer, args.rel_sigma,
                                               noise_gen, exact_last=not args.no_exact_last,
                                               control_variate=args.control_variate)
                mids = mid_errors(model, cap, errors)
                for key, (gB, gA) in lora_gradients(adapters, cap, errors, mids, y.numel()).items():
                    adapters[key].B.grad, adapters[key].A.grad = gB, gA
                train_loss = clean.mean().item()
                del clean, errors, mids  # keep only what the next step needs
            elif args.method == 'dust-bottleneck':
                clean, grads = bottleneck_gradients(model, adapters, cap, x, y, args.draws_per_layer, args.rel_sigma,
                                                    noise_gen, control_variate=args.control_variate)
                for key, (_, gA) in grads.items():
                    adapters[key].A.grad = gA
                train_loss = clean.mean().item()
                del clean, grads
            else:  # MeZO: projected two-point estimates along random directions of all LoRA parameters
                with torch.no_grad():
                    grads = [torch.zeros_like(p) for p in params]
                    losses = []
                    for _ in range(args.mezo_pairs):
                        zs = [torch.randn(p.shape, device=device, generator=noise_gen) for p in params]
                        for p, z in zip(params, zs):
                            p.add_(args.mezo_eps * z)
                        hp = model.model(input_ids=x, use_cache=False).last_hidden_state
                        lp = token_losses(model, hp, y, normed=True).mean()
                        for p, z in zip(params, zs):
                            p.sub_(2 * args.mezo_eps * z)
                        hm = model.model(input_ids=x, use_cache=False).last_hidden_state
                        lm = token_losses(model, hm, y, normed=True).mean()
                        for p, z in zip(params, zs):
                            p.add_(args.mezo_eps * z)
                        proj = (lp - lm) / (2 * args.mezo_eps)
                        for g, z in zip(grads, zs):
                            g.add_(proj * z / args.mezo_pairs)
                        losses.append(((lp + lm) / 2).item())
                    for p, g in zip(params, grads):
                        p.grad = g
                    train_loss = sum(losses) / len(losses)
            opt.step()
            torch.mps.synchronize()
        seconds += time.perf_counter() - t0
        peak = max(peak, mem.peak)
        record = dict(step=step, train_loss=train_loss, seconds=seconds)
        if step % args.eval_every == 0 or step == args.steps:
            record['eval_loss'] = evaluate(model, eval_rows, device)
        history.append(record)
        print(json.dumps(record), flush=True)
    result = dict(method=args.method, history=history, peak_memory_gb=peak / 1e9,
                  seconds_per_step=seconds / args.steps, final_eval=history[-1].get('eval_loss'))
    (args.output / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    print(f"{args.method}: eval {history[0]['eval_loss']:.4f} -> {result['final_eval']:.4f}, "
          f"{result['seconds_per_step']:.2f}s/step, peak {result['peak_memory_gb']:.1f} GB", flush=True)


if __name__ == '__main__':
    main()
