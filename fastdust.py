"""Faster DUST: the estimator of dust.py with optional, clearly tiered changes.

Each option has a tier:
  E   exact: the same estimator; with the same noise, the same numbers up to float rounding
  N   numerical: the same algorithm with different arithmetic (e.g. torch.compile fusion)

With the default Options (E only), FastDUST draws the same noise in the same order as DUST and gives the same
estimates. Draws run through an explicit functional forward instead of module hooks, so a draw that jitters
block l starts at the jittered module from cached clean activations instead of rerunning the whole network.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

import dust
from dust import DUST, average_ranks
from model import BF16_DEVICES, GPTConfig, apply_rotary_emb, attention, has_ve, norm


@dataclass
class Options:
    # E: rerun a draw only from the jittered module onward (cached clean activations upstream)
    truncate: bool = True
    # N: torch.compile the functional forward pieces, the head-draw and the local-draw kernels
    compile: bool = False

    def tiers(self):
        tier = {'truncate': 'E', 'compile': 'N'}
        return {f.name: tier[f.name] for f in fields(self) if getattr(self, f.name)}


def _attn_core(x_mix, ve, wq, wk, wv, wg, cos, sin, window: int, n_head: int):
    """The attention of one block up to (not including) its output projection; same ops as the model."""
    B, T, _ = x_mix.shape
    xn = norm(x_mix)
    q = F.linear(xn, wq).view(B, T, n_head, -1)
    k = F.linear(xn, wk).view(B, T, n_head, -1)
    v = F.linear(xn, wv).view(B, T, n_head, -1)
    if ve is not None:
        gate = 2 * torch.sigmoid(F.linear(xn[..., :wg.shape[1]], wg))
        v = v + gate.unsqueeze(-1) * ve.view(B, T, n_head, -1)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    return attention(q, k, v, window).contiguous().view(B, T, -1)


def _block(x, x0, lam, lam0, ve, wq, wk, wv, wg, wproj, wfc, wmproj, cos, sin, window: int, n_head: int):
    """One unjittered block, residual mix included."""
    x = lam * x + lam0 * x0
    x = x + F.linear(_attn_core(x, ve, wq, wk, wv, wg, cos, sin, window, n_head), wproj)
    return x + F.linear(F.relu(F.linear(norm(x), wfc)).square(), wmproj)


def _tail_from_o(x_mix, o, wproj, wfc, wmproj, jo, jy, jm):
    """A block from its attention output on: optional jitter on o, on the projection, on the MLP output."""
    if jo is not None:
        o = o + jo.to(o.dtype)
    y = F.linear(o, wproj)
    if jy is not None:
        y = y + jy.to(y.dtype)
    x = x_mix + y
    m = F.linear(F.relu(F.linear(norm(x), wfc)).square(), wmproj)
    if jm is not None:
        m = m + jm.to(m.dtype)
    return x + m


def _tail_from_hidden(x_mid, h, wmproj, jh, jm):
    """A block from its MLP hidden layer on, with jitter on the hidden layer and/or the MLP output."""
    if jh is not None:
        h = h + jh.to(h.dtype)
    m = F.linear(F.relu(h).square(), wmproj)
    if jm is not None:
        m = m + jm.to(m.dtype)
    return x_mid + m


def _head_loss(x, w, y, vocab: int):
    logits = F.linear(norm(x), w)[..., :vocab].float()
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction='none')


def _head_chunk(raw, a, sigma: float, rest, target, lo: int, slab: int, clean_target, clean_loss, valid,
                n: int, b: int, t: int):
    """One chunk of head draws on one slab (same math as DUST.head_error); returns the summed estimate."""
    changed = raw[..., lo:lo + slab].float().repeat(n, 1, 1) + sigma * a.float()
    changed = 15 * torch.tanh(changed / 15)
    partition = rest.repeat(n, 1) + changed.exp().sum(-1)
    in_slab = (target >= lo) & (target < lo + slab)
    target_logit = torch.where(
        in_slab, changed.gather(-1, (target - lo).clamp(0, slab - 1).unsqueeze(-1)).squeeze(-1),
        clean_target.repeat(n, 1),
    )
    loss = (partition.log() - target_logit).view(n, b, t)
    reward = torch.where(valid.unsqueeze(0), clean_loss.unsqueeze(0) - loss, 0)
    if b > 1:
        reward = reward - reward.mean(1, keepdim=True)
    return reward, changed


def _local_chunk(kind: str, a, raw, qp, kp, values, baseline, credit, target, z, ve, sigma: float, window: int,
                 cos, sin, n: int, b: int, t: int, h: int, d: int, dtype: torch.dtype):
    """One chunk of attention-internal draws for one block (same math as DUST.local_errors)."""
    def processed(x):
        return norm(apply_rotary_emb(x, cos, sin)).to(dtype)

    def repeat(x):
        return x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)

    divisor = sigma
    if kind in ('q', 'k'):
        perturbed = processed((raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d))
        qa, ka = (perturbed, repeat(kp)) if kind == 'q' else (repeat(qp), perturbed)
        delta = attention(qa, ka, repeat(values.to(dtype)), window).view(n, b, t, h, d) - baseline
    else:
        if kind == 'gate':
            delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
            delta_value = delta_gate.unsqueeze(-1) * ve
            divisor = 2 * sigma
        else:
            delta_value = sigma * a.view(n, b, t, h, d)
            if kind == 've':
                delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
        delta = attention(
            repeat(qp), repeat(kp), delta_value.reshape(n * b, t, h, d).to(dtype), window,
        ).view(n, b, t, h, d)
    score = torch.einsum('bthd,nbthd->nbth', target.view(b, t, h, d).to(dtype), delta)
    if kind != 'q':
        score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
    estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
    return estimate.reshape(b, t, -1) / divisor


class FastDUST(DUST):
    """DUST with a functional, truncatable forward for the draws. Hooks are not used."""

    def __init__(self, model, settings, options=None, seed=42, rank=0, world=1):
        super().__init__(model, settings, seed, rank, world)
        self.close()  # the functional forward replaces the hooks
        self.options = options or Options()
        cfg = model.config
        self.n_head = cfg.n_head
        self.blocks = []
        for i, block in enumerate(model.transformer.h):
            attn, mlp = block.attn, block.mlp
            self.blocks.append(dict(
                wq=attn.c_q.weight, wk=attn.c_k.weight, wv=attn.c_v.weight,
                wg=attn.ve_gate.weight if attn.ve_gate is not None else None,
                wproj=attn.c_proj.weight, wfc=mlp.c_fc.weight, wmproj=mlp.c_proj.weight,
                window=model.window_sizes[i][0],
            ))
        compile_fn = torch.compile if self.options.compile else (lambda f, **kw: f)
        self._block_fn = compile_fn(_block)
        self._attn_fn = compile_fn(_attn_core)
        self._tail_o_fn = compile_fn(_tail_from_o)
        self._tail_h_fn = compile_fn(_tail_from_hidden)
        self._head_fn = compile_fn(_head_loss)
        self._head_chunk_fn = compile_fn(_head_chunk)
        self._local_fn = compile_fn(_local_chunk)

    # ---- noise -------------------------------------------------------------------------------------------
    def chunk_noise(self, n, *shape):
        """Noise for one chunk of n draws, shape [n * shape[0], ...]."""
        a = self.noise(n * shape[0], *shape[1:])
        return a

    # ---- clean pass ----------------------------------------------------------------------------------------
    def capture(self, x, y):
        """The clean forward, caching what the hooks of DUST cache plus the residual stream at each block."""
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        m, cfg = self.model, self.model.config
        T = self.length
        cos, sin = m.cos[:, :T], m.sin[:, :T]
        self.cos_sin = (cos, sin)
        inputs, outputs = self.inputs, self.outputs
        self.mix, self.mid, self.o, self.ve = [], [], [], []
        with self.autocast():
            e = F.embedding(x, m.transformer.wte.weight)
            inputs['transformer.wte'], outputs['transformer.wte'] = x, e
            h = norm(e)
            self.x0 = x0 = h
            for i, blk in enumerate(self.blocks):
                h = m.resid_lambdas[i] * h + m.x0_lambdas[i] * x0
                self.mix.append(h)
                ve = None
                if str(i) in m.value_embeds:
                    ve = F.embedding(x, m.value_embeds[str(i)].weight)
                    inputs[f'value_embeds.{i}'], outputs[f'value_embeds.{i}'] = x, ve
                self.ve.append(ve)
                prefix = f'transformer.h.{i}.'
                xn = norm(h)
                q_raw, k_raw, v_raw = (F.linear(xn, blk[w]) for w in ('wq', 'wk', 'wv'))
                for name, out in (('c_q', q_raw), ('c_k', k_raw), ('c_v', v_raw)):
                    inputs[prefix + 'attn.' + name] = xn
                    outputs[prefix + 'attn.' + name] = out
                B = self.batch
                q, k, v = (t_.view(B, T, self.n_head, -1) for t_ in (q_raw, k_raw, v_raw))
                if ve is not None:
                    gate_in = xn[..., :blk['wg'].shape[1]]
                    z = F.linear(gate_in, blk['wg'])
                    inputs[prefix + 'attn.ve_gate'], outputs[prefix + 'attn.ve_gate'] = gate_in, z
                    v = v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve.view(B, T, self.n_head, -1)
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                q, k = norm(q), norm(k)
                o = attention(q, k, v, blk['window']).contiguous().view(B, T, -1)
                self.o.append(o)
                yp = F.linear(o, blk['wproj'])
                inputs[prefix + 'attn.c_proj'], outputs[prefix + 'attn.c_proj'] = o, yp
                h = h + yp
                self.mid.append(h)
                hn = norm(h)
                hid = F.linear(hn, blk['wfc'])
                inputs[prefix + 'mlp.c_fc'], outputs[prefix + 'mlp.c_fc'] = hn, hid
                act = F.relu(hid).square()
                ym = F.linear(act, blk['wmproj'])
                inputs[prefix + 'mlp.c_proj'], outputs[prefix + 'mlp.c_proj'] = act, ym
                h = h + ym
            xn = norm(h)
            raw = F.linear(xn, m.lm_head.weight)
            inputs['lm_head'], outputs['lm_head'] = xn, raw
            logits = 15 * torch.tanh(raw[..., :cfg.vocab_size].float() / 15)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction='none')
        self.clean_loss = loss.view_as(y).float()

    # ---- perturbed forward -----------------------------------------------------------------------------------
    def _rep(self, t, n):
        return None if t is None else t.repeat(n, *([1] * (t.dim() - 1)))

    def _start(self, sites):
        """Where a draw jittering these sites must start: ('embed', 0), ('o', l), ('proj', l) or ('hidden', l)."""
        if 'transformer.wte' in sites:
            return 'embed', 0
        layers = set()
        for s in sites:
            layers.add(int(s[2:]) if s.startswith('o@') else int(s.split('.')[2]))
        if len(layers) != 1:
            raise ValueError(f'one draw jitters one block: {sites}')
        layer = layers.pop()
        if any(s.startswith('o@') for s in sites):
            return 'o', layer
        if any(s.endswith('attn.c_proj') for s in sites):
            return 'proj', layer
        return 'hidden', layer

    def _forward_from(self, kind, layer, jitter, n, reps):
        """Per-token losses of n repeated batches with jitter at one block's modules (or the embedding).

        Truncated, the jittered block starts from its cached clean input (and cached attention output or
        hidden layer); otherwise everything upstream is recomputed, as DUST does.
        """
        m = self.model
        cos, sin = self.cos_sin
        L = len(self.blocks)

        def block(j, x, x0):
            b = self.blocks[j]
            return self._block_fn(
                x, x0, m.resid_lambdas[j], m.x0_lambdas[j], reps['ve'][j], b['wq'], b['wk'], b['wv'], b['wg'],
                b['wproj'], b['wfc'], b['wmproj'], cos, sin, b['window'], self.n_head,
            )

        if kind == 'embed' or not self.options.truncate:
            e = reps['e']
            if 'transformer.wte' in jitter:
                e = e + jitter['transformer.wte'].to(e.dtype)
            x = x0 = norm(e)
            for j in range(L if kind == 'embed' else layer):
                x = block(j, x, x0)
            if kind == 'embed':
                return self._head_fn(x, m.lm_head.weight, reps['y'], m.config.vocab_size)
            b = self.blocks[layer]
            x_mix = m.resid_lambdas[layer] * x + m.x0_lambdas[layer] * x0
            o = self._attn_fn(x_mix, reps['ve'][layer], b['wq'], b['wk'], b['wv'], b['wg'], cos, sin,
                              b['window'], self.n_head)
            if kind == 'hidden':
                x_mid = x_mix + F.linear(o, b['wproj'])
                hidden = F.linear(norm(x_mid), b['wfc'])
        else:
            x0, x_mix, o = reps['x0'], reps.get('mix'), reps.get('o')
            x_mid, hidden = reps.get('mid'), reps.get('hidden')
        b, prefix = self.blocks[layer], f'transformer.h.{layer}.'
        if kind in ('o', 'proj'):
            x = self._tail_o_fn(x_mix, o, b['wproj'], b['wfc'], b['wmproj'], jitter.get(f'o@{layer}'),
                                jitter.get(prefix + 'attn.c_proj'), jitter.get(prefix + 'mlp.c_proj'))
        else:
            x = self._tail_h_fn(x_mid, hidden, b['wmproj'], jitter.get(prefix + 'mlp.c_fc'),
                                jitter.get(prefix + 'mlp.c_proj'))
        for j in range(layer + 1, L):
            x = block(j, x, x0)
        return self._head_fn(x, m.lm_head.weight, reps['y'], m.config.vocab_size)

    def _reps(self, kind, layer, n):
        """The clean activations a draw starting at (kind, layer) needs, repeated n times."""
        truncated = self.options.truncate and kind != 'embed'
        reps = dict(y=self._rep(self.y, n),
                    ve=[self._rep(v, n) if (j >= layer or not truncated) else None for j, v in enumerate(self.ve)])
        if not truncated:
            reps['e'] = self._rep(self.outputs['transformer.wte'], n)
            return reps
        reps['x0'] = self._rep(self.x0, n)
        if kind in ('o', 'proj'):
            reps['mix'], reps['o'] = self._rep(self.mix[layer], n), self._rep(self.o[layer], n)
        else:
            reps['mid'] = self._rep(self.mid[layer], n)
            reps['hidden'] = self._rep(self.outputs[f'transformer.h.{layer}.mlp.c_fc'], n)
        return reps

    def direct_errors(self, sites, draws, sigma):
        """As DUST.direct_errors (one-sided loss drops centered over each chunk), via the functional forward."""
        n = self.settings.nrep
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}
        kind, layer = self._start(sites)
        reps = self._reps(kind, layer, n)
        for _ in range(draws // n):
            noise = {s: self.chunk_noise(n, self.batch, self.length, width(s)) for s in sites}
            jitter = {s: sigma * a for s, a in noise.items()}
            with self.autocast():
                perturbed_loss = self._forward_from(kind, layer, jitter, n, reps).view(n, self.batch, self.length)
            reward = self.clean_loss.unsqueeze(0) - perturbed_loss.float()
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            for site, a in noise.items():
                errors[site] -= torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype), a.view(n, self.batch, self.length, -1),
                ).float() / sigma
        return {s: error / draws for s, error in errors.items()}  # the same rounding as DUST.direct_errors

    # ---- head ---------------------------------------------------------------------------------------------------
    def head_error(self, draws, sigma=0.05):
        if not self.options.compile:
            return super().head_error(draws, sigma)
        n, b, t = self.settings.nrep, self.batch, self.length
        raw = self.outputs['lm_head']
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        logits = 15 * torch.tanh(raw[..., :vocab].float() / 15)
        slab_sums = logits.exp().view(b, t, slabs, slab).sum(-1)
        total = slab_sums.sum(-1)
        target = self.y.clamp(min=0)
        clean_target = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        target = target.repeat(n, 1)
        valid = self.y != -1
        error = torch.zeros_like(raw, dtype=torch.float32)
        for chunk in range(draws // n):
            block = chunk % slabs
            lo = block * slab
            a = self.noise(n * b, t, slab)
            reward, _ = self._head_chunk_fn(
                raw, a, sigma, total - slab_sums[..., block], target, lo, slab, clean_target,
                self.clean_loss, valid, n, b, t,
            )
            error[..., lo:lo + slab] -= torch.einsum(
                'nbt,nbtd->btd', (reward / sigma).to(self.dtype), a.view(n, b, t, slab),
            ).float()
        return error / (draws // slabs)

    def local_errors(self, kind, draws, targets, sigma=0.05):
        """DUST.local_errors; compiled per chunk when Options.compile (same noise order, fused arithmetic)."""
        if not self.options.compile:
            return super().local_errors(kind, draws, targets, sigma)
        cfg = self.model.config
        n, b, t, h, d = self.settings.nrep, self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
                 for i in layers}
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()
        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp = norm(apply_rotary_emb(q, cos, sin)).to(self.dtype)
            kp = norm(apply_rotary_emb(k, cos, sin)).to(self.dtype)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]
            credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)
            raw = q if kind == 'q' else k
            clean[i] = raw, qp, kp, values, baseline, credit, z, ve, window
        for _ in range(draws // n):
            for i in layers:
                a = self.noise(n, b, t, width, dtype=torch.float32)
                raw, qp, kp, values, baseline, credit, z, ve, window = clean[i]
                errors[i] += self._local_fn(kind, a, raw, qp, kp, values, baseline, credit, targets[i], z, ve,
                                            sigma, window, cos, sin, n, b, t, h, d, self.dtype)
        return {sites[i]: errors[i] / draws for i in layers}

    def reassemble(self, site, error):
        """DUST.reassemble with the cached input cast to float32 here (it is cached in its native dtype)."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            return super().reassemble(site, error)
        return torch.einsum('btd,bti->di', error, x.float())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', choices=dust.DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--population', type=int, choices=dust.DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-interval', type=int, help='Override the evaluation interval.')
    parser.add_argument('--eval-sequences', type=int, default=544, help='Validation sequences per evaluation.')
    parser.add_argument('--scale-draws', type=json.loads, default=None,
                        help='JSON dict multiplying recipe streams, e.g. {"writers": 2}.')
    parser.add_argument('--writers', type=json.loads, default=None,
                        help='JSON list of writer draws per block, overriding the recipe.')
    for f in fields(Options):
        default = f.default
        parser.add_argument(f'--{f.name.replace("_", "-")}', action=argparse.BooleanOptionalAction, default=default)
    args = parser.parse_args()
    options = Options(**{f.name: getattr(args, f.name) for f in fields(Options)})

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('mps')
    settings = dust.recipe(args.tokens, args.population, args.optimizer)
    if args.lr is not None:
        settings.lr = args.lr
    if args.momentum is not None:
        settings.momentum = args.momentum
    for stream, factor in (args.scale_draws or {}).items():
        value = getattr(settings, stream)
        if isinstance(value, list):
            setattr(settings, stream, [int(v * factor) for v in value])
        elif isinstance(value, dict):
            setattr(settings, stream, {k: int(v * factor) for k, v in value.items()})
        else:
            setattr(settings, stream, int(value * factor))
    if args.writers is not None:
        settings.writers = list(args.writers)
    defaults = dust.DEFAULT_CONFIGS[args.tokens]
    config = GPTConfig()
    steps = defaults['steps']
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    paths = dict(train=args.data_dir / f'train_{args.tokens}.pt', val=args.data_dir / 'val.pt',
                 test=args.data_dir / 'test.pt')
    training = dust.load_sequences(paths['train'], steps * 8)
    validation = dust.load_sequences(paths['val'], 544)[:args.eval_sequences]
    testing = dust.load_sequences(paths['test'], 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    model = dust.make_model(config, device, args.seed)
    if options.compile:
        dust.compile_model(model)  # for the clean evaluation and the residual-scalar draws
    optimizer = dust.make_optimizer(model, settings)
    estimator = FastDUST(model, settings, options, args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        method='fastdust', tokens=args.tokens, population=args.population, seed=args.seed, steps=steps,
        options=asdict(options), tiers=options.tiers(), recipe=asdict(settings),
        eval_sequences=args.eval_sequences, eval_interval=args.eval_interval or defaults['eval_interval'],
        source_sha256=dust.file_hash(Path(__file__)), dust_sha256=dust.file_hash(Path(dust.__file__)),
        torch_version=torch.__version__, device=device.type,
    )
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    best = dust.evaluate(model, validation, device)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'best.pt')
    history = [dict(step=0, val_loss=best)]
    interval = args.eval_interval or defaults['eval_interval']
    start = time.monotonic()
    print(f'fastdust {args.tokens}: {steps} steps, population={args.population}, tiers={options.tiers()}, '
          f'val={best:.4f}', flush=True)
    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            value = dust.evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / 'best.pt')
        history.append(record)
        print(json.dumps(record), flush=True)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
    result = dict(best_val=best, test_at_best_val=dust.evaluate(model, testing, device), history=history)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Best validation: {best:.4f}; test: {result['test_at_best_val']:.4f}", flush=True)


if __name__ == '__main__':
    main()
