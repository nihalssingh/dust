"""Minimal SGD backpropagation baseline with the same model, data, and evaluation as DUST."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mac'))
from dust import GPTConfig, compile_model, evaluate, file_hash, load_sequences, make_model
from model import BF16_DEVICES, norm


DEFAULT_CONFIGS = {
    '1M': dict(lr=0.175, momentum=0.98, steps=61, eval_interval=2),
    '10M': dict(lr=0.2, momentum=0.95, steps=610, eval_interval=10),
}

# Backprop's Adam settings at 1M tokens (paper Appendix F, Table 7): matrices and head 0.002, token and value
# embeddings 0.3, residual scalars 0.2, beta1 0.8, beta2 0.95, eps 1e-8, no weight decay. --lr scales all three.
ADAM_DEFAULTS = dict(lr=0.002, embedding_lr=0.3, scalar_lr=0.2, beta1=0.8, beta2=0.95, eps=1e-8)


def make_adam(model, lr, beta1, weight_decay, decoupled):
    scale = lr / ADAM_DEFAULTS['lr']
    groups = {'matrices': [], 'embeddings': [], 'scalars': []}
    for name, parameter in model.named_parameters():
        if name == 'transformer.wte.weight' or 'value_embeds' in name:
            groups['embeddings'].append(parameter)
        elif 'lambdas' in name:
            groups['scalars'].append(parameter)
        else:
            groups['matrices'].append(parameter)
    param_groups = [
        {'params': groups['matrices'], 'lr': lr, 'weight_decay': weight_decay},
        {'params': groups['embeddings'], 'lr': ADAM_DEFAULTS['embedding_lr'] * scale, 'weight_decay': 0.0},
        {'params': groups['scalars'], 'lr': ADAM_DEFAULTS['scalar_lr'] * scale, 'weight_decay': 0.0},
    ]
    kind = torch.optim.AdamW if decoupled else torch.optim.Adam
    return kind(param_groups, betas=(beta1, ADAM_DEFAULTS['beta2']), eps=ADAM_DEFAULTS['eps'])


def make_optimizer(model, lr, momentum):
    groups = [
        {'params': [p for name, p in model.named_parameters()
                    if name != 'transformer.wte.weight'], 'lr': lr},
        {'params': [model.transformer.wte.weight], 'lr': 1000.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def checkpointed_forward(model, idx, targets):
    """GPT.forward with each block recomputed during backward (activation checkpointing)."""
    B, T = idx.size()
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    x = norm(model.transformer.wte(idx))
    x0 = x
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = checkpoint(block, x, ve, cos_sin, model.window_sizes[i], use_reentrant=False)
    x = norm(x)
    logits = model.lm_head(x)[..., :model.config.vocab_size].float()
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def same_token_step(model, optimizer, x, y):
    """Analysis only: an SGD step on (approximately) DUST's infinite-population target. Every direct-loss site
    gets the same-token gradient (each token's activation credited only with its own loss) from one backward
    pass with all cross-token paths cut; the attention internals (q, k, v, gate, value embeddings) take the
    full gradient from a second, ordinary pass instead of eight per-block passes (approximation), as do the
    residual scalars, whose weight-space ES targets the full gradient."""
    import cosine_probe
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type in BF16_DEVICES):
        grads, _ = cosine_probe.backprop(model, x, y, first_diagonal=0)
        full, _ = cosine_probe.backprop(model, x, y)
        for name, g in full.items():
            if ('attn.' in name and 'c_proj' not in name) or 'value_embeds' in name or 'lambdas' in name:
                grads[name] = g  # DUST's local draws and scalar ES target these through the full gradient
        with torch.no_grad():
            loss = model(x, y)
    for name, parameter in model.named_parameters():
        parameter.grad = grads[name].to(parameter.dtype)
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Non-finite same-token gradient for {name}')
    optimizer.step()
    return loss.item()


def train_step(model, optimizer, x, y, checkpointing=False):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type in BF16_DEVICES):
        loss = checkpointed_forward(model, x, y) if checkpointing else model(x, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite backpropagation loss')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Non-finite backpropagation gradient for {name}')
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/backprop'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, help='Override the base learning rate for the selected token budget.')
    parser.add_argument('--momentum', type=float, help='Override SGD momentum for the selected token budget.')
    parser.add_argument('--steps', type=int, help='Stop early; keep the full selected training subset.')
    parser.add_argument(
        '--optimizer', choices=['sgd', 'adam', 'adamw'], default='sgd',
        help="adam/adamw default to the paper's 1M Adam settings (Table 7); --lr sets the matrix rate and "
             'scales the embedding and scalar rates with it, --momentum sets beta1.',
    )
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Matrix weight decay (adam/adamw only).')
    parser.add_argument('--checkpointing', action='store_true', help='Recompute each block during backward.')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model.')
    parser.add_argument(
        '--target', choices=['full', 'same-token'], default='full',
        help="same-token: analysis run on an approximation of DUST's infinite-population target (2 backward passes "
             'per step).',
    )
    parser.add_argument(
        '--device', choices=['cuda', 'mps'],
        help='Default: CUDA when available, otherwise the Apple GPU (MPS).',
    )
    args = parser.parse_args()
    defaults = DEFAULT_CONFIGS[args.tokens]
    if args.optimizer == 'sgd':
        args.lr = defaults['lr'] if args.lr is None else args.lr
        args.momentum = defaults['momentum'] if args.momentum is None else args.momentum
    else:
        args.lr = ADAM_DEFAULTS['lr'] if args.lr is None else args.lr
        args.momentum = ADAM_DEFAULTS['beta1'] if args.momentum is None else args.momentum

    if args.compile and args.checkpointing:
        parser.error('--compile does not cover the checkpointed forward; use one or the other')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('This baseline uses one GPU; launch it with python, not torchrun.')
    if args.lr <= 0 or not 0 <= args.momentum < 1:
        parser.error('Use a positive learning rate and momentum in [0, 1).')
    device_type = args.device or (
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else None
    )
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda', 0)
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
    elif device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        parser.error('Training requires a CUDA GPU or an Apple GPU (MPS).')

    config = GPTConfig()
    steps = defaults['steps']
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    paths = dict(
        train=args.data_dir / f'train_{args.tokens}.pt',
        val=args.data_dir / 'val.pt',
        test=args.data_dir / 'test.pt',
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            f"Missing data: {', '.join(missing)}. Run python prepare_data.py "
            f"--tokens {args.tokens} --output {args.data_dir}"
        )
    training = load_sequences(paths['train'], steps * 8)
    validation = load_sequences(paths['val'], 544)
    testing = load_sequences(paths['test'], 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    model = make_model(config, device, args.seed).requires_grad_(True)
    eval_model = model
    if args.compile and args.target == 'same-token':
        eval_model = torch.compile(model)  # training passes swap the attention function, so stay eager there
    elif args.compile:
        compile_model(model)
    if args.optimizer == 'sgd':
        optimizer = make_optimizer(model, args.lr, args.momentum)
    else:
        optimizer = make_adam(model, args.lr, args.momentum, args.weight_decay, args.optimizer == 'adamw')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        method='backprop', tokens=args.tokens, seed=args.seed, steps=steps,
        optimizer=args.optimizer, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay if args.optimizer != 'sgd' else 0.0,
        checkpointing=args.checkpointing, device=device.type, compiled=args.compile, target=args.target,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=544, test_sequences=544, model=asdict(config),
        parameter_groups=[dict(lr=g['lr'], momentum=g.get('momentum'), betas=g.get('betas'),
                               parameters=sum(p.numel() for p in g['params']))
                          for g in optimizer.param_groups],
        source_sha256=file_hash(Path(__file__)),
        model_source_sha256=file_hash(Path(__file__).resolve().parents[1] / 'dust.py'),
        model_sha256=file_hash(Path(__file__).resolve().parents[1] / 'model.py'),
        torch_version=torch.__version__,
        data_sha256={name: file_hash(path) for name, path in paths.items()},
    )
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    best = evaluate(eval_model, validation, device)

    def save_best():
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            args.output / 'best.pt',
        )

    save_best()
    history = [dict(step=0, val_loss=best)]
    interval = defaults['eval_interval']
    start = time.monotonic()
    print(f'Backprop {args.optimizer}: {args.tokens}, {steps} steps, val={best:.4f}', flush=True)

    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        if args.target == 'same-token':
            loss = same_token_step(model, optimizer, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        else:
            loss = train_step(
                model, optimizer, batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), args.checkpointing,
            )
        record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            value = evaluate(eval_model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                save_best()
        history.append(record)
        print(json.dumps(record), flush=True)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
    result = dict(best_val=best, test_at_best_val=evaluate(model, testing, device), history=history)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Best validation: {best:.4f}; test: {result['test_at_best_val']:.4f}", flush=True)


if __name__ == '__main__':
    main()
