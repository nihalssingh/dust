"""Minimal SGD backpropagation baseline with the same model, data, and evaluation as DUST."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, compile_model, evaluate, file_hash, load_sequences, make_model
from model import BF16_DEVICES


DEFAULT_CONFIGS = {
    '1M': dict(lr=0.175, momentum=0.98, steps=61, eval_interval=2),
    '10M': dict(lr=0.2, momentum=0.95, steps=610, eval_interval=10),
}


def make_optimizer(model, lr, momentum):
    groups = [
        {'params': [p for name, p in model.named_parameters()
                    if name != 'transformer.wte.weight'], 'lr': lr},
        {'params': [model.transformer.wte.weight], 'lr': 1000.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type in BF16_DEVICES):
        loss = model(x, y)
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
    parser.add_argument('--compile', action='store_true', help='torch.compile the model.')
    parser.add_argument(
        '--device', choices=['cuda', 'mps'],
        help='Default: CUDA when available, otherwise the Apple GPU (MPS).',
    )
    args = parser.parse_args()
    defaults = DEFAULT_CONFIGS[args.tokens]
    args.lr = defaults['lr'] if args.lr is None else args.lr
    args.momentum = defaults['momentum'] if args.momentum is None else args.momentum

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
    if args.compile:
        compile_model(model)
    optimizer = make_optimizer(model, args.lr, args.momentum)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        method='backprop', tokens=args.tokens, seed=args.seed, steps=steps,
        lr=args.lr, momentum=args.momentum, weight_decay=0.0,
        device=device.type, compiled=args.compile,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=544, test_sequences=544, model=asdict(config),
        parameter_groups=[dict(lr=g['lr'], momentum=g['momentum'],
                               parameters=sum(p.numel() for p in g['params']))
                          for g in optimizer.param_groups],
        source_sha256=file_hash(Path(__file__)),
        model_source_sha256=file_hash(Path(__file__).resolve().parents[1] / 'dust.py'),
        model_sha256=file_hash(Path(__file__).resolve().parents[1] / 'model.py'),
        torch_version=torch.__version__,
        data_sha256={name: file_hash(path) for name, path in paths.items()},
    )
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    best = evaluate(model, validation, device)

    def save_best():
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            args.output / 'best.pt',
        )

    save_best()
    history = [dict(step=0, val_loss=best)]
    interval = defaults['eval_interval']
    start = time.monotonic()
    print(f'Backprop SGD: {args.tokens}, {steps} steps, val={best:.4f}', flush=True)

    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        loss = train_step(model, optimizer, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            value = evaluate(model, validation, device)
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
