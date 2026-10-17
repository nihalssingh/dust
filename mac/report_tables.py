"""Markdown tables for mac/README.md, straight from the result files.

    python mac/report_tables.py ../runs
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import describe, train_seconds  # noqa: E402

TYPES = [('head', 'head'), ('attention out', 'attn out'), ('MLP out', 'MLP out'), ('MLP in', 'MLP in'),
         ('token embedding', 'embedding'), ('query', 'q'), ('key', 'k'), ('value', 'v'), ('value gate', 'gate'),
         ('value embedding', 'value emb'), ('residual scalars', 'scalars')]


def bench_table(paths, baseline):
    rows = [json.loads(p.read_text()) for p in paths]
    base = next(r for r in rows if r['config'] == baseline)
    sec = lambda r: sum(c['seconds'] for c in r['cells']) / len(r['cells'])
    gb = lambda r: sum(c['peak_gb'] for c in r['cells']) / len(r['cells'])
    cos = lambda r, t: sum(c['cos'][t]['full'] for c in r['cells']) / len(r['cells'])
    out = [f'| variant | tiers | s / update | speedup vs {baseline} | peak GB | ' + ' | '.join(s for _, s in TYPES) + ' |',
           '|---|---|---|---|---|' + '---|' * len(TYPES)]
    for r in rows:
        tiers = sorted(set(r.get('tiers', {}).values()), key=['E', 'N', 'A1', 'A2'].index) if r.get('tiers') else ['N']
        label = r['config'] + (f" (writers x{json.loads(r['scale'])['writers']})" if r.get('scale') else '')
        out.append(f"| {label} | {'+'.join(dict.fromkeys(tiers))} | {sec(r):.1f} | {sec(base) / sec(r):.2f}x | "
                   f"{gb(r):.1f} | " + ' | '.join(f'{cos(r, t):.3f}' for t, _ in TYPES) + ' |')
    return '\n'.join(out)


def runs_table(paths):
    out = ['| run | method | s / update | wall-clock (min) | best val | test at best val | paper |',
           '|---|---|---|---|---|---|---|']
    rows = []
    for run in paths:
        if not (run / 'result.json').exists():
            continue
        result = json.loads((run / 'result.json').read_text())
        name, tokens, population, paper = describe(run)
        rows.append((tokens, result['test_at_best_val'], f"| {run.name} | {name} | {train_seconds(result['history']):.1f} | "
                     f"{result['history'][-1]['elapsed_seconds'] / 60:.1f} | {result['best_val']:.4f} | "
                     f"{result['test_at_best_val']:.4f} | {paper or ''} |"))
    return '\n'.join(out + [r[2] for r in sorted(rows)])


def lora_table(paths):
    out = ['| run | method | s / step | peak GB | held-out loss start -> end |', '|---|---|---|---|---|']
    for run in paths:
        f = run / 'result.json'
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        out.append(f"| {run.name} | {r['method']} | {r['seconds_per_step']:.2f} | {r['peak_memory_gb']:.1f} | "
                   f"{r['history'][0]['eval_loss']:.4f} -> {r['final_eval']:.4f} |")
    return '\n'.join(out)


def main(root):
    root = Path(root)
    v2 = root / 'bench' / 'v2'
    for tag, baseline in (('init', 'reference+N'), ('bp1M', 'E+N')):
        paths = sorted(v2.glob(f'{tag}_*.json'))
        if paths:
            print(f'### Gradient quality and speed, {tag}\n')
            print(bench_table(paths, baseline) + '\n')
    runs = sorted([p for p in (root / 'full').glob('*') if p.is_dir()] + [root / 'bp_sgd_1M_s42'])
    print('### Training runs\n\n' + runs_table(runs) + '\n')
    mem = root / 'bench' / 'memory.json'
    if mem.exists():
        print('### Peak transient memory of one update (8 x 2048 tokens)\n\n| | GB |\n|---|---|')
        for k, v in json.loads(mem.read_text()).items():
            print(f'| {k} | {v:.2f} |')
        print()
    lora = sorted(p for p in (root / 'lora').glob('*') if p.is_dir())
    if lora:
        print('### LoRA fine-tuning of Llama-3.2-1B on tiny-shakespeare\n\n' + lora_table(lora))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../runs')
