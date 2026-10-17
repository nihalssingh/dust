"""Summarize training runs: training-only seconds per update, best validation, test at best validation.

    python mac/summarize.py ../runs/full/* ../runs/bp_sgd_1M_s42
"""
import json
from pathlib import Path
import statistics
import sys

PAPER = {  # Table 1 of the paper (test loss at the validation-selected checkpoint, mean of 3 seeds)
    ('dust', '1M', 256): 6.067, ('dust', '1M', 1024): 5.934, ('dust', '10M', 256): 5.231,
    ('backprop', '1M', None): 5.959, ('backprop', '10M', None): 4.989,
    ('backprop-adam', '1M', None): 5.361,  # Figure 3
    # Figure 3, DUST under Adam at 1M tokens (5-seed means, from the figure's data)
    ('dust-adam', '1M', 64): 6.078, ('dust-adam', '1M', 256): 5.836, ('dust-adam', '1M', 1024): 5.594,
    ('dust-adam', '1M', 4096): 5.500, ('dust-adam', '1M', 16384): 5.402,
}


def describe(run):
    cfg = json.loads((run / 'config.json').read_text())
    if cfg.get('method') == 'backprop':
        target = cfg.get('target', 'full')
        name = 'backprop' if cfg.get('optimizer', 'sgd') == 'sgd' else f"backprop-{cfg['optimizer']}"
        if target != 'full':
            name += f' ({target} target, analysis)'
        return name, cfg['tokens'], None, PAPER.get((name, cfg['tokens'], None))
    tiers = cfg.get('tiers') or {}
    level = 'A2' if 'A2' in tiers.values() else 'A1' if 'A1' in tiers.values() else \
        'E+N' if 'N' in tiers.values() else 'E'
    optimizer = (cfg.get('recipe') or {}).get('optimizer', 'sgd')
    name = f'dust {level}' if cfg.get('method') == 'fastdust' else 'dust reference'
    key = 'dust' if optimizer == 'sgd' else f'dust-{optimizer}'
    if optimizer != 'sgd':
        name += f' ({optimizer})'
    return name, cfg['tokens'], cfg.get('population'), PAPER.get((key, cfg['tokens'], cfg.get('population')))


def train_seconds(history):
    """Median training time per update, from consecutive records with no evaluation in between."""
    gaps = [b['elapsed_seconds'] - a['elapsed_seconds'] for a, b in zip(history[1:], history[2:])
            if 'val_loss' not in a and 'elapsed_seconds' in a]
    return statistics.median(gaps) if gaps else float('nan')


def main(paths):
    rows = []
    for path in paths:
        run = Path(path)
        if not (run / 'result.json').exists():
            continue
        result = json.loads((run / 'result.json').read_text())
        name, tokens, population, paper = describe(run)
        history = result['history']
        rows.append(dict(run=run.name, method=name, tokens=tokens, population=population,
                         sec=train_seconds(history), total_min=history[-1]['elapsed_seconds'] / 60,
                         best_val=result['best_val'], test=result['test_at_best_val'], paper=paper))
    print(f'{"run":28s} {"method":34s} {"tok":>4s} {"pop":>5s} {"s/upd":>6s} {"min":>6s} {"best val":>9s} '
          f'{"test":>7s} {"paper":>6s}')
    for r in sorted(rows, key=lambda r: (r['tokens'], r['test'])):
        print(f'{r["run"][:28]:28s} {r["method"][:34]:34s} {r["tokens"]:>4s} {str(r["population"] or ""):>5s} '
              f'{r["sec"]:6.1f} {r["total_min"]:6.1f} {r["best_val"]:9.4f} {r["test"]:7.4f} '
              f'{r["paper"] if r["paper"] else "":>6}')


if __name__ == '__main__':
    main(sys.argv[1:])
