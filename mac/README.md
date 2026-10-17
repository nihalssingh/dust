# DUST on Apple silicon

This branch (`mac-mps`, forked from `qlabs-eng/dust` at `b20f7c0`) runs the paper's code on an Apple-silicon
Mac.
Every change is labeled with what it does to the algorithm:

| tier | meaning | how it is verified |
|---|---|---|
| **E**, exact | the paper's estimator; with the same noise, the same numbers (engineering only) | the port's `dust.py` bit-for-bit against upstream (`mac/equivalence_check.py`); on MPS the same estimator up to kernel rounding |
| **N**, numerical | the same algorithm with different arithmetic (kernels, fusion, precision) | the estimate's cosine with the true gradient is unchanged |

Tested on a MacBook Pro (M5, 10-core, 24 GB unified memory), macOS 26, PyTorch 2.14.1.

## Setup

```bash
uv venv --python 3.12 ../.venv
uv pip install --python ../.venv/bin/python torch==2.14.1 tokenizers==0.23.1 datasets==4.8.4
../.venv/bin/python prepare_data.py --output ../data      # streams FineWeb, about a minute
```

PyTorch 2.14 matters: its MPS attention kernel is about 16x faster than 2.10's for this model (sliding-window
SDPA at batch 8: 97 ms in 2.10, 6 ms in 2.14).

## Running

```bash
# the paper's code, on MPS (bf16 autocast like the paper's CUDA runs), compiled
python dust.py --tokens 1M --population 256 --compile --data-dir ../data --output ../runs/ref

# backprop baselines: SGD (default), --optimizer adam|adamw
python baselines/backprop.py --tokens 1M --compile --data-dir ../data --output ../runs/bp
```

## The changes

### Port (E/N)

| change | tier | notes |
|---|---|---|
| CUDA-or-MPS device selection, bf16 autocast and bf16 embedding storage on MPS | N | mirrors the paper's CUDA setup |
| full-window attention via `is_causal` | E | the window covers the sequence, so the mask is exactly causal |
| sliding-window attention in query chunks of 256, each against only its in-window keys | E | bitwise identical on MPS; 1.8x faster for the six windowed blocks |
| `torch.compile` (`--compile`) | N | 2x; cosines with the true gradient unchanged |

## Caveats

- The paper does not give the per-block split of its Adam allocation; `dust.py --optimizer adam` rescales the
  SGD split to the paper's Adam totals (an assumption).
