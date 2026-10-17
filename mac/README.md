# DUST on Apple silicon

This branch (`mac-mps`, forked from `qlabs-eng/dust` at `b20f7c0`) runs the paper's code on an Apple-silicon
Mac.
Every change is labeled with what it does to the algorithm:

| tier | meaning | how it is verified |
|---|---|---|
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
# the paper's code, on MPS (bf16 autocast like the paper's CUDA runs)
python dust.py --tokens 1M --population 256 --data-dir ../data --output ../runs/ref
```

## The changes

### Port (N)

| change | tier | notes |
|---|---|---|
| CUDA-or-MPS device selection, bf16 autocast and bf16 embedding storage on MPS | N | mirrors the paper's CUDA setup |
