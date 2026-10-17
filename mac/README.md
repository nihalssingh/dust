# DUST on Apple silicon

This branch (`mac-mps`, forked from `qlabs-eng/dust` at `b20f7c0`) runs the paper's code on an Apple-silicon
Mac and makes it several times cheaper per update.
Every change is labeled with what it does to the algorithm:

| tier | meaning | how it is verified |
|---|---|---|
| **E**, exact | the paper's estimator; with the same noise, the same numbers (engineering only) | bit-for-bit against `dust.py` on CPU in float32 with the same noise (`mac/test_fastdust.py`), and the port's `dust.py` bit-for-bit against upstream (`mac/equivalence_check.py`); on MPS the same estimator up to kernel rounding |
| **N**, numerical | the same algorithm with different arithmetic (kernels, fusion, precision) | the estimate's cosine with the true gradient is unchanged |
| **A1**, algorithmic, same target | a different estimator of the same expected update (the paper's same-token target), with less variance or cost | expectation checks against exact targets, then cosine and training loss |

Tested on a MacBook Pro (M5, 10-core, 24 GB unified memory), macOS 26, PyTorch 2.14.1.

## Setup

```bash
uv venv --python 3.12 ../.venv
uv pip install --python ../.venv/bin/python torch==2.14.1 tokenizers==0.23.1 datasets==4.8.4
../.venv/bin/python prepare_data.py --output ../data      # streams FineWeb, about a minute
../.venv/bin/python mac/test_fastdust.py                   # exactness tests, CPU
```

PyTorch 2.14 matters: its MPS attention kernel is about 16x faster than 2.10's for this model (sliding-window
SDPA at batch 8: 97 ms in 2.10, 6 ms in 2.14).

## Running

```bash
# the paper's code, on MPS (bf16 autocast like the paper's CUDA runs), compiled
python dust.py --tokens 1M --population 256 --compile --data-dir ../data --output ../runs/ref

# the same estimator, faster (tiers E + N)
python fastdust.py --tokens 1M --population 256 --compile --data-dir ../data --output ../runs/exact

# backprop baselines: SGD (default), --optimizer adam|adamw, --checkpointing, --target same-token (analysis)
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

### `fastdust.py` options

| option | tier | what it does |
|---|---|---|
| `truncate` (default) | E | a draw that jitters block l reruns only from the jittered module onward, from cached clean activations |
| (functional forward, per-chunk local/head kernels, per-sequence closed form, native-dtype caches) | E / N when compiled | same math, fewer passes and less memory |
| `exact_head` | A1 | the head's error from the analytic softmax/cross-entropy derivative instead of 1,024 slab draws (cos 1) |
| `block_output_draws` | A1 | draws jitter only each block's output; the attention projection's error is that error mapped back through the MLP's exact per-token Jacobian. Removes the co-jittered site's interference |
| `derive_hidden` | A1 | MLP hidden error = 2 relu(h) * (W_proj^T e): no MLP-hidden draws |
| `derive_attn_out` | A1 | attention-output error = W_o^T e: no attention-output draws |
| `derive_embedding` | A1 | token-embedding error from the block errors through the residual mix (and block 0's same-token attention map): no embedding draws |
| `closed_form_local` | A1 | the attention internals' estimates replaced by the closed-form expectation of the paper's own local estimator (credit decay 0.98, own-token queries): no local draws, no local sampling noise |
| `exact_last_block` | A1 | the last block's output error exactly, from the exact head error through the final norm |

Two analysis tools: `mac/cosine_probe.py` computes DUST's infinite-population target (the same-token gradient)
exactly with an attention whose backward cuts cross-token paths, verified against brute force;
`baselines/backprop.py --target same-token` trains on an approximation of it.

## Caveats

- The paper does not give the per-block split of its Adam allocation; `dust.py --optimizer adam` rescales the
  SGD split to the paper's Adam totals (an assumption).
