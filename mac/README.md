# DUST on Apple silicon

This branch (`mac-mps`, forked from `qlabs-eng/dust` at `b20f7c0`) runs the paper's code on an Apple-silicon
Mac, makes it several times cheaper per update, and adds a forward-only LoRA trainer for Hugging Face models.
Every change is labeled with what it does to the algorithm:

| tier | meaning | how it is verified |
|---|---|---|
| **E**, exact | the paper's estimator; with the same noise, the same numbers (engineering only) | bit-for-bit against `dust.py` on CPU in float32 with the same noise (`mac/test_fastdust.py`), and the port's `dust.py` bit-for-bit against upstream (`mac/equivalence_check.py`) at 256 tokens, too short to reach the query-chunked window attention; that path matches one band mask within float32 round-off on CPU (`mac/test_fastdust.py`) and bitwise on MPS in `mac/window_attention_bench.py`'s copy of it; on MPS the same estimator up to kernel rounding |
| **N**, numerical | the same algorithm with different arithmetic (kernels, fusion, precision) | the estimate's cosine with the true gradient is unchanged (`mac/bench.py`) |
| **A1**, algorithmic, same target | a different estimator of the same expected update (the paper's same-token target), with less variance or cost | `exact_head` against backprop's head gradient (`mac/test_fastdust.py`), and Dust-LoRA's exact last layer and MLP map against autograd (`mac/test_dust_lora.py`); otherwise cosine with the true gradient (`mac/bench.py`, and for Dust-LoRA `mac/test_dust_lora.py` on a tiny random Llama) and training loss |
| **A2**, approximation | changes what is estimated (biased, reduced or hybrid) | cosine and training loss; the bias is stated |

Tested on a MacBook Pro (M5, 10-core, 24 GB unified memory), macOS 26, PyTorch 2.14.1.

## Setup

```bash
uv venv --python 3.12 ../.venv
uv pip install --python ../.venv/bin/python torch==2.14.1 tokenizers==0.23.1 datasets==4.8.4
../.venv/bin/python prepare_data.py --output ../data      # streams FineWeb, about a minute
../.venv/bin/python mac/test_fastdust.py                   # exactness tests, CPU
../.venv/bin/python mac/cosine_probe.py --self-test        # same-token target against brute force, CPU
```

`prepare_data.py` can hang at exit after it has written every split. The data are complete once
`../data/manifest.json` is written and the four files match its sha256s; then stop the process.

PyTorch 2.14 matters: its MPS attention kernel is about 16x faster than 2.10's for this model.

### LoRA fine-tuning

`mac/dust_lora.py` runs in a second environment with Hugging Face `transformers`, reads tiny-shakespeare from
`../data/shakespeare/input.txt` (its default `--data`; `prepare_data.py` does not write it) and loads the
gated `meta-llama/Llama-3.2-1B`: accept its license on Hugging Face and log in before the first run, which
downloads it. Once it is cached, set `HF_HUB_OFFLINE=1` so runs load it without contacting the Hub.

```bash
uv venv --python 3.12 ../.venv-hf
uv pip install --python ../.venv-hf/bin/python torch==2.14.1 transformers==5.20.1
mkdir -p ../data/shakespeare
curl -L -o ../data/shakespeare/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt  # 1,115,394 bytes
../.venv-hf/bin/hf auth login
../.venv-hf/bin/python mac/test_dust_lora.py               # CPU tests on a tiny random Llama
```

## Running

```bash
source ../.venv/bin/activate  # the environment from Setup (the LoRA command below runs in ../.venv-hf)

# the paper's code, on MPS (bf16 autocast like the paper's CUDA runs), compiled
python dust.py --tokens 1M --population 256 --compile --data-dir ../data --output ../runs/ref

# the same estimator, faster (tiers E + N)
python fastdust.py --tokens 1M --population 256 --compile --data-dir ../data --output ../runs/exact

# "A1 best": same expected update as the paper, much lower variance; 14.4 s per update, about 2x cheaper than
# the command above (31.1 s) and 3x cheaper than compiled dust.py (44.1 s)
python fastdust.py --tokens 1M --population 256 --compile --exact-head --derive-hidden --derive-attn-out \
  --block-output-draws --derive-embedding --closed-form-local --loo-baseline --exact-last-block \
  --data-dir ../data --output ../runs/a1

# backprop baselines: SGD (default); --optimizer adam|adamw with --weight-decay on the matrices (default 0,
# where adamw equals adam); --checkpointing (not with --compile); --target same-token (analysis)
python baselines/backprop.py --tokens 1M --compile --data-dir ../data --output ../runs/bp

# LoRA fine-tuning of a local Hugging Face model without a backward pass (second environment, see Setup), with
# the best measured settings, about 8 minutes; --method backprop or mezo to compare
../.venv-hf/bin/python mac/dust_lora.py --method dust --control-variate --layers 8 --draws-per-layer 4 \
  --lr 3e-4 --model meta-llama/Llama-3.2-1B --steps 60 --output ../runs/lora/dust
```

`mac/bench.py` measures any variant's seconds per update, peak memory and cosine with the float32 backprop
gradient on a fixed batch in about a minute; `mac/run_queue.sh` runs jobs from a queue file one at a time
(start it under `caffeinate -ims` so the Mac stays awake).

## The changes

### Port (E/N)

| change | tier | notes |
|---|---|---|
| CUDA-or-MPS device selection, bf16 autocast and bf16 embedding storage on MPS | N | mirrors the paper's CUDA setup |
| full-window attention via `is_causal` | E | the window covers the sequence, so the mask is exactly causal |
| sliding-window attention in query chunks of 256, each against only its in-window keys | E | bitwise identical on MPS in `mac/window_attention_bench.py` (its own copy of the chunking), within float32 round-off on CPU (`mac/test_fastdust.py`); the attention call in the six windowed blocks runs 1.7-1.8x faster (batch 8 and 16) |
| `torch.compile` (`--compile`) | N | 2x; cosines with the true gradient unchanged |

Not tiered, since they switch to the paper's own Adam configuration or only change the baseline:
`--optimizer adam` in `dust.py` and `fastdust.py` (the paper's Adam settings, Appendix F Table 7, and its Adam
draw totals and embedding noise, Appendix E Table 3, with an assumed per-block split, see Caveats), and the
backprop baseline's `--optimizer adam|adamw`, `--weight-decay` and `--checkpointing` (per-block activation
checkpointing; not with `--compile`).

### `fastdust.py` options

| option | tier | what it does |
|---|---|---|
| `truncate` (default) | E | a draw that jitters block l reruns only from the jittered module onward, from cached clean activations |
| (functional forward, per-chunk local/head kernels, native-dtype caches) | E / N when compiled | same math, fewer passes and less memory |
| `exact_head` | A1 | the head's error from the analytic softmax/cross-entropy derivative instead of 1,024 slab draws (cos 1) |
| `block_output_draws` | A1 | draws jitter only each block's output; the attention projection's error is that error mapped back through the MLP's exact per-token Jacobian. Removes the co-jittered site's interference |
| `derive_hidden` | A1 | MLP hidden error = 2 relu(h) * (W_proj^T e): no MLP-hidden draws |
| `derive_attn_out` | A1 | attention-output error = W_o^T e: no attention-output draws |
| `derive_embedding` | A1 | token-embedding error from the block errors through the residual mix (and block 0's same-token attention map): no embedding draws |
| `closed_form_local` | A1 | the attention internals' estimates replaced by the closed-form expectation of the paper's own local estimator (credit decay gamma from the recipe: 0.98, or 0.99 at 10M tokens with population 16,384; own-token queries): no local draws, no local sampling noise |
| `loo_baseline` | A1 | each token's reward baseline is the mean over all of a site's draws (leave-one-out) instead of the mean of its 2-draw chunk: K - 1 effective draws instead of K / 2. The paper's (n - 1) / n step scale is kept so its tuned learning rates still apply |
| `exact_last_block` | A1 | the last block's output error exactly, from the exact head error through the final norm; needs `block_output_draws` and `exact_head` |
| `skip_control_variate` | A1 | each block-output draw's skip-path part, (product of later residual scalars) x the exact last-block error, is subtracted from every token's loss change and added back afterwards: unbiased, and worth more than doubling the draws; needs `loo_baseline`, `block_output_draws` and `exact_head` |
| `orthogonal_scalar_es` | A1 | the residual-scalar ES directions are rows of a random rotation (same norm), so they do not overlap |
| `antithetic`, `rademacher` | A1 | no measurable benefit here (the noise is first-order interference, which they do not cancel); `antithetic` cannot be combined with `loo_baseline` |
| `local_attention_backward` | A2 | attention internals from one attention layer's exact local backward (credit decay 1 instead of gamma: a different target) |
| `scalars_from_errors` | A2 | residual scalars from the block errors (same-token target) instead of weight-space ES (full target); measured worse, not recommended |
| `freeze_attention`, `freeze_scalars` | A2 | no updates for those parameters |

`fastdust.py` also takes `--optimizer adam` (the paper's Adam settings, as in `dust.py`),
`--writers '[w1, ..., w8]'`, which replaces the recipe's writer draws per block (the block-output draws with
`block_output_draws`), and `--scale-draws '{"writers": 2}'`, which multiplies recipe streams (`writers`,
`hidden`, `hubs`, `embedding`, `head`, `local`) and truncates to whole draws. Counts must stay multiples of
the draws per forward pass (2 at population 256), head draws multiples of that times the 16 vocabulary slabs
(32 at population 256), and `loo_baseline` needs at least 2 draws per drawn block.

Two analysis tools, not tiered: `mac/cosine_probe.py` computes DUST's infinite-population target (the
same-token gradient) with an attention whose backward cuts cross-token paths, verified against brute force.
It matches DUST's target exactly for the direct-loss sites and queries; for keys, values, gates and value
embeddings it uses credit decay 1 instead of gamma; for the residual scalars it gives the same-token gradient,
while DUST's weight-space ES targets the full one. `baselines/backprop.py --target same-token` trains on an
approximation of DUST's target.

### `mac/dust_lora.py`: fine-tuning a Hugging Face model without a backward pass

LoRA adapters sit on each layer's two residual writers (`o_proj`, `down_proj`). `--method dust` (A1) uses the
counterparts of `fastdust.py`'s `block_output_draws`, `loo_baseline` and `exact_last_block`: an update runs
one clean forward, then for each adapted layer jitters that layer's output at every token and reruns only the
layers above it, rewarding each token's jitter by its own loss change with a leave-one-out baseline. The last
layer's error is exact (a per-token derivative of the final norm, head and cross-entropy; `--no-exact-last`
draws it instead), `--control-variate` (A1, as `skip_control_variate`) subtracts each draw's skip-path term
known from it, `o_proj`'s error comes from `down_proj`'s through the MLP's exact per-token Jacobian, and the
LoRA gradients are outer products with the cached adapter inputs. No backward pass runs through the network:
the only autograd graphs are one layer's MLP and, for the exact last-layer error, the final norm, head and
cross-entropy. `--layers N` adapts only the top N layers, which makes draws much cheaper.

`--method dust-bottleneck` (A2) trains frozen-B adapters (B a fixed random orthonormal basis, A from zero, so
only A trains): each draw jitters every adapter's rank-r bottleneck `A x` at every token at once and reruns
the whole network, `--draws-per-layer` counts these joint draws, and `--control-variate` subtracts each
adapter's skip-path term. `--frozen-b` (A2) gives the other methods the same adapters.
`--method backprop [--checkpointing]` and `--method mezo` (two-point weight-space ES, `--mezo-pairs` +/- pairs
per step) are baselines, not tiered, with the same adapters, data and evaluation.

## Results

One seed per cell unless stated; the paper reports 3-seed means (5-seed for its Adam results, Figure 3).
MacBook Pro M5 (24 GB), batch 8 x 2048 tokens, the paper's tuned learning rates and momentum throughout.

### Replication at 1M and 10M tokens (test loss at the best-validation checkpoint)

| | this Mac | paper |
|---|---|---|
| backprop, SGD | 5.944 | 5.959 |
| backprop, Adam | 5.368 | 5.361 |
| DUST, population 256, the paper's estimator (`fastdust.py --compile`, E + N: `dust.py`'s estimator up to kernel rounding) | 6.041 | 6.067 |
| backprop, SGD, 10M tokens | 5.030 | 4.989 |
| backprop, Adam (the paper's 1M settings), 10M tokens | 3.949 | |

### Cheaper DUST at 1M tokens (SGD)

| estimator | loss-rewarded draws / update | s / update | test | paper cell of similar quality |
|---|---|---|---|---|
| the paper's code as written (eager) | 256 | ~95 | | |
| the paper's estimator, E + N | 256 | 31.1 | 6.041 | population 256: 6.067 |
| A1 best, quarter allocation | 30 | 7.1 | 6.032 | population 256: 6.067 (population 64: 6.290) |
| A1 best+, half allocation | 58 | 9.6 | 5.944 | population 1024: 5.934 |
| A1 best | 108 | 14.4 | 5.940 | population 1024: 5.934 |
| A1 best+ | 108 | 14.2 | 5.915 | population 4096: 5.911 |
| *backprop, SGD* | | *1.1* | *5.944* | |

"A1 best" is `--exact-head --derive-hidden --derive-attn-out --block-output-draws --derive-embedding
--closed-form-local --loo-baseline --exact-last-block`; "A1 best+" adds `--skip-control-variate
--orthogonal-scalar-es`. Both keep the paper's expected update. The A2 hybrid with local attention backward
(`--local-attention-backward` instead of `--closed-form-local`) reached 5.941, the same as A1 best.

The quarter and half allocations set the block-output draws per block with
`--writers '[10, 8, 4, 2, 2, 2, 2, 2]'` and `--writers '[20, 16, 8, 6, 4, 2, 2, 2]'`: the population-256 tilt,
`[38, 32, 16, 10, 6, 4, 2, 2]`, scaled down in whole 2-draw chunks with at least one chunk per block. The
quarter allocation is not the paper's population-64 allocation (Appendix E: `(8, 7, 4, 3, 1, 1, 1, 1)` writer
draws). With `exact_last_block` the last block draws nothing, so the table's draw counts leave out its 2.

Under Adam (`--optimizer adam`: the paper's Table 7 settings and the Adam allocation, which gives A1 best+ 62
block-output draws): DUST A1 best+ 5.673 at 10.0 s / update, against the paper's DUST at population 256,
5.836, and backprop with Adam, 5.368.

### Where the speed and quality come from

Speed of one update at population 256 (fixed batch, after compilation): the paper's code as written runs at
about 95 s on MPS (eager); `torch.compile` (N) brings it to 47.5 s; query-chunked window attention (E) to
44.1 s; truncated draws through a functional forward (E) to 30-32 s, with the same estimator. From there each
A1 option is added in turn. Cosine of each estimate with the float32 backprop gradient, mean over layers, at a
checkpoint trained by DUST for 1M tokens (two noise seeds; the same ladder was also measured at
initialization, in local results that are not in the repo):

| variant | tiers | s / update | speedup vs E+N | peak GB | head | attn out | MLP out | MLP in | embedding | q | k | v | gate | value emb | scalars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E+N | E+N | 30.4 | 1.00x | 7.1 | 0.835 | 0.428 | 0.265 | 0.160 | 0.218 | 0.024 | 0.003 | 0.022 | 0.040 | 0.007 | 0.702 |
| L1 +exact_head | E+N+A1 | 29.6 | 1.03x | 7.1 | 1.000 | 0.428 | 0.265 | 0.160 | 0.218 | 0.024 | 0.004 | 0.025 | 0.069 | 0.004 | 0.481 |
| L2 +derive_hidden | E+N+A1 | 19.5 | 1.56x | 6.2 | 1.000 | 0.418 | 0.262 | 0.203 | 0.215 | 0.024 | 0.002 | 0.030 | 0.020 | 0.009 | 0.489 |
| L3 +derive_attn_out | E+N+A1 | 17.4 | 1.74x | 6.2 | 1.000 | 0.418 | 0.262 | 0.203 | 0.215 | 0.038 | 0.007 | 0.032 | 0.037 | 0.007 | 0.514 |
| L4 +block_output_draws | E+N+A1 | 16.6 | 1.83x | 6.2 | 1.000 | 0.497 | 0.367 | 0.292 | 0.216 | 0.043 | 0.003 | 0.038 | 0.067 | 0.013 | 0.454 |
| L5 +derive_embedding | E+N+A1 | 15.7 | 1.93x | 6.5 | 1.000 | 0.497 | 0.367 | 0.292 | 0.443 | 0.045 | 0.002 | 0.041 | 0.052 | 0.008 | 0.464 |
| L6 +closed_form_local | E+N+A1 | 13.7 | 2.22x | 5.9 | 1.000 | 0.497 | 0.367 | 0.292 | 0.443 | 0.233 | 0.086 | 0.273 | 0.101 | 0.131 | 0.532 |
| L7 +loo_baseline | E+N+A1 | 13.8 | 2.20x | 5.9 | 1.000 | 0.563 | 0.437 | 0.361 | 0.565 | 0.297 | 0.113 | 0.325 | 0.121 | 0.165 | 0.532 |
| L8 +exact_last_block | E+N+A1 | 13.8 | 2.20x | 6.0 | 1.000 | 0.650 | 0.538 | 0.468 | 0.565 | 0.405 | 0.189 | 0.425 | 0.307 | 0.359 | 0.455 |
| L9 +skip_control_variate | E+N+A1 | 14.0 | 2.17x | 6.3 | 1.000 | 0.768 | 0.661 | 0.587 | 0.666 | 0.524 | 0.242 | 0.546 | 0.420 | 0.433 | 0.455 |
| L10 +orthogonal_scalar_es | E+N+A1 | 14.0 | 2.17x | 6.3 | 1.000 | 0.768 | 0.661 | 0.587 | 0.666 | 0.524 | 0.242 | 0.546 | 0.420 | 0.433 | 0.359 |

Every rung keeps the paper's expected update. Cosine squared grows at most in proportion to population
(doubling A1 best's block-output draws at initialization raises it only 1.3 to 1.6x), so the last rung is
worth at least 3x (attention out) to 13x (MLP in) the draws of the first, at half the time per update; the
attention internals go from noise to useful estimates.

### Against backprop

- **Compute.** At equal test loss, A1 best+ (half allocation) costs 9.6 s per update against backprop's 1.1 s,
  about 9x; the paper's estimator at the same quality (population 1024) would cost about 4 x 31 s. At equal
  wall-clock the gap is larger: in the 14-18 minutes a DUST run takes on 1M tokens, backprop trains on 10M
  tokens: 5.030 with SGD and 3.949 with Adam, against DUST's best 1M results, 5.915 with SGD and 5.673 with
  Adam.
- **Memory.** Peak transient memory of one update (8 x 2048 tokens, above the resident model):

| method | GB |
|---|---|
| inference forward | 0.9 |
| backprop | 12.7 |
| backprop with per-block activation checkpointing | 3.7 |
| DUST, `dust.py` | 6.5 |
| DUST, `fastdust.py` E (truncated) | 7.5 |
| DUST, A1 best / A1 best+ | 6.1 / 6.4 |

  DUST needs about half of plain backprop's memory but more than backprop with checkpointing: it keeps clean
  activations (like checkpoint boundaries) plus the draws' forward buffers.
- **Adam.** Backprop gains 0.58 nats from Adam at 1M tokens (5.944 to 5.368) and 1.08 nats at 10M (5.030 to
  3.949). DUST gains less (A1 best+ 5.915 to 5.673 at 1M), so under Adam it trails backprop by 0.30 nats at
  1M (the paper's population 256: 0.48). DUST's parity with backprop holds only when both use SGD.

### Fine-tuning Llama-3.2-1B with LoRA (`mac/dust_lora.py`)

Rank-8 adapters on every `o_proj` and `down_proj` (or the top 8 layers), tiny-shakespeare (90/10 split),
batch 4 x 512 tokens, 60 steps, Adam, held-out loss on 32 sequences. Every DUST run used `--control-variate`.

| method | s / step | peak GB | held-out loss |
|---|---|---|---|
| backprop, all layers | 1.82 | 11.8 | 3.081 -> 2.981 |
| backprop with checkpointing, all layers | 2.38 | 4.8 | 3.081 -> 2.981 |
| backprop, top 8 layers | 1.21 | 8.1 | 3.081 -> 3.003 |
| backprop, frozen random B (A only) | 1.80 | 11.8 | 3.081 -> 3.050 |
| **DUST, top 8 layers, 4 draws / layer, lr 3e-4** | 7.71 | 5.6 | 3.081 -> **3.047** |
| DUST, top 8 layers, lr 1e-4 | 7.71 | 5.6 | 3.081 -> 3.062 |
| DUST, all layers, 2 draws / layer, lr 3e-4 | 11.89 | 5.6 | 3.081 -> 3.060 |
| DUST, all layers, lr 1e-3 | 11.97 | 5.6 | diverged (4.019) |
| bottleneck DUST (A2, frozen B, 16 draws), lr 3e-4 | 9.83 | 5.6 | 3.081 -> 3.122 (worse) |
| MeZO (weight-space ES, 11 +/- pairs), lr 3e-4 | 12.55 | 4.1 | 3.081 -> 3.077 |
| MeZO, lr 1e-3 | 12.30 | 4.1 | 3.081 -> 3.087 |

DUST fine-tunes a 1B model on this Mac with no backward pass through the network, and at matched compute per
step it makes about 5x the progress of weight-space ES (all layers: -0.021 against -0.004). It is about 13x to
31x less efficient than backprop in wall-clock to the same loss, depending on the loss (top 8 layers,
evaluated every 10 steps: DUST's final 3.047 took 462.4 s of updates, and backprop first reaches it at step
30, after 36.4 s; DUST first reaches 3.050 at step 50, after 385.6 s, and backprop at step 10, after 12.3 s),
needs a small learning rate (Adam at 1e-3 diverges on its noisy estimates), and uses a little more memory than
backprop with checkpointing.

### What did not help

- `antithetic` and `rademacher` noise: no measurable change; the dominant noise is first-order interference.
- `scalars_from_errors` (A2): residual-scalar cosine with the full gradient 0.03 at initialization against
  0.63 for the paper's ES.
- `local_attention_backward` (A2): much better attention-internal cosines than the closed form, the same
  training loss at 1M tokens.
- Bottleneck DUST for LoRA (A2): worse than no training at lr 3e-4, the only learning rate tried with the code
  in this branch.

## Caveats

- One seed per training cell (the paper uses three, and five for its Adam ladder); differences under about
  0.02 nats are within seed noise.
- Gradient-quality numbers are cosines on one fixed batch at two checkpoints (initialization and a 1M-token
  DUST checkpoint), two noise seeds each. A backprop-trained checkpoint is a poor benchmark here: its residual
  stream grows to magnitudes near 2,000 in deep blocks, where bf16 spacing swallows a 0.2 jitter.
- The paper does not give the per-block split of its Adam allocation; `dust.py --optimizer adam` (and
  `fastdust.py --optimizer adam`, which uses the same recipe) rescales the SGD split to the paper's Adam
  totals (an assumption).
