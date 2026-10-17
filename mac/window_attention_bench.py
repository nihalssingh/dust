"""Is query-chunked SDPA (each chunk sees only its in-window keys) faster than one band-masked SDPA on MPS?"""
import time
import torch
from torch.nn import functional as F

dev = torch.device('mps')


def masked(q, k, v, window):
    T = q.shape[-2]
    p = torch.arange(T, device=q.device)
    d = p[:, None] - p[None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=(d >= 0) & (d <= window))


def chunked(q, k, v, window, chunk):
    T = q.shape[-2]
    outs = []
    for s0 in range(0, T, chunk):
        s1 = min(T, s0 + chunk)
        k0 = max(0, s0 - window)
        if k0 == 0 and s1 - 1 <= window:  # no window cut inside this chunk: causal over keys [0, s1)
            qp = torch.arange(s0, s1, device=q.device)[:, None]
            kp = torch.arange(0, s1, device=q.device)[None, :]
            outs.append(F.scaled_dot_product_attention(q[..., s0:s1, :], k[..., :s1, :], v[..., :s1, :],
                                                       attn_mask=qp >= kp))
            continue
        qp = torch.arange(s0, s1, device=q.device)[:, None]
        kp = torch.arange(k0, s1, device=q.device)[None, :]
        m = (qp - kp >= 0) & (qp - kp <= window)
        outs.append(F.scaled_dot_product_attention(q[..., s0:s1, :], k[..., k0:s1, :], v[..., k0:s1, :], attn_mask=m))
    return torch.cat(outs, dim=-2)


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1000


for B in (8, 16):
    q, k, v = (torch.randn(B, 8, 2048, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
    ref = masked(q, k, v, 1024)
    base = timeit(lambda: masked(q, k, v, 1024))
    line = f'B={B}: masked {base:6.2f} ms'
    for chunk in (256, 512, 1024):
        out = chunked(q, k, v, 1024, chunk)
        err = (out.float() - ref.float()).abs().max().item()
        line += f' | chunk{chunk} {timeit(lambda: chunked(q, k, v, 1024, chunk)):6.2f} ms (max err {err:.1e})'
    causal = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    print(line + f' | is_causal {causal:6.2f} ms', flush=True)
