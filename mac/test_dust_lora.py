"""CPU tests for dust_lora.py on a tiny random Llama: truncated reruns, the MLP map, and estimate quality."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dust_lora as dl  # noqa: E402


def main():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=512, max_position_embeddings=256)
    cfg._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
    adapters = dl.add_lora(model, rank=4, alpha=8)
    for ad in adapters.values():  # nonzero B so both factors get gradient
        ad.B.data.normal_(0, 0.02)
    x = torch.randint(0, 512, (2, 64))
    y = torch.randint(0, 512, (2, 64))
    cap = dl.Capture(model)
    ok = True

    # 1) zero-noise truncated reruns reproduce the clean per-token losses
    clean = cap.clean(x, y)
    kw = dl.repeat_kwargs(cap.kwargs, 2)
    worst = 0.0
    for l in range(3):
        z = torch.cat([cap.out[l], cap.out[l]])
        for j in range(l + 1, 3):
            z = model.model.layers[j](z, **kw)
            z = z[0] if isinstance(z, tuple) else z
        loss = dl.token_losses(model, z, y.repeat(2, 1)).view(2, 2, 64)
        worst = max(worst, (loss - clean).abs().max().item())
    print(f"{'PASS' if worst < 1e-5 else 'FAIL'} truncated rerun from cached layer outputs: max |dloss| {worst:.1e}")
    ok &= worst < 1e-5

    # 2) the MLP map against autograd through the whole layer tail (x_mid -> layer output)
    e = {l: torch.randn(2, 64, 64) for l in range(3)}
    mids = dl.mid_errors(model, cap, e)
    worst = 0.0
    for l, layer in enumerate(model.model.layers):
        xm = cap.mid[l].clone().requires_grad_(True)
        out = xm + layer.mlp(layer.post_attention_layernorm(xm))
        (g,) = torch.autograd.grad(out, xm, e[l])
        worst = max(worst, ((g - mids[l]).norm() / g.norm()).item())
    print(f"{'PASS' if worst < 1e-5 else 'FAIL'} MLP per-token map vs autograd: rel {worst:.1e}")
    ok &= worst < 1e-5

    # 3) estimate quality: cosine with backprop's LoRA gradients grows with draws
    params = [p for ad in adapters.values() for p in (ad.A, ad.B)]
    for p in params:
        p.requires_grad_(True)
    loss = dl.token_losses(model, model.model(input_ids=x, use_cache=False).last_hidden_state, y, normed=True).mean()
    loss.backward()
    true = {k: (ad.B.grad.clone(), ad.A.grad.clone()) for k, ad in adapters.items()}
    for p in params:
        p.requires_grad_(False)
        p.grad = None
    flat = lambda d: torch.cat([torch.cat([b.flatten(), a.flatten()]) for b, a in d.values()])
    last = -1.0
    for draws in (8, 64, 512):
        gen = torch.Generator().manual_seed(1)
        _, errors = dl.dust_gradients(model, adapters, cap, x, y, draws, 0.05, gen)
        est = dl.lora_gradients(adapters, cap, errors, dl.mid_errors(model, cap, errors), y.numel())
        cos = F.cosine_similarity(flat(est), flat(true), dim=0).item()
        print(f'     draws/layer {draws:4d}: cos(Dust-LoRA, backprop-LoRA) = {cos:.3f}')
        ok &= cos > last
        last = cos
    print(f"{'PASS' if last > 0.5 else 'FAIL'} estimate aligns with backprop as draws grow (final {last:.3f})")
    # 4) the exact last-layer error gives the last layer's adapters exactly backprop's gradient
    e_last = dl.last_layer_error(model, cap, y)
    mids_last = dl.mid_errors(model, cap, {**{l: torch.zeros_like(e_last) for l in range(2)}, 2: e_last})
    exact = dl.lora_gradients({k: v for k, v in adapters.items() if k.startswith('2.')}, cap,
                              {2: e_last}, mids_last, y.numel())
    worst = max(((g - t).norm() / t.norm()).item() for k in exact for g, t in zip(exact[k], true[k]))
    print(f"{'PASS' if worst < 1e-4 else 'FAIL'} exact last layer reproduces backprop's last-layer LoRA gradients: rel {worst:.1e}")
    ok &= worst < 1e-4
    ok &= last > 0.5
    # 5) top-layer adapters only: draws only at adapted layers, and the estimate still aligns with backprop
    model2 = LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
    ad2 = dl.add_lora(model2, rank=4, alpha=8, layers=2)
    for ad in ad2.values():
        ad.B.data.normal_(0, 0.02)
    cap2 = dl.Capture(model2)
    ps = [p for ad in ad2.values() for p in (ad.A, ad.B)]
    for p in ps:
        p.requires_grad_(True)
    dl.token_losses(model2, model2.model(input_ids=x, use_cache=False).last_hidden_state, y, normed=True).mean().backward()
    true2 = {k: (ad.B.grad.clone(), ad.A.grad.clone()) for k, ad in ad2.items()}
    for p in ps:
        p.requires_grad_(False)
    _, err2 = dl.dust_gradients(model2, ad2, cap2, x, y, 256, 0.05, torch.Generator().manual_seed(3))
    est2 = dl.lora_gradients(ad2, cap2, err2, dl.mid_errors(model2, cap2, err2), y.numel())
    cos2 = F.cosine_similarity(flat(est2), flat(true2), dim=0).item()
    good = sorted(err2) == [1, 2] and cos2 > 0.5
    print(f"{'PASS' if good else 'FAIL'} top-2-layer adapters: anchors {sorted(err2)}, cos with backprop {cos2:.3f}")
    ok &= good
    # 6) skip-path control variate: unbiased (same mean) and closer to backprop at equal draws
    flat_c = lambda d: torch.cat([torch.cat([b.flatten(), a.flatten()]) for b, a in d.values()])
    for draws in (8, 32):
        cs = {}
        for cv in (False, True):
            vals = []
            for seed in range(4):
                _, e = dl.dust_gradients(model2, ad2, cap2, x, y, draws, 0.05, torch.Generator().manual_seed(10 + seed),
                                         control_variate=cv)
                g = dl.lora_gradients(ad2, cap2, e, dl.mid_errors(model2, cap2, e), y.numel())
                vals.append(F.cosine_similarity(flat_c(g), flat_c(true2), dim=0).item())
            cs[cv] = sum(vals) / len(vals)
        print(f'     draws/layer {draws:3d}: cos without CV {cs[False]:.3f}, with skip CV {cs[True]:.3f}')
    good = cs[True] >= cs[False] - 0.02
    print(f"{'PASS' if good else 'FAIL'} skip control variate does not hurt (and should help)")
    ok &= good
    # 7) bottleneck DUST (frozen B): its A-gradient aligns with backprop's and improves with draws
    model3 = LlamaForCausalLM(cfg).float().eval().requires_grad_(False)
    ad3 = dl.add_lora(model3, rank=4, alpha=8, frozen_b=True)
    cap3 = dl.Capture(model3)
    for a in ad3.values():
        a.A.requires_grad_(True)
    dl.token_losses(model3, model3.model(input_ids=x, use_cache=False).last_hidden_state, y, normed=True).mean().backward()
    trueA = torch.cat([a.A.grad.flatten() for a in ad3.values()])
    for a in ad3.values():
        a.A.requires_grad_(False)
        a.A.grad = None
    last = {}
    for cv in (False, True):
        line = f'     bottleneck {"with" if cv else "no  "} CV:'
        for draws in (8, 64):
            vals = []
            for seed in range(3):
                _, g = dl.bottleneck_gradients(model3, ad3, cap3, x, y, draws, 0.05, torch.Generator().manual_seed(20 + seed),
                                               control_variate=cv)
                vals.append(F.cosine_similarity(torch.cat([g[k][1].flatten() for k in ad3]), trueA, dim=0).item())
            last[(cv, draws)] = sum(vals) / len(vals)
            line += f'  {draws:3d} draws cos {last[(cv, draws)]:.3f}'
        print(line)
    good = last[(True, 64)] > 0.5 and last[(True, 64)] > last[(True, 8)]
    print(f"{'PASS' if good else 'FAIL'} bottleneck DUST aligns with backprop's A-gradient")
    ok &= good
    print('ALL PASSED' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
