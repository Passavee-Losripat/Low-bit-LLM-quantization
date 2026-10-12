"""Post-training weight quantization, implemented from scratch: RTN, GPTQ, AWQ (int4, group-wise).

Every method does two things for each nn.Linear inside the decoder layers:
  1. writes fake-quantized fp16 weights into the model (so accuracy can be evaluated directly), and
  2. returns the integer form, packed for the Triton kernel:  results[name] = {"packed": (qweight, scales, zeros),
     "in_scale": None | [K] tensor}.  Packing reproduces the fake-quant weights exactly (same fp16 scales).
so the kernel-backed model can be checked against the fake-quant model (perplexity parity).

Methods
-------
RTN   round each weight to the nearest level of its group (baseline).
GPTQ  (Frantar et al., 2023) quantizes one column at a time and pushes each column's rounding error onto the
      not-yet-quantized columns, weighted by the inverse Hessian H = 2 X^T X of the layer's calibration inputs.
      Minimizes ||X W^T - X Q^T||^2 layer by layer instead of ||W - Q||^2.
AWQ   (Lin et al., 2024) scales up weight input-channels that see large activations before quantizing
      (s = mean|x|^alpha, alpha grid-searched per layer group) and divides the activations by s at runtime:
      y = (x / s) @ Q(W * s)^T.  Salient channels get finer effective resolution. Here s is applied as an explicit
      input scale (production code folds it into the preceding RMSNorm / linear; mathematically identical).

Calibration is sequential: when layer i is calibrated, layers < i are already quantized, so each layer
compensates for the error that reaches it (what happens at inference).
"""
import math

import torch
import torch.nn as nn

from w4a16 import W4A16Linear, W8A16Linear, _SCALE_MIN, pack_int4, quantize_int4

# =============================================================================================
# helpers
# =============================================================================================


def decoder_layers(model):
    return model.model.layers


def find_linears(module, skip=()):
    return {n: m for n, m in module.named_modules()
            if isinstance(m, nn.Linear) and not any(s in n for s in skip)}


def _group_params(wg: torch.Tensor, sym: bool):
    """wg [..., group] -> (scale, zero) fp32, scale rounded to fp16 exactly like w4a16.quantize_int4."""
    if sym:
        scale = (wg.abs().amax(-1) / 7).clamp(min=_SCALE_MIN).half().float()
        zero = torch.full_like(scale, 8.0)
    else:
        mn, mx = wg.amin(-1), wg.amax(-1)
        scale = ((mx - mn) / 15).clamp(min=_SCALE_MIN).half().float()
        zero = torch.round(-mn / scale).clamp(0, 15)
    return scale, zero


@torch.no_grad()
def fake_quant_int4(w: torch.Tensor, group_size: int = 128, sym: bool = False) -> torch.Tensor:
    """Dequantized int4 weights (fp32), bit-consistent with what the kernel computes."""
    qint, s, z = quantize_int4(w, group_size, sym)
    N, K = w.shape
    gs = K if group_size in (-1, 0) else group_size
    return ((qint.float().reshape(N, K // gs, gs) - z.float()[..., None]) * s.float()[..., None]).reshape(N, K)


class _StopForward(Exception):
    pass


@torch.no_grad()
def forward_until(model, calib_ids: torch.Tensor, layer_idx: int, batch_size: int = 8):
    """Run calibration batches through the model and stop right after decoder layer `layer_idx`
    (later layers are never computed: about half the cost of full forward passes)."""
    def stop(_m, _i, _o):
        raise _StopForward

    handle = decoder_layers(model)[layer_idx].register_forward_hook(stop)
    dev = next(model.parameters()).device
    try:
        for b in range(0, len(calib_ids), batch_size):
            try:
                model(calib_ids[b:b + batch_size].to(dev), use_cache=False)
            except _StopForward:
                pass
    finally:
        handle.remove()


def _record(results, prefix, name, qint, scales, zeros, in_scale=None):
    results[f"{prefix}.{name}"] = {"packed": tuple(t.cpu() for t in pack_int4(qint, scales, zeros)),
                                   "in_scale": None if in_scale is None else in_scale.cpu()}


# =============================================================================================
# RTN (kernel-consistent version; the notebook's section 1 has the didactic version)
# =============================================================================================
@torch.no_grad()
def rtn_quantize_model(model, group_size=128, sym=False, skip=("lm_head",)):
    results = {}
    for i, layer in enumerate(decoder_layers(model)):
        for name, m in find_linears(layer, skip).items():
            qint, s, z = quantize_int4(m.weight.data, group_size, sym)
            m.weight.data.copy_(fake_quant_int4(m.weight.data, group_size, sym).to(m.weight.dtype))
            _record(results, f"model.layers.{i}", name, qint, s, z)
    return results


# =============================================================================================
# GPTQ
# =============================================================================================
@torch.no_grad()
def gptq_quantize_weight(W: torch.Tensor, H: torch.Tensor, group_size: int = 128, sym: bool = False,
                         percdamp: float = 0.01, blocksize: int = 128):
    """GPTQ for one weight matrix W [N, K] given Hessian H [K, K] (= 2/n X^T X of its inputs).
    Returns (Wq fp16 [N,K], qint uint8 [N,K], scales fp16 [N,G], zeros fp16 [N,G])."""
    W = W.clone().float()
    N, K = W.shape
    gs = K if group_size in (-1, 0) else group_size
    assert K % gs == 0 and (gs % blocksize == 0 or blocksize % gs == 0), "group/block sizes must nest"
    H = H.clone().float()

    dead = torch.diag(H) == 0                       # inputs that are always zero carry no information
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))     # dampening keeps the Cholesky numerically stable
    idx = torch.arange(K, device=W.device)
    H[idx, idx] += damp
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper Cholesky factor of H^-1: row i holds the update weights

    Q = torch.zeros_like(W)
    qint = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(N, K // gs, device=W.device)
    zeros = torch.zeros(N, K // gs, device=W.device)
    s = z = None

    for i1 in range(0, K, blocksize):               # "lazy batch" updates: work on 128-column blocks
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % gs == 0:                          # new group: scale/zero from the *error-updated* weights
                grp = W1[:, i:i + gs] if gs <= blocksize else W[:, c:c + gs]
                s, z = _group_params(grp, sym)
                scales[:, c // gs], zeros[:, c // gs] = s, z
            w = W1[:, i]
            d = Hinv1[i, i]
            qi = torch.clamp(torch.round(w / s) + z, 0, 15)
            q = (qi - z) * s
            Q1[:, i] = q
            qint[:, c] = qi.to(torch.uint8)
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)   # compensate inside the block
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]          # compensate all later blocks at once

    return Q.half(), qint, scales.half(), zeros.half()


@torch.no_grad()
def gptq_quantize_model(model, calib_ids, group_size=128, sym=False, percdamp=0.01, batch_size=8,
                        skip=("lm_head",), log=print):
    results = {}
    layers = decoder_layers(model)
    for i, layer in enumerate(layers):
        lins = find_linears(layer, skip)
        H, count = {}, {}

        def hook_for(name):
            def hook(_mod, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                t = x.shape[0]
                if name not in H:
                    H[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                    count[name] = 0
                H[name].mul_(count[name] / (count[name] + t))          # running average of 2/n X^T X
                count[name] += t
                x = x * math.sqrt(2 / count[name])
                H[name].addmm_(x.T, x)
            return hook

        handles = [m.register_forward_hook(hook_for(n)) for n, m in lins.items()]
        forward_until(model, calib_ids, i, batch_size)
        for h in handles:
            h.remove()
        for name, m in lins.items():
            Wq, qint, s, z = gptq_quantize_weight(m.weight.data, H.pop(name), group_size, sym, percdamp)
            m.weight.data.copy_(Wq.to(m.weight.dtype))
            _record(results, f"model.layers.{i}", name, qint, s, z)
        if log:
            log(f"GPTQ layer {i + 1}/{len(layers)} done")
    return results


# =============================================================================================
# AWQ
# =============================================================================================
@torch.no_grad()
def awq_search_scale(X: torch.Tensor, W: torch.Tensor, group_size=128, sym=False, n_grid=20):
    """X [T, K] calibration inputs, W [N, K] (all linears sharing this input, concatenated).
    Returns (best input scale s [K], best alpha, error table)."""
    Xf = X.float()
    x_mag = Xf.abs().mean(0)
    ref = Xf @ W.float().T
    best_err, best_s, best_alpha, errs = float("inf"), None, None, []
    for r in range(n_grid + 1):
        alpha = r / n_grid
        s = x_mag.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        Wq = fake_quant_int4((W.float() * s).half(), group_size, sym) / s
        err = ((Xf @ Wq.T) - ref).pow(2).mean().item()
        errs.append((alpha, err))
        if err < best_err:
            best_err, best_s, best_alpha = err, s, alpha
    return best_s, best_alpha, errs


def _input_groups(lins, first_inputs):
    """Group linears that received the very same input tensor object (q/k/v; gate/up). The first batch's
    input tensors are held alive in `first_inputs`, so identity cannot be confused by memory reuse."""
    groups = []
    for name in lins:
        for g in groups:
            if first_inputs[g[0]] is first_inputs[name]:
                g.append(name)
                break
        else:
            groups.append([name])
    return groups


@torch.no_grad()
def awq_quantize_model(model, calib_ids, group_size=128, sym=False, n_grid=20, max_tokens=4096,
                       batch_size=8, skip=("lm_head",), log=print, seed=0):
    results = {}
    layers = decoder_layers(model)
    n_batches = math.ceil(len(calib_ids) / batch_size)
    per_batch = max(1, max_tokens // n_batches)
    gen = torch.Generator().manual_seed(seed)
    for i, layer in enumerate(layers):
        lins = find_linears(layer, skip)
        X, first_inp = {}, {}

        def hook_for(name):
            def hook(_mod, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                if name not in first_inp:
                    first_inp[name] = inp[0]                       # keep alive until grouping is done
                rows = torch.randperm(x.shape[0], generator=gen)[:per_batch].to(x.device)
                X.setdefault(name, []).append(x[rows].half())
            return hook

        handles = [m.register_forward_hook(hook_for(n)) for n, m in lins.items()]
        forward_until(model, calib_ids, i, batch_size)
        for h in handles:
            h.remove()

        groups = _input_groups(lins, first_inp)
        first_inp.clear()
        for names in groups:
            Xg = torch.cat(X[names[0]], 0)
            Wcat = torch.cat([lins[n].weight.data for n in names], 0)
            s, alpha, _ = awq_search_scale(Xg, Wcat, group_size, sym, n_grid)
            for n in names:
                m = lins[n]
                scaled = (m.weight.data.float() * s).half()
                qint, sc, zr = quantize_int4(scaled, group_size, sym)
                m.weight.data.copy_((fake_quant_int4(scaled, group_size, sym) / s).to(m.weight.dtype))
                _record(results, f"model.layers.{i}", n, qint, sc, zr, in_scale=s.half())
            if log:
                log(f"AWQ layer {i + 1}/{len(layers)} {'+'.join(names)}: alpha={alpha:.2f}")
        del X
    return results


# =============================================================================================
# swapping nn.Linear <-> kernel-backed modules
# =============================================================================================
def _set_module(model, name, new):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent), child, new)


def swap_w4(model, results, group_size=128):
    """Replace every linear in `results` with a W4A16Linear holding those exact packed weights.
    Returns the original modules so unswap() can restore them."""
    originals = {}
    for name, r in results.items():
        lin = model.get_submodule(name)
        originals[name] = lin
        _set_module(model, name, W4A16Linear(lin, group_size, packed=r["packed"], in_scale=r["in_scale"]))
    return originals


def swap_w8(model, skip=("lm_head",)):
    originals = {}
    for i, layer in enumerate(decoder_layers(model)):
        for n, lin in find_linears(layer, skip).items():
            name = f"model.layers.{i}.{n}"
            originals[name] = lin
            _set_module(model, name, W8A16Linear(lin))
    return originals


def unswap(model, originals):
    for name, lin in originals.items():
        _set_module(model, name, lin)


@torch.no_grad()
def fake_quant_int8_per_channel_model(model, skip=("lm_head",)):
    """Fake-quant twin of swap_w8 (for perplexity parity checks)."""
    from w4a16 import dequant_int8_ref, quantize_int8_per_channel
    for layer in decoder_layers(model):
        for _, m in find_linears(layer, skip).items():
            m.weight.data.copy_(dequant_int8_ref(*quantize_int8_per_channel(m.weight.data)).to(m.weight.dtype))
