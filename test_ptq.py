"""Tests for ptq.py (GPTQ, AWQ, RTN, model swapping).   Run:  pytest -q test_ptq.py

Synthetic tests check the algorithms do what they claim (GPTQ / AWQ beat RTN on the objective they optimize).
The tiny-model tests build a 2-layer randomly initialised Qwen2 from a config (no download) and run the full
sequential calibration + kernel swap; without a GPU the kernels run in the Triton CPU interpreter.
"""
import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

import ptq                                                       # noqa: E402
from w4a16 import W4A16Linear, W8A16Linear, dequant_ref          # noqa: E402

DEV = "cuda" if HAS_CUDA else "cpu"


def _correlated_inputs(T, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(K, K, generator=g) / K ** 0.5 + torch.eye(K)
    return torch.randn(T, K, generator=g) @ mix                  # correlated features, like real activations


def _hessian(X):
    return 2 * X.T @ X / X.shape[0]


def _out_err(X, W, Wq):
    return ((X @ W.T - X @ Wq.T) ** 2).mean().item()


# ------------------------------------------------------------------------------ GPTQ
@pytest.mark.parametrize("group_size", [128, -1])
def test_gptq_beats_rtn_on_output_error(group_size):
    K, N = 256, 64
    X = _correlated_inputs(2048, K)
    W = torch.randn(N, K, generator=torch.Generator().manual_seed(1)) * 0.02
    Wq, qint, s, z = ptq.gptq_quantize_weight(W.half(), _hessian(X), group_size)
    rtn = ptq.fake_quant_int4(W.half(), group_size)
    e_gptq, e_rtn = _out_err(X, W, Wq.float()), _out_err(X, W, rtn)
    assert e_gptq < 0.8 * e_rtn, (e_gptq, e_rtn)


def test_gptq_outputs_are_kernel_consistent():
    K, N = 256, 32
    X = _correlated_inputs(1024, K)
    W = (torch.randn(N, K) * 0.02).half()
    Wq, qint, s, z = ptq.gptq_quantize_weight(W, _hessian(X), 128)
    assert int(qint.max()) <= 15 and s.shape == (N, 2) and z.shape == (N, 2)
    packed = ptq.pack_int4(qint, s, z)
    assert torch.equal(dequant_ref(*packed, 128), Wq)            # packing reproduces GPTQ's weights bit-exactly


def test_gptq_handles_dead_inputs():
    K, N = 128, 16
    X = _correlated_inputs(512, K)
    X[:, :5] = 0                                                 # some input channels never fire
    Wq, *_ = ptq.gptq_quantize_weight((torch.randn(N, K) * 0.02).half(), _hessian(X), 128)
    assert torch.isfinite(Wq.float()).all()


# ------------------------------------------------------------------------------ AWQ
def test_awq_beats_rtn_with_outlier_channels():
    K, N = 256, 64
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2048, K, generator=g)
    X[:, :8] *= 30                                               # a few salient input channels with huge activations
    W = torch.randn(N, K, generator=g) * 0.02
    s, alpha, errs = ptq.awq_search_scale(X.half(), W.half(), 128)
    rtn_err = dict(errs)[0.0]                                    # alpha = 0  <=>  s = 1  <=>  plain RTN
    best_err = min(e for _, e in errs)
    assert alpha > 0 and best_err < 0.8 * rtn_err, (alpha, best_err, rtn_err)


def test_input_groups_by_identity():
    a, b = torch.zeros(2), torch.zeros(2)
    lins = {"q": 0, "k": 0, "v": 0, "o": 0, "gate": 0, "up": 0, "down": 0}
    first = {"q": a, "k": a, "v": a, "o": b, "gate": torch.zeros(2), "up": None, "down": torch.zeros(2)}
    first["up"] = first["gate"]
    assert ptq._input_groups(lins, first) == [["q", "k", "v"], ["o"], ["gate", "up"], ["down"]]


# ------------------------------------------------------------------------------ tiny Qwen2, end to end
def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(DEV).half().eval()
    for p in model.parameters():                                 # larger random weights -> non-trivial quantization
        p.data.normal_(0, 0.05)
    return model


@pytest.fixture(scope="module")
def calib():
    return torch.randint(0, 512, (8, 32), generator=torch.Generator().manual_seed(0))


@torch.no_grad()
def _logits(model, ids):
    return model(ids.to(DEV), use_cache=False).logits.float()


@pytest.mark.parametrize("method", ["rtn", "gptq", "awq"])
def test_tiny_model_quantize_then_swap_to_kernel(method, calib):
    model = _tiny_model()
    if method == "rtn":
        res = ptq.rtn_quantize_model(model, 128)
    elif method == "gptq":
        res = ptq.gptq_quantize_model(model, calib, 128, batch_size=4, log=None)
    else:
        res = ptq.awq_quantize_model(model, calib, 128, n_grid=5, max_tokens=64, batch_size=4, log=None)
    assert len(res) == 2 * 7                                     # 2 layers x (q,k,v,o,gate,up,down)
    if method == "awq":
        assert all(r["in_scale"] is not None for r in res.values())
    ids = calib[:2, :16]
    fake = _logits(model, ids)                                   # fake-quant model (weights written in place)
    originals = ptq.swap_w4(model, res, 128)
    assert all(isinstance(model.get_submodule(n), W4A16Linear) for n in res)
    kern = _logits(model, ids)                                   # same weights, Triton kernels (M=32 -> "dot" path)
    rel = ((kern - fake).abs().max() / fake.abs().max()).item()
    assert rel < 2e-2, rel
    dec = _logits(model, ids[:1, :1])                            # M=1 -> "gemv" path
    assert torch.isfinite(dec).all()
    ptq.unswap(model, originals)
    assert all(isinstance(model.get_submodule(n), torch.nn.Linear) for n in res)
    assert torch.allclose(_logits(model, ids), fake)             # unswap restores the fake-quant model exactly


def test_tiny_model_w8_swap_matches_fake_quant(calib):
    model = _tiny_model()
    ids = calib[:2, :16]
    originals = ptq.swap_w8(model)
    assert all(isinstance(model.get_submodule(n), W8A16Linear) for n in originals)
    kern = _logits(model, ids)
    ptq.unswap(model, originals)
    ptq.fake_quant_int8_per_channel_model(model)
    fake = _logits(model, ids)
    assert ((kern - fake).abs().max() / fake.abs().max()).item() < 2e-2


def test_gptq_beats_rtn_on_tiny_model(calib):
    """Layer-output error on held-out tokens: GPTQ should not be worse than RTN."""
    ref = _tiny_model()
    ids = torch.randint(0, 512, (4, 32), generator=torch.Generator().manual_seed(7))
    target = _logits(ref, ids)
    errs = {}
    for method in ("rtn", "gptq"):
        m = _tiny_model()
        if method == "rtn":
            ptq.rtn_quantize_model(m, 128)
        else:
            ptq.gptq_quantize_model(m, calib, 128, batch_size=4, log=None)
        errs[method] = ((_logits(m, ids) - target) ** 2).mean().item()
    assert errs["gptq"] <= errs["rtn"] * 1.05, errs
