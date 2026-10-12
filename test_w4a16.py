"""Correctness tests for the Triton kernels.   Run:  pytest -q test_w4a16.py

On a GPU machine these run the real kernels. Without a GPU they run in Triton's CPU interpreter
(TRITON_INTERPRET=1): that checks indexing / unpacking / masking logic, not speed.
"""
import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")     # must be set before triton is imported

import w4a16                                            # noqa: E402
from w4a16 import (W4A16Linear, W8A16Linear, dequant_int4, dequant_int8_ref, dequant_ref,  # noqa: E402
                    pack_int4, quantize_int4, quantize_int8_per_channel, quantize_pack_int4,
                    unpack_int4, w4a16_matmul, w8a16_matmul)

DEV = "cuda" if HAS_CUDA else "cpu"
REL_TOL = 2e-3      # max|err| / max|ref|; fp16 output rounding alone is ~5e-4

GEMV4 = [dict(BLOCK_N=32, BLOCK_K=64, num_warps=2, num_stages=2), dict(BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2)]
DOT4 = [dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, num_warps=2, num_stages=2), dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2)]
DEQ4 = [dict(BLOCK_KW=4, BLOCK_N=32, num_warps=2, num_stages=2), dict(BLOCK_KW=16, BLOCK_N=64, num_warps=4, num_stages=2)]
GEMV8 = [dict(BLOCK_N=32, BLOCK_K=32, num_warps=2, num_stages=2), dict(BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)]
DOT8 = [dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, num_warps=2, num_stages=2), dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)]


def _rel_err(out, ref):
    return ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def _weights(N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, K, generator=g) * 0.02).half().to(DEV)


def _acts(M, K, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, K, generator=g).half().to(DEV)


# ------------------------------------------------------------------------------ packing (pure PyTorch)
@pytest.mark.parametrize("sym", [False, True])
def test_int4_quantize_pack_roundtrip(sym):
    W = _weights(64, 256)
    qint, s, z = quantize_int4(W, 128, sym)
    assert qint.shape == (64, 256) and int(qint.max()) <= 15 and s.shape == (64, 2)
    qw, s2, z2 = pack_int4(qint, s, z)
    assert qw.dtype == torch.int32 and qw.shape == (32, 64) and s2.shape == (2, 64)
    assert torch.equal(unpack_int4(qw).T.to(torch.uint8), qint)            # pack/unpack is lossless
    Wd = dequant_ref(qw, s2, z2, 128)
    assert Wd.shape == W.shape
    # reconstruction error <= half a level (+ fp16 slack)
    err = (Wd.float() - W.float()).reshape(64, 2, 128).abs().amax(-1)
    assert (err <= 0.51 * s.float() + 2e-3 * W.float().abs().max()).all()


def test_int4_sym_matches_signed_levels():
    W = _weights(32, 128)
    qint, s, z = quantize_int4(W, 128, sym=True)
    assert torch.all(z == 8)
    signed = qint.float() - 8
    assert signed.min() >= -8 and signed.max() <= 7


def test_int8_roundtrip():
    W = _weights(64, 256)
    q, s = quantize_int8_per_channel(W)
    assert q.dtype == torch.int8 and q.shape == (256, 64) and s.shape == (64,)
    Wd = dequant_int8_ref(q, s)
    assert Wd.shape == W.shape
    slack = 2e-3 * W.float().abs().amax(1)
    assert ((Wd.float() - W.float()).abs().amax(1) <= 0.51 * s.float() + slack).all()


# ------------------------------------------------------------------------------ W4A16 kernels
@pytest.mark.parametrize("M", [1, 3])
@pytest.mark.parametrize("N,K", [(96, 256), (100, 384)])               # N=100: not a multiple of BLOCK_N
@pytest.mark.parametrize("cfg", GEMV4)
def test_w4_gemv(M, N, K, cfg):
    W, A = _weights(N, K), _acts(M, K)
    qw, s, z = quantize_pack_int4(W, 128)
    ref = A.float() @ dequant_ref(qw, s, z, 128).float().T
    out = w4a16_matmul(A, qw, s, z, 128, path="gemv", config=cfg)
    assert out.shape == (M, N) and out.dtype == torch.float16
    assert _rel_err(out, ref) < REL_TOL


@pytest.mark.parametrize("M", [1, 3, 16, 40])
@pytest.mark.parametrize("N,K", [(96, 256), (100, 384)])
@pytest.mark.parametrize("cfg", DOT4)
def test_w4_dot(M, N, K, cfg):
    W, A = _weights(N, K), _acts(M, K)
    qw, s, z = quantize_pack_int4(W, 128)
    ref = A.float() @ dequant_ref(qw, s, z, 128).float().T
    out = w4a16_matmul(A, qw, s, z, 128, path="dot", config=cfg)
    assert out.shape == (M, N) and out.dtype == torch.float16
    assert _rel_err(out, ref) < REL_TOL


@pytest.mark.parametrize("cfg", DEQ4)
@pytest.mark.parametrize("N,K", [(96, 256), (100, 384)])
def test_w4_dequant_kernel_is_exact(cfg, N, K):
    W = _weights(N, K)
    qw, s, z = quantize_pack_int4(W, 128)
    Wkn = dequant_int4(qw, s, z, 128, config=cfg)
    assert Wkn.shape == (K, N)
    assert torch.equal(Wkn, dequant_ref(qw, s, z, 128).T)                  # same fp32 math -> bit-identical


def test_w4_dequant_path():
    W, A = _weights(100, 384), _acts(70, 384)
    qw, s, z = quantize_pack_int4(W, 128)
    ref = A.float() @ dequant_ref(qw, s, z, 128).float().T
    out = w4a16_matmul(A, qw, s, z, 128, path="dequant", config=DEQ4[0])
    assert _rel_err(out, ref) < REL_TOL


@pytest.mark.parametrize("path,cfg", [("gemv", GEMV4[0]), ("dot", DOT4[0])])
def test_w4_every_nibble_and_group(path, cfg):
    """One-hot activations pick single weight columns: catches nibble/group indexing bugs random data can hide."""
    N, K = 64, 256
    W = _weights(N, K)
    qw, s, z = quantize_pack_int4(W, 128)
    Wd = dequant_ref(qw, s, z, 128)
    cols = torch.arange(16) * 16 + torch.arange(16) % 8                     # all 8 nibble positions, both groups
    for i, c in enumerate(cols.tolist()):
        A = torch.zeros(1, K, dtype=torch.float16, device=DEV)
        A[0, c] = 1
        out = w4a16_matmul(A, qw, s, z, 128, path=path, config=cfg)
        assert torch.allclose(out[0].float(), Wd[:, c].float(), atol=2e-3, rtol=2e-3), (path, i, c)


@pytest.mark.parametrize("path,cfg", [("gemv", GEMV4[0]), ("dot", DOT4[0])])
def test_w4_group64_and_sym(path, cfg):
    W, A = _weights(64, 256), _acts(3, 256)
    for sym in (False, True):
        qw, s, z = quantize_pack_int4(W, 64, sym)
        ref = A.float() @ dequant_ref(qw, s, z, 64).float().T
        assert _rel_err(w4a16_matmul(A, qw, s, z, 64, path=path, config=cfg), ref) < REL_TOL


def test_w4_auto_dispatch(monkeypatch):
    monkeypatch.setattr(w4a16, "GEMV_MAX_M", 1)
    monkeypatch.setattr(w4a16, "DOT_MAX_M", 16)
    assert [w4a16._pick_path(m) for m in (1, 2, 16, 17)] == ["gemv", "dot", "dot", "dequant"]


# ------------------------------------------------------------------------------ W8A16 kernels
@pytest.mark.parametrize("path,cfgs", [("gemv", GEMV8), ("dot", DOT8)])
@pytest.mark.parametrize("M", [1, 3, 40])
@pytest.mark.parametrize("N,K", [(96, 256), (100, 200)])               # K=200: exercises the k mask
def test_w8(path, cfgs, M, N, K):
    if path == "gemv" and M > 3:
        pytest.skip("gemv is for tiny M")
    W, A = _weights(N, K), _acts(M, K)
    q, s = quantize_int8_per_channel(W)
    ref = A.float() @ dequant_int8_ref(q, s).float().T
    for cfg in cfgs:
        out = w8a16_matmul(A, q, s, path=path, config=cfg)
        assert out.shape == (M, N) and out.dtype == torch.float16
        assert _rel_err(out, ref) < REL_TOL


def test_w8_dequant_path():
    W, A = _weights(100, 200), _acts(70, 200)
    q, s = quantize_int8_per_channel(W)
    ref = A.float() @ dequant_int8_ref(q, s).float().T
    assert _rel_err(w8a16_matmul(A, q, s, path="dequant"), ref) < REL_TOL


# ------------------------------------------------------------------------------ real Qwen2.5-1.5B MLP shapes, decode
@pytest.mark.parametrize("N,K", [(8960, 1536), (1536, 8960)])
def test_real_mlp_shapes_decode(N, K):
    W, A = _weights(N, K), _acts(1, K)
    qw, s, z = quantize_pack_int4(W, 128)
    ref4 = A.float() @ dequant_ref(qw, s, z, 128).float().T
    assert _rel_err(w4a16_matmul(A, qw, s, z, 128, path="gemv", config=dict(BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2)), ref4) < REL_TOL
    q8, s8 = quantize_int8_per_channel(W)
    ref8 = A.float() @ dequant_int8_ref(q8, s8).float().T
    assert _rel_err(w8a16_matmul(A, q8, s8, path="gemv", config=dict(BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2)), ref8) < REL_TOL


# ------------------------------------------------------------------------------ nn.Module wrappers
def test_linear_wrappers_match_reference():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 96, bias=True).half().to(DEV)
    lin.weight.data = _weights(96, 256)
    x = _acts(6, 256).reshape(2, 3, 256)
    m4 = W4A16Linear(lin, 128)
    w4 = dequant_ref(m4.qweight, m4.scales, m4.zeros, 128)
    ref4 = x.float() @ w4.float().T + lin.bias.float()
    assert m4(x).shape == (2, 3, 96)
    assert _rel_err(m4(x), ref4) < REL_TOL
    m8 = W8A16Linear(lin)
    ref8 = x.float() @ dequant_int8_ref(m8.q, m8.scales).float().T + lin.bias.float()
    assert _rel_err(m8(x), ref8) < REL_TOL


def test_w4_linear_with_awq_input_scale():
    """y = (x / s) @ Q(W * s)^T  must equal  x @ (Q(W * s) / s)^T."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 64, bias=False).half().to(DEV)
    lin.weight.data = _weights(64, 256)
    s_in = (torch.rand(256, device=DEV) + 0.5)
    scaled = torch.nn.Linear(256, 64, bias=False).half().to(DEV)
    scaled.weight.data = (lin.weight.float() * s_in).half()
    packed = quantize_pack_int4(scaled.weight.data, 128)
    m = W4A16Linear(lin, 128, packed=packed, in_scale=s_in)
    x = _acts(3, 256)
    w_eff = dequant_ref(*packed, 128).float() / s_in
    assert _rel_err(m(x), x.float() @ w_eff.T) < 3e-3


# ------------------------------------------------------------------------------ fused bias, all paths
@pytest.mark.parametrize("path,cfg4,cfg8", [("gemv", GEMV4[0], GEMV8[0]), ("dot", DOT4[0], DOT8[0]), ("dequant", DEQ4[0], None)])
def test_fused_bias(path, cfg4, cfg8):
    M = 1 if path == "gemv" else 20
    W, A = _weights(100, 256), _acts(M, 256)
    b = (torch.randn(100) * 0.1).half().to(DEV)
    qw, s, z = quantize_pack_int4(W, 128)
    ref4 = A.float() @ dequant_ref(qw, s, z, 128).float().T + b.float()
    assert _rel_err(w4a16_matmul(A, qw, s, z, 128, path=path, config=cfg4, bias=b), ref4) < REL_TOL
    q8, s8 = quantize_int8_per_channel(W)
    ref8 = A.float() @ dequant_int8_ref(q8, s8).float().T + b.float()
    assert _rel_err(w8a16_matmul(A, q8, s8, path=path, config=cfg8, bias=b), ref8) < REL_TOL


# ------------------------------------------------------------------------------ fast launch cache
def test_fast_launch_uses_cached_config(monkeypatch):
    """A cached config must be used (and give correct results) without going through autotune/defaults."""
    W, A = _weights(100, 256), _acts(1, 256)
    qw, s, z = quantize_pack_int4(W, 128)
    ref = A.float() @ dequant_ref(qw, s, z, 128).float().T
    monkeypatch.setattr(w4a16, "_BEST", {("w4_gemv", 128, 1, 100, 256): dict(BLOCK_N=16, BLOCK_K=32, num_warps=1, num_stages=2)})
    def no_default(*_a, **_k):
        raise RuntimeError("default config used")
    monkeypatch.setattr(w4a16, "AUTOTUNE", False)
    monkeypatch.setattr(w4a16, "_default_cfg", no_default)                       # any non-cached launch now fails
    assert _rel_err(w4a16_matmul(A, qw, s, z, 128, path="gemv"), ref) < REL_TOL
    monkeypatch.setattr(w4a16, "_BEST", {})
    with pytest.raises(RuntimeError, match="default config used"):
        w4a16_matmul(A, qw, s, z, 128, path="gemv")                              # proves the cache was what was used


# ------------------------------------------------------------------------------ autotuned paths (GPU only)
@pytest.mark.skipif(not HAS_CUDA, reason="autotune benchmarks real kernels; needs a GPU")
@pytest.mark.parametrize("M", [1, 16, 512])
def test_autotuned_paths(M):
    N, K = 8960, 1536
    W, A = _weights(N, K), _acts(M, K)
    qw, s, z = quantize_pack_int4(W, 128)
    ref4 = A.float() @ dequant_ref(qw, s, z, 128).float().T
    for path in ("gemv", "dot", "dequant"):
        if path == "gemv" and M > 16:
            continue
        assert _rel_err(w4a16_matmul(A, qw, s, z, 128, path=path), ref4) < REL_TOL, path
    q8, s8 = quantize_int8_per_channel(W)
    assert _rel_err(w8a16_matmul(A, q8, s8), A.float() @ dequant_int8_ref(q8, s8).float().T) < REL_TOL
    if M <= 16:                                                  # the tuned config is now cached for direct launches
        assert any(k[0] == "w4_gemv" and k[-2:] == (N, K) for k in w4a16._BEST), "fast-launch cache not populated"
        assert _rel_err(w4a16_matmul(A, qw, s, z, 128, path="gemv"), ref4) < REL_TOL
