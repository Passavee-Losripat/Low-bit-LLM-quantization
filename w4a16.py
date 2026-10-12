"""Weight-only quantized linear layers in Triton: W4A16 (group-wise int4) and W8A16 (per-channel int8).

Both compute  C = A @ W^T   with  A : [M, K] fp16 activations,  W : [N, K] (an nn.Linear weight),
and dequantize W *inside* the kernel, so the fp16 weights never exist in GPU memory.

v2 design (why v1 was ~10x slower than fp16 at decode, and what changed)
------------------------------------------------------------------------
v1 stored packed weights as [N, K//8]. A [BLOCK_K, BLOCK_N] weight tile then needs, for neighbouring
output channels n, words that are a whole row (K/8 words) apart in memory, and 8 neighbouring k share
one word. Neither tile axis is contiguous, so every load was a scattered 4-byte access: measured ~6 GB/s
on a T4 vs ~213 GB/s for fp16 cuBLAS.
v2 stores weights K-major, like the GPTQ kernels:
    qweight [K//8, N]  int32     word (kw, n) holds k = 8*kw .. 8*kw+7 of output channel n (k = 8*kw + j in bits 4j..4j+3)
    scales  [K//G, N]  fp16      one per (group, output channel)
    zeros   [K//G, N]  fp16      integer-valued zero-points; dequant: w = (q - zero) * scale
so neighbouring output channels are neighbouring words: loads along N are coalesced.

Three execution paths, chosen by M (number of tokens) -- see _pick_path and the thresholds below:
    "gemv"    M tiny (decode). No tensor cores: each word is loaded ONCE, its 8 nibbles are multiplied with
              8 strided slices of the activation row. Memory-bound, so this is about bytes/second.
    "dot"     M moderate. Tiled tl.dot (tensor cores) on dequantized tiles.
    "dequant" M large (prefill). Compute-bound: dequantize W once to fp16 with a Triton kernel, then call
              cuBLAS. Trying to beat cuBLAS's matmul at large M is not the point of a weight-only kernel.

Algebraic trick used by gemv and dot (scale/zero are constant inside one K-tile because BLOCK_K | GROUP):
    sum_k a_k * (q_k - z) * s  =  s * (sum_k a_k * q_k)  -  s * z * (sum_k a_k)
so the per-element work is just unpack + multiply-add; scale and zero are applied once per tile.

Public API (the notebooks and tests rely on these):
    quantize_int4(W, group_size, sym)        -> qint [N,K] uint8 (0..15), scales [N,G] fp16, zeros [N,G] fp16
    pack_int4(qint, scales, zeros)           -> qweight [K//8,N] int32, scales [G,N], zeros [G,N]   (kernel layout)
    quantize_pack_int4(W, group_size, sym)   -> pack_int4(quantize_int4(...))
    dequant_ref(qweight, scales, zeros, gs)  -> [N,K] fp16   pure-PyTorch reference
    dequant_int4(qweight, scales, zeros, gs) -> [K,N] fp16   Triton dequant kernel (used by the "dequant" path)
    w4a16_matmul(A, qweight, scales, zeros, group_size, path=None, config=None) -> [M,N] fp16

    quantize_int8_per_channel(W)             -> q [K,N] int8 (K-major), scales [N] fp16
    dequant_int8_ref(q, scales)              -> [N,K] fp16
    w8a16_matmul(A, q, scales, path=None, config=None) -> [M,N] fp16
"""
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # lets the notebook import this on a box without triton
    triton = tl = None

# ---------------------------------------------------------------------------------------------
# Path thresholds. These are starting guesses; the notebook's benchmark measures the crossovers on
# your GPU and overwrites them (w4a16.GEMV_MAX_M = ..., w4a16.DOT_MAX_M = ...).
# ---------------------------------------------------------------------------------------------
GEMV_MAX_M = 1      # M <= GEMV_MAX_M          -> "gemv"
DOT_MAX_M = 64      # GEMV_MAX_M < M <= this   -> "dot";   M > DOT_MAX_M -> "dequant" (+ cuBLAS)

_SCALE_MIN = 1e-5   # keeps the fp16 scale non-zero for (near-)constant groups


def _pick_path(M: int) -> str:
    if M <= GEMV_MAX_M:
        return "gemv"
    if M <= DOT_MAX_M:
        return "dot"
    return "dequant"


# =============================================================================================
# 1. Quantization / packing (plain PyTorch) and reference dequantizers
# =============================================================================================
@torch.no_grad()
def quantize_int4(w: torch.Tensor, group_size: int = 128, sym: bool = False):
    """Group-wise round-to-nearest int4 (groups of `group_size` along K).
    asym: q in 0..15, w ~ (q - zero) * scale with zero in 0..15
    sym : same storage with zero fixed at 8, i.e. q - 8 in -8..7 and scale = absmax / 7.
    The scale is rounded to fp16 BEFORE quantizing, so the stored scale reproduces these levels exactly."""
    N, K = w.shape
    gs = K if group_size in (-1, 0) else group_size
    assert K % gs == 0, f"K={K} not divisible by group_size={gs}"
    wg = w.float().reshape(N, K // gs, gs)
    if sym:
        scale = (wg.abs().amax(-1) / 7).clamp(min=_SCALE_MIN).half().float()
        zero = torch.full_like(scale, 8.0)
    else:
        mn, mx = wg.amin(-1), wg.amax(-1)
        scale = ((mx - mn) / 15).clamp(min=_SCALE_MIN).half().float()
        zero = torch.round(-mn / scale).clamp(0, 15)
    q = torch.round(wg / scale[..., None] + zero[..., None]).clamp(0, 15)
    return q.reshape(N, K).to(torch.uint8), scale.half(), zero.half()


@torch.no_grad()
def pack_int4(qint: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    """[N,K] ints 0..15 + [N,G] scales/zeros  ->  kernel layout: qweight [K//8, N] int32, scales/zeros [G, N]."""
    N, K = qint.shape
    assert K % 8 == 0
    q = qint.to(torch.int32).T.reshape(K // 8, 8, N)            # [KW, 8, N]: q[kw, j, n] = qint[n, 8*kw + j]
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=qint.device)
    for j in range(8):
        packed |= q[:, j, :] << (4 * j)                          # int32 wraps for j = 7: fine, we mask on unpack
    return packed.contiguous(), scales.T.contiguous(), zeros.T.contiguous()


def quantize_pack_int4(w: torch.Tensor, group_size: int = 128, sym: bool = False):
    return pack_int4(*quantize_int4(w, group_size, sym))


@torch.no_grad()
def unpack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """[K//8, N] int32 -> [K, N] int32 values 0..15."""
    KW, N = qweight.shape
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
    return ((qweight.unsqueeze(1) >> shifts) & 0xF).reshape(KW * 8, N)


@torch.no_grad()
def dequant_ref(qweight, scales, zeros, group_size: int = 128):
    """Reference dequantization -> [N, K] fp16 (same math as the kernels: fp32 (q - z) * s, cast to fp16)."""
    q = unpack_int4(qweight).float()                             # [K, N]
    K = q.shape[0]
    gs = K if group_size in (-1, 0) else group_size
    s = scales.float().repeat_interleave(gs, dim=0)
    z = zeros.float().repeat_interleave(gs, dim=0)
    return ((q - z) * s).half().T.contiguous()


@torch.no_grad()
def quantize_int8_per_channel(w: torch.Tensor):
    """Symmetric per-output-channel int8. Returns q as [K, N] (K-major, coalesced loads along N) and scales [N]."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / 127).clamp(min=_SCALE_MIN).half().float()
    q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.T.contiguous(), scale.half()


@torch.no_grad()
def dequant_int8_ref(q, scales):
    return (q.float() * scales.float()[None, :]).half().T.contiguous()


# =============================================================================================
# 2. W4A16 kernels
# =============================================================================================
@triton.jit
def _w4_gemv_kernel(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr, b_ptr,
    M, N, K, M_BUCKET,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    GROUP: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """One program = one row m of A x BLOCK_N output channels. Each packed word is loaded exactly once."""
    tl.static_assert(GROUP % BLOCK_K == 0, "BLOCK_K must divide the group size")
    tl.static_assert(BLOCK_K % 8 == 0, "BLOCK_K must be a multiple of 8")
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_kw = tl.arange(0, BLOCK_K // 8)                       # word rows inside one K tile
    offs_k = tl.arange(0, BLOCK_K)
    a_row = a_ptr + pid_m * stride_am

    acc2d = tl.zeros((BLOCK_K // 8, BLOCK_N), dtype=tl.float32)  # partial sums kept 2-D: one reduction at the end
    corr = tl.zeros((BLOCK_N,), dtype=tl.float32)                # sum over tiles of s * z * sum(a)

    for k0 in range(0, K, BLOCK_K):
        # [BLOCK_K/8, BLOCK_N] words; consecutive n are consecutive addresses -> coalesced
        words = tl.load(qw_ptr + (k0 // 8 + offs_kw)[:, None] * stride_qk + offs_n[None, :] * stride_qn,
                        mask=n_mask[None, :], other=0)
        part = tl.zeros((BLOCK_K // 8, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(8):                            # unrolled: nibble j of every word = k = 8*kw + j
            a_j = tl.load(a_row + (k0 + offs_kw * 8 + j) * stride_ak).to(tl.float32)   # [BLOCK_K/8]
            q_j = ((words >> (4 * j)) & 0xF).to(tl.float32)                            # [BLOCK_K/8, BLOCK_N]
            part += a_j[:, None] * q_j
        g = k0 // GROUP
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0).to(tl.float32)
        a_sum = tl.sum(tl.load(a_row + (k0 + offs_k) * stride_ak).to(tl.float32), axis=0)
        acc2d += part * s[None, :]
        corr += s * z * a_sum

    acc = tl.sum(acc2d, axis=0) - corr
    if HAS_BIAS:                                                 # fused bias: no extra kernel launch
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + pid_m * stride_cm + offs_n * stride_cn, acc.to(tl.float16), mask=n_mask)


@triton.jit
def _w4_dot_kernel(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr, b_ptr,
    M, N, K, M_BUCKET,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    GROUP: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiled tensor-core version for moderate M. One program = one [BLOCK_M, BLOCK_N] tile of C."""
    tl.static_assert(GROUP % BLOCK_K == 0, "BLOCK_K must divide the group size")
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):                              # K % GROUP == 0 => K % BLOCK_K == 0: no k mask
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=m_mask[:, None], other=0.0)                                     # [BM, BK] fp16
        # [BK, BN] view of the packed words: 8 neighbouring k rows read the same (coalesced) word row
        words = tl.load(qw_ptr + (offs_k[:, None] // 8) * stride_qk + offs_n[None, :] * stride_qn,
                        mask=n_mask[None, :], other=0)
        q = ((words >> ((offs_k[:, None] % 8) * 4)) & 0xF).to(tl.float16)                # exact: 0..15
        g = k0 // GROUP
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0).to(tl.float32)
        part = tl.dot(a, q)                                                              # sum_k a*q, fp32
        a_sum = tl.sum(a.to(tl.float32), axis=1)                                         # [BM]
        acc += part * s[None, :] - a_sum[:, None] * (s * z)[None, :]

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.float16),
             mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _w4_dequant_kernel(
    qw_ptr, s_ptr, z_ptr, out_ptr,
    KW, N,
    stride_qk, stride_qn, stride_sg, stride_sn, stride_ok, stride_on,
    GROUP: tl.constexpr, BLOCK_KW: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Unpack + dequantize to a [K, N] fp16 matrix (for the large-M path, followed by cuBLAS)."""
    tl.static_assert(GROUP % 8 == 0, "group size must be a multiple of 8")
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_kw = pid_k * BLOCK_KW + tl.arange(0, BLOCK_KW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_kw < KW)[:, None] & (offs_n < N)[None, :]
    words = tl.load(qw_ptr + offs_kw[:, None] * stride_qk + offs_n[None, :] * stride_qn, mask=mask, other=0)
    g = (offs_kw * 8) // GROUP                                   # all 8 k of a word share a group
    s = tl.load(s_ptr + g[:, None] * stride_sg + offs_n[None, :] * stride_sn, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + g[:, None] * stride_sg + offs_n[None, :] * stride_sn, mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(8):
        q = ((words >> (4 * j)) & 0xF).to(tl.float32)
        w = ((q - z) * s).to(tl.float16)
        tl.store(out_ptr + (offs_kw * 8 + j)[:, None] * stride_ok + offs_n[None, :] * stride_on, w, mask=mask)


# =============================================================================================
# 3. W8A16 kernels (q is [K, N] int8, scale per output channel -> applied once at the end)
# =============================================================================================
@triton.jit
def _w8_gemv_kernel(
    a_ptr, q_ptr, s_ptr, c_ptr, b_ptr,
    M, N, K, M_BUCKET,
    stride_am, stride_ak, stride_qk, stride_qn, stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    a_row = a_ptr + pid_m * stride_am
    acc2d = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(a_row + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + offs_k[:, None] * stride_qk + offs_n[None, :] * stride_qn,
                    mask=k_mask[:, None] & n_mask[None, :], other=0).to(tl.float32)
        acc2d += a[:, None] * q
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = tl.sum(acc2d, axis=0) * s
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + pid_m * stride_cm + offs_n * stride_cn, acc.to(tl.float16), mask=n_mask)


@triton.jit
def _w8_dot_kernel(
    a_ptr, q_ptr, s_ptr, c_ptr, b_ptr,
    M, N, K, M_BUCKET,
    stride_am, stride_ak, stride_qk, stride_qn, stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        q = tl.load(q_ptr + offs_k[:, None] * stride_qk + offs_n[None, :] * stride_qn,
                    mask=k_mask[:, None] & n_mask[None, :], other=0)
        acc = tl.dot(a, q.to(tl.float16), acc)                   # int8 -> fp16 is exact
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * s[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.float16),
             mask=m_mask[:, None] & n_mask[None, :])


# =============================================================================================
# 4. Autotune configs (one autotuner per kernel + group size, built lazily)
# =============================================================================================
def _cfg(warps, stages, **blocks):
    return triton.Config(blocks, num_warps=warps, num_stages=stages)


# gemv: many programs (small BLOCK_N) keep 40 SMs busy; [BLOCK_K/8, BLOCK_N] register tile stays small
W4_GEMV_CONFIGS = [
    _cfg(2, 2, BLOCK_N=32, BLOCK_K=128), _cfg(2, 3, BLOCK_N=64, BLOCK_K=64), _cfg(4, 2, BLOCK_N=64, BLOCK_K=128),
    _cfg(4, 3, BLOCK_N=128, BLOCK_K=64), _cfg(4, 2, BLOCK_N=128, BLOCK_K=128), _cfg(8, 2, BLOCK_N=256, BLOCK_K=64),
    _cfg(4, 4, BLOCK_N=32, BLOCK_K=64), _cfg(8, 3, BLOCK_N=256, BLOCK_K=128), _cfg(2, 4, BLOCK_N=16, BLOCK_K=128),
]
W4_DOT_CONFIGS = [
    _cfg(4, 3, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64), _cfg(4, 3, BLOCK_M=16, BLOCK_N=128, BLOCK_K=64),
    _cfg(2, 4, BLOCK_M=16, BLOCK_N=32, BLOCK_K=128), _cfg(4, 3, BLOCK_M=32, BLOCK_N=64, BLOCK_K=64),
    _cfg(4, 3, BLOCK_M=32, BLOCK_N=128, BLOCK_K=32), _cfg(4, 3, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64),
    _cfg(8, 3, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32), _cfg(8, 2, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32),
]
W4_DEQUANT_CONFIGS = [
    _cfg(4, 2, BLOCK_KW=8, BLOCK_N=128), _cfg(4, 2, BLOCK_KW=16, BLOCK_N=64), _cfg(8, 2, BLOCK_KW=16, BLOCK_N=256),
    _cfg(4, 2, BLOCK_KW=4, BLOCK_N=256),
]
W8_GEMV_CONFIGS = [
    _cfg(2, 3, BLOCK_N=32, BLOCK_K=128), _cfg(4, 3, BLOCK_N=64, BLOCK_K=64), _cfg(4, 2, BLOCK_N=128, BLOCK_K=32),
    _cfg(4, 4, BLOCK_N=64, BLOCK_K=32), _cfg(8, 2, BLOCK_N=256, BLOCK_K=16), _cfg(2, 4, BLOCK_N=16, BLOCK_K=128),
]
W8_DOT_CONFIGS = [
    _cfg(4, 3, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64), _cfg(4, 3, BLOCK_M=16, BLOCK_N=128, BLOCK_K=64),
    _cfg(2, 4, BLOCK_M=16, BLOCK_N=32, BLOCK_K=128), _cfg(4, 3, BLOCK_M=32, BLOCK_N=64, BLOCK_K=64),
    _cfg(4, 3, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64), _cfg(8, 3, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32),
    _cfg(8, 2, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32),
]

# Fixed configs used when autotune is off (CPU interpreter, or w4a16.AUTOTUNE = False for quick debugging).
DEFAULT_CONFIGS = {
    "w4_gemv": dict(BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2),
    "w4_dot": dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3),
    "w4_dequant": dict(BLOCK_KW=8, BLOCK_N=128, num_warps=4, num_stages=2),
    "w8_gemv": dict(BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3),
    "w8_dot": dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3),
}
AUTOTUNE = os.environ.get("TRITON_INTERPRET", "0") != "1"   # the interpreter cannot benchmark configs

# Fast launch path. The first call for a (kernel, group, M-bucket, N, K) goes through triton.autotune; its winning
# config is then cached here and every later call launches the compiled kernel directly. That skips the
# autotuner's per-call key building, which matters at batch-1 decode where ~200 small kernels run per token.
FAST_LAUNCH = True
KERNEL_VERSION = 3          # bump when kernels change, so the notebook re-measures kernel rows
_TUNED = {}
_BEST = {}


def _default_cfg(name, group):
    cfg = dict(DEFAULT_CONFIGS[name])
    if group is not None and "BLOCK_K" in cfg:
        while group % cfg["BLOCK_K"]:                           # shrink BLOCK_K until it divides the group
            cfg["BLOCK_K"] //= 2
    return cfg


def _tuned(name, kernel, configs, key, group=None):
    """Autotuned wrapper of `kernel`, restricted to configs valid for this group size (BLOCK_K | group)."""
    k = (name, group)
    if k not in _TUNED:
        cfgs = [c for c in configs if group is None or "BLOCK_K" not in c.kwargs or group % c.kwargs["BLOCK_K"] == 0]
        assert cfgs, f"no valid autotune config for {name} with group {group}"
        _TUNED[k] = triton.autotune(configs=cfgs, key=key)(kernel)
    return _TUNED[k]


def _config_dict(cfg):
    d = dict(cfg.kwargs)
    d["num_warps"], d["num_stages"] = cfg.num_warps, cfg.num_stages
    return d


def _run(name, kernel, configs, key, fast_key, grid, config, args, consts, group=None):
    """Launch `kernel`. Priority: explicit config > cached tuned config > default (no autotune) > autotune."""
    if config is None and FAST_LAUNCH:
        config = _BEST.get(fast_key)
    if config is None and not AUTOTUNE:
        config = _default_cfg(name, group)
    if config is not None:
        kernel[grid(config)](*args, **consts, **config)
        return
    tuned = _tuned(name, kernel, configs, key, group)
    tuned[grid](*args, **consts)
    best = getattr(tuned, "best_config", None)
    if best is not None:
        _BEST[fast_key] = _config_dict(best)


def clear_launch_cache():
    _BEST.clear()


def _m_bucket(M: int) -> int:
    """Nearby M share a tuned config, so a 100- vs 101-token prompt does not re-tune."""
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        if M <= b:
            return b
    return 4096


# =============================================================================================
# 5. Python entry points
# =============================================================================================
def dequant_int4(qweight, scales, zeros, group_size: int = 128, config: dict = None):
    """Triton dequantization: packed [K//8, N] -> fp16 [K, N]."""
    KW, N = qweight.shape
    K = KW * 8
    gs = K if group_size in (-1, 0) else group_size
    out = torch.empty((K, N), device=qweight.device, dtype=torch.float16)
    args = (qweight, scales, zeros, out, KW, N, qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1), out.stride(0), out.stride(1))
    grid = lambda meta: (triton.cdiv(KW, meta["BLOCK_KW"]), triton.cdiv(N, meta["BLOCK_N"]))
    _run("w4_dequant", _w4_dequant_kernel, W4_DEQUANT_CONFIGS, ["KW", "N"], ("w4_dequant", gs, KW, N),
         grid, config, args, dict(GROUP=gs))
    return out


def w4a16_matmul(A, qweight, scales, zeros, group_size: int = 128, path: str = None, config: dict = None, bias=None):
    """C = A @ dequant(W)^T (+ bias).  path: None (auto by M) | "gemv" | "dot" | "dequant".
    config: fixed block sizes for the chosen kernel (skips autotune); used by tests/experiments."""
    assert A.dtype == torch.float16 and A.dim() == 2, "A must be a 2-D fp16 tensor"
    if not A.is_contiguous():
        A = A.contiguous()
    M, K = A.shape
    KW, N = qweight.shape
    assert KW * 8 == K, f"qweight must be [K//8, N]; got {tuple(qweight.shape)} for K={K}"
    gs = K if group_size in (-1, 0) else group_size
    path = path or _pick_path(M)

    if path == "dequant":
        W = dequant_int4(qweight, scales, zeros, gs, config=config)
        return A @ W if bias is None else torch.addmm(bias, A, W)

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    mb = _m_bucket(M)
    args = (A, qweight, scales, zeros, C, C if bias is None else bias, M, N, K, mb,
            A.stride(0), A.stride(1), qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1), C.stride(0), C.stride(1))
    consts = dict(GROUP=gs, HAS_BIAS=bias is not None)
    if path == "gemv":
        _run("w4_gemv", _w4_gemv_kernel, W4_GEMV_CONFIGS, ["M_BUCKET", "N", "K"], ("w4_gemv", gs, mb, N, K),
             lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"])), config, args, consts, gs)
    elif path == "dot":
        _run("w4_dot", _w4_dot_kernel, W4_DOT_CONFIGS, ["M_BUCKET", "N", "K"], ("w4_dot", gs, mb, N, K),
             lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"])), config, args, consts, gs)
    else:
        raise ValueError(f"unknown path {path!r}")
    return C


def w8a16_matmul(A, q, scales, path: str = None, config: dict = None, bias=None):
    """C = A @ (q * scales) (+ bias)  with q [K, N] int8 (per-output-channel scales [N])."""
    assert A.dtype == torch.float16 and A.dim() == 2, "A must be a 2-D fp16 tensor"
    if not A.is_contiguous():
        A = A.contiguous()
    M, K = A.shape
    K2, N = q.shape
    assert K2 == K and q.dtype == torch.int8, "q must be [K, N] int8"
    path = path or _pick_path(M)

    if path == "dequant":
        W = q * scales                                            # one fused elementwise op (int8*fp16 -> fp16)
        return A @ W if bias is None else torch.addmm(bias, A, W)

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    mb = _m_bucket(M)
    args = (A, q, scales, C, C if bias is None else bias, M, N, K, mb,
            A.stride(0), A.stride(1), q.stride(0), q.stride(1), C.stride(0), C.stride(1))
    consts = dict(HAS_BIAS=bias is not None)
    if path == "gemv":
        _run("w8_gemv", _w8_gemv_kernel, W8_GEMV_CONFIGS, ["M_BUCKET", "N", "K"], ("w8_gemv", mb, N, K),
             lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"])), config, args, consts)
    elif path == "dot":
        _run("w8_dot", _w8_dot_kernel, W8_DOT_CONFIGS, ["M_BUCKET", "N", "K"], ("w8_dot", mb, N, K),
             lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"])), config, args, consts)
    else:
        raise ValueError(f"unknown path {path!r}")
    return C


# =============================================================================================
# 6. nn.Module wrappers, so a model's nn.Linear layers can be swapped for kernel-backed ones
# =============================================================================================
class W4A16Linear(torch.nn.Module):
    """Drop-in nn.Linear replacement. `packed` = (qweight, scales, zeros) from pack_int4, or None to RTN-quantize
    `linear` here. `in_scale` (optional, [K]) divides the input first: used by AWQ (y = (x / s) @ Q(W * s)^T).
    The bias is fused into the kernel (no separate add launch)."""

    def __init__(self, linear, group_size=128, packed=None, sym=False, in_scale=None):
        super().__init__()
        qw, s, z = packed if packed is not None else quantize_pack_int4(linear.weight.data, group_size, sym)
        dev = linear.weight.device
        self.register_buffer("qweight", qw.to(dev).contiguous())
        self.register_buffer("scales", s.to(dev).contiguous())
        self.register_buffer("zeros", z.to(dev).contiguous())
        self.register_buffer("inv_in_scale", None if in_scale is None else (1.0 / in_scale.float()).half().to(dev))
        self.bias = linear.bias
        self.group_size, self.in_features, self.out_features = group_size, linear.in_features, linear.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if self.inv_in_scale is not None:
            x2 = x2 * self.inv_in_scale
        y = w4a16_matmul(x2, self.qweight, self.scales, self.zeros, self.group_size, bias=self.bias)
        return y.view(*shp[:-1], self.out_features)


class W8A16Linear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        q, s = quantize_int8_per_channel(linear.weight.data)
        self.register_buffer("q", q)
        self.register_buffer("scales", s)
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        shp = x.shape
        y = w8a16_matmul(x.reshape(-1, shp[-1]), self.q, self.scales, bias=self.bias)
        return y.view(*shp[:-1], self.out_features)


# =============================================================================================
# 7. What to optimize next (after measuring v2)
# =============================================================================================
# * Split-K for layers with small N at decode (e.g. down_proj N=1536): more programs to fill 40 SMs.
# * dot path: each word is still read 8x from cache (once per nibble row). Unpacking a [BK/8, BN] tile
#   into 8 sub-tiles and issuing 8 tl.dot calls on strided slices of A removes that.
# * Fold the AWQ input scale into the preceding RMSNorm weight instead of an extra elementwise multiply.
# * Decode is launch-bound at batch 1: see decode_bench.py for the CUDA-graph decode loop.
