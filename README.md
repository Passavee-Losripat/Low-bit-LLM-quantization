# Low-bit LLM quantization

Weight-only quantization of Qwen2.5-1.5B on a single Colab T4 (fp16 only, about 320 GB/s). I wrote the RTN, GPTQ and AWQ quantizers myself, then wrote Triton kernels that keep the weights packed in GPU memory and dequantize them in registers. Everything is compared against fp16 cuBLAS and bitsandbytes.

## Summary

- 4-bit GPTQ is +0.58 perplexity over fp16 (9.84 vs 9.27 on WikiText-2), with ARC-Easy unchanged (0.750 vs 0.751). Weights drop from 3.09 GB to 1.16 GB, 2.65x smaller.
- Batch-1 decode with the W4A16 kernels runs at 79.1 tok/s vs 52.3 tok/s for fp16, 1.51x faster. Both use CUDA graphs. Greedy output is token-identical to HF `generate`.
- At the kernel level, the 8960x1536 gate/up layer takes 47 µs vs 129 µs for fp16 cuBLAS (2.75x) at M=1.
- With more tokens per call, the W4 path is slower than fp16 cuBLAS. For the gate/up layer it is 0.55x at M=16 and 0.72x at M=512. More on that below.

## Setup

| | |
|---|---|
| Model | Qwen2.5-1.5B, fp16 |
| GPU | Tesla T4 (40 SMs, no bf16) |
| Perplexity | WikiText-2 test, 2048-token windows |
| Tasks | lm-eval zero-shot ARC-Easy (acc) and HellaSwag (acc_norm), 1000 examples each. Standard error is about 0.014, so differences under 0.03 are noise |
| Decode | batch 1, greedy, 128-token prompt, 128 new tokens |
| Versions | torch 2.11, transformers 5.18, Triton 3.6 |

Int4 weights are asymmetric with group size 128 along K unless noted. `lm_head` is tied to the embeddings and stays fp16.

## Quantizers

- **RTN**: round to the nearest level in each group.
- **GPTQ**: column-by-column quantization with error feedback through the inverse Hessian `H = 2XᵀX` of each layer's calibration inputs. Updates are done in blocks of 128 columns, with 1% dampening and no act-order.
- **AWQ**: scale each input channel by `s = mean|x|^α` before quantizing, and divide the activations by `s`. α is grid-searched per group of linears that share an input. There is no clipping search.

Calibration uses 128 windows of 512 tokens from the WikiText-2 train split. Layers are calibrated in order, on top of the already-quantized earlier layers.

## Kernels

`C = A·Wᵀ (+ bias)`, with the weights packed as `[K/8, N]` (int4) or `[K, N]` (int8). There are three paths, chosen by M:

| Path | Used for | How it works |
|---|---|---|
| `gemv` | M ≤ 4 (decode) | No tensor cores. Each packed word is loaded once and unpacked into 8 nibbles |
| `dot` | moderate M | Tiled `tl.dot` on tiles dequantized in registers. It never won for W4 in my sweep |
| `dequant` | M > 4 | Dequantize to fp16 once, then call cuBLAS |

The scale and zero-point are pulled out of the inner loop using `Σ a(q−z)s = s·Σ aq − s·z·Σ a`, and the bias is fused into the kernel. After the first autotuned call for a shape, the chosen config is cached and the compiled kernel is launched directly.

## Results

| Method | Bits | PPL | ARC-E | HellaSwag | Weights (GB) | tok/s, `generate` | tok/s, CUDA graph |
|---|---|---|---|---|---|---|---|
| FP16 | 16 | 9.265 | 0.751 | 0.583 | 3.087 | 24.0 | 52.3 |
| RTN (mine) | 8 | 9.268 | 0.743 | 0.582 | 1.818* | | |
| RTN (mine) | 4 | 10.360 | 0.705 | 0.572 | 1.163* | | |
| GPTQ (mine) | 4 | 9.844 | 0.750 | 0.578 | 1.163* | | |
| AWQ (mine) | 4 | 10.017 | 0.729 | 0.576 | 1.163* | | |
| bitsandbytes NF4 | 4 | 9.928 | 0.716 | 0.588 | 1.122 | 18.4 | |
| bitsandbytes INT8 | 8 | 9.318 | 0.745 | 0.583 | 1.777 | 6.1 | |
| W8A16 Triton | 8 | 9.279 | 0.750 | 0.583 | 1.779 | 18.4 | 66.5 |
| W4A16 Triton (RTN) | 4 | 10.357 | 0.706 | 0.571 | 1.163 | 18.1 | 78.6 |
| W4A16 Triton (GPTQ) | 4 | 9.844 | 0.751 | 0.579 | 1.163 | 17.8 | 79.1 |
| W4A16 Triton (AWQ) | 4 | 10.017 | 0.730 | 0.577 | 1.164 | 17.2 | 76.8 |

\* Theoretical packed size, since these rows are fake-quant. The weights shrink 2.65x instead of 4x mostly because the 0.47 GB embedding table stays fp16; the per-group scale and zero-point add about 0.25 bits per weight on top. Each Triton row has the same perplexity as its fake-quant version.

Kernel latency at M=1 (GPU time, `triton.testing.do_bench`, median):

| Layer (N×K) | fp16 cuBLAS | W4A16 (path) | Speedup | GB/s |
|---|---|---|---|---|
| gate/up 8960×1536 | 129 µs | 47 µs (`gemv`) | 2.75x | 157 |
| down 1536×8960 | 126 µs | 70 µs (`gemv`) | 1.81x | 105 |
| q/o 1536×1536 | 34 µs | 16 µs (`gemv`) | 2.05x | 77 |
| k/v 256×1536 | 10 µs | 14 µs (`dequant`) | 0.70x | 15 |
| 11008×4096 (7B-sized) | 367 µs | 114 µs (`gemv`) | 3.22x | 211 |

Summing all seven linears in a decoder layer at M=1 gives 0.47 ms for fp16 and 0.23 ms for W4A16. With more tokens per call it flips: at M=16 the W4 path runs at 0.41x to 0.64x of fp16 depending on the layer, and at M=512 to 1024 at 0.72x to 0.96x for the Qwen layers.

Ablations (perplexity):

| Change | PPL |
|---|---|
| RTN 4-bit, per-channel / g128 / g64 / g32 | 13.51 / 10.36 / 10.05 / 9.85 |
| RTN g128, first and last 2 layers kept in fp16 | 10.00 |
| RTN g128, also quantizing the tied `lm_head` | 10.91 |
| GPTQ g128, 16 / 64 / 128 calibration windows | 10.07 / 9.90 / 9.84 |
| GPTQ per-channel (RTN per-channel is 13.51) | 11.78 |

The most sensitive layer is the last one by a wide margin, followed by layers 1 and 26. Layers 9 to 15 barely matter.

## What I found

**GPTQ closes about half of RTN's gap.** RTN loses 1.10 PPL at 4 bits and GPTQ loses 0.58. At per-channel granularity, where RTN gives 13.5, GPTQ gets back to 11.8. Over half of the gain is already there with only 16 calibration windows.

**My AWQ did worse than GPTQ** (+0.75 PPL). Two likely reasons: I skipped the weight-clipping search, and I pick α on each linear's own output rather than on the whole block. I haven't tested either one.

**My first kernel was 9x slower than fp16.** It stored the packed weights as `[N, K/8]`, so neighbouring threads read words that were far apart in memory, and it got about 6 GB/s. Switching to `[K/8, N]`, adding a GEMV path that reads each word once, and moving the scale and zero-point out of the inner loop together took the gate/up layer from 1183 µs to 47 µs (25x). I changed all three at once, so I can't say how much of that each one contributed, but the layout is the likely main cause.

**Faster kernels did not mean faster decode at first.** With the kernels at 2.1x faster per layer, end-to-end decode was still 0.56x of fp16. I measured, for one layer, how long Python took to enqueue a call against how long the GPU took to run it:

| Call | CPU µs to enqueue | GPU µs |
|---|---|---|
| fp16 `nn.Linear` | 51 | 130 |
| W4 kernel through `triton.autotune` | 148 | 47 |
| W4 kernel, cached config, direct launch | 55 | 47 |
| `W4A16Linear` module | 141 | 47 |

A decode step makes 196 linear calls, so the GPU was waiting on Python. Caching the tuned config reduced the launch cost, and capturing the whole decode step in a CUDA graph removed it. The comparison also needed care: fp16 goes from 24 to 52 tok/s with CUDA graphs, so the fair number is 1.51x against graph-captured fp16, not 3.3x against `generate`.

**The end-to-end gain is smaller than the per-layer gain.** The linears are 2.1x faster per layer but decode is 1.51x, because the fp16 `lm_head` (about 0.47 GB read per token), attention and norms are unchanged.

**Large batches belong to cuBLAS.** Dequantizing once and calling cuBLAS keeps prefill at roughly 0.7 to 1.0x of fp16, while my first fused kernel was 0.015x. M=16 is the weakest case, at 0.41x to 0.64x depending on the layer.

## Not done yet

- The `W4A16Linear` module still costs 141 µs per call against 55 µs for the bare launch, so `generate` without CUDA graphs is still slower than fp16 (0.74x). Registering the kernels as `torch.library` custom ops should let `torch.compile` capture them.
- `lm_head` is still fp16 and is the largest remaining read per token.
- A multi-row GEMV for M = 2 to 16 would fix the weakest batch size.

## Files

| File | Contents |
|---|---|
| `quant_llm.ipynb` | Evaluation, quantizer experiments, ablations, kernel sweep and plots |
| `ptq.py` | RTN, GPTQ, AWQ, and swapping the kernels into a model |
| `w4a16.py` | Triton W4A16 and W8A16 kernels |
| `decode_bench.py` | Decode timing with `generate`, a static-cache loop, and CUDA graphs |
| `test_*.py` | pytest tests for the kernels, quantizers and decode loop |
| `kernel_bench.csv`, `launch_overhead.csv`, `figures/` | Raw data and plots |

## Running it

Use a Colab T4 runtime. Upload the notebook, the three `.py` files and the tests, run `pytest -q test_w4a16.py test_ptq.py test_decode.py`, then run all cells (about 1.5 hours). Set `USE_DRIVE = True` to keep results if the runtime resets. Marlin-style int4 kernels need an Ampere GPU or newer, so they are not included in the comparison.
