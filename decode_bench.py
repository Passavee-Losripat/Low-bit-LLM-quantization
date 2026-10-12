"""Decode-speed measurement that separates GPU work from Python / kernel-launch overhead.

At batch 1, each generated token runs ~200 small kernels. HF `generate` launches them one by one from Python,
so the GPU often waits for the CPU (launch-bound) and faster kernels barely change tokens/s.
This module measures greedy decode three ways on the same model:

    generate      HF model.generate with its default (dynamic) KV cache          -> what a user sees by default
    static eager  our own loop with a StaticCache, one Python forward per token  -> same work, no generate() machinery
    cuda graph    the same decode step captured once in a CUDA graph and replayed -> launch overhead removed;
                  the GPU runs the whole token's kernels back to back

All three are greedy, so they must produce the same tokens (checked), which keeps the speed comparison honest.
"""
import time

import torch


def make_static_cache(model, max_len: int):
    from transformers import StaticCache
    p = next(model.parameters())
    try:                                                    # transformers >= 5: lazy allocation
        return StaticCache(config=model.config, max_cache_len=max_len)
    except TypeError:                                       # transformers 4.x signature
        return StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len, device=p.device, dtype=p.dtype)


def _set_cache_length(cache, n: int):
    """Rewind the cache's write position to n. In transformers 5.x a StaticCache layer keeps its own position
    counter as a GPU tensor (`cumulative_length`, advanced in place on every forward: that is what makes CUDA-graph
    replay work). Warm-up steps advance it, so we rewind it after warming up. 4.x caches use cache_position only."""
    for layer in getattr(cache, "layers", []):
        cl = getattr(layer, "cumulative_length", None)
        if torch.is_tensor(cl):
            cl.fill_(n)
        elif isinstance(cl, int):
            layer.cumulative_length = n
        if isinstance(getattr(layer, "cumulative_length_int", None), int):
            layer.cumulative_length_int = n


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def static_greedy(model, prompt_ids: torch.Tensor, new_tokens: int, use_graph: bool = False, warmup: int = 3):
    """Greedy decode with a static KV cache. Returns (tokens [1, new_tokens], decode tokens/s).
    Timing covers tokens 2..new_tokens (the first new token comes out of prefill)."""
    dev = prompt_ids.device
    P = prompt_ids.shape[1]
    cache = make_static_cache(model, P + new_tokens + warmup + 2)   # headroom: a mis-rewound cache shows up as a token mismatch, not a crash

    logits = model(prompt_ids, past_key_values=cache, cache_position=torch.arange(P, device=dev), use_cache=True).logits
    static_ids = logits[:, -1:].argmax(-1)                  # [1, 1] next-token buffer (fixed address)
    static_pos = torch.tensor([P], device=dev)              # cache position buffer (fixed address)
    out = torch.empty(1, new_tokens, dtype=torch.long, device=dev)
    out[:, 0] = static_ids[:, 0]

    def step():
        lg = model(static_ids, past_key_values=cache, cache_position=static_pos, use_cache=True).logits
        return lg[:, -1:].argmax(-1)

    # Warm-up triggers kernel autotuning (which must happen before graph capture). It writes KV at positions >= P,
    # so afterwards the cache is rewound to P; those slots are overwritten by the real steps (and masked until then).
    if use_graph:
        assert dev.type == "cuda", "CUDA graphs need a GPU"
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                step()
        torch.cuda.current_stream().wait_stream(s)
        _set_cache_length(cache, P)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):                       # recorded, not executed: the cache does not advance here
            static_next = step()

        def run():
            graph.replay()
            return static_next
    else:
        for _ in range(warmup):
            step()
        _set_cache_length(cache, P)
        run = step

    _sync(dev)
    t0 = time.perf_counter()
    for i in range(1, new_tokens):
        nxt = run()
        out[:, i] = nxt[:, 0]
        static_ids.copy_(nxt)
        static_pos += 1
    _sync(dev)
    dt = time.perf_counter() - t0
    return out, (new_tokens - 1) / dt


@torch.no_grad()
def generate_greedy(model, prompt_ids: torch.Tensor, new_tokens: int, pad_token_id=None):
    """Reference tokens + timing from HF generate (dynamic cache)."""
    dev = prompt_ids.device
    kw = dict(attention_mask=torch.ones_like(prompt_ids), do_sample=False, pad_token_id=pad_token_id)
    model.generate(prompt_ids, max_new_tokens=4, min_new_tokens=4, **kw)        # warm-up
    _sync(dev)
    t0 = time.perf_counter()
    o1 = model.generate(prompt_ids, max_new_tokens=1, min_new_tokens=1, **kw)
    _sync(dev)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    o = model.generate(prompt_ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **kw)
    _sync(dev)
    tn = time.perf_counter() - t0
    return o[:, prompt_ids.shape[1]:], (new_tokens - 1) / max(tn - t1, 1e-9)


def decode_report(model, prompt_ids, new_tokens=128, pad_token_id=None, graph=True, repeats=3):
    """Best-of-`repeats` tokens/s for generate / static eager / CUDA graph, plus token agreement with generate."""
    ref, _ = generate_greedy(model, prompt_ids, new_tokens, pad_token_id)
    rep = {"tok_s_generate": max(generate_greedy(model, prompt_ids, new_tokens, pad_token_id)[1] for _ in range(repeats))}
    try:
        toks, best = None, 0.0
        for _ in range(repeats):
            toks, tps = static_greedy(model, prompt_ids, new_tokens, use_graph=False)
            best = max(best, tps)
        rep["tok_s_static_eager"] = best
        rep["match_static_eager"] = _agree(toks, ref)
    except Exception as e:                                   # report instead of crashing the notebook
        rep["tok_s_static_eager"] = float("nan")
        rep["static_error"] = repr(e)[:300]
        return rep
    if graph:
        try:
            best = 0.0
            for _ in range(repeats):
                toks, tps = static_greedy(model, prompt_ids, new_tokens, use_graph=True)
                best = max(best, tps)
            rep["tok_s_cuda_graph"] = best
            rep["match_cuda_graph"] = _agree(toks, ref)
        except Exception as e:                               # report instead of crashing the notebook
            rep["tok_s_cuda_graph"] = float("nan")
            rep["graph_error"] = repr(e)[:300]
    return rep


def _agree(a, b):
    """Fraction of positions before the first disagreement (greedy decodes diverge for good after one flip)."""
    n = min(a.shape[1], b.shape[1])
    neq = (a[0, :n] != b[0, :n]).nonzero()
    return 1.0 if len(neq) == 0 else neq[0].item() / n


def per_call_overhead(fn, n=300):
    """(cpu_us, wall_us) per call: cpu = time for Python to *enqueue* one call; wall = steady-state time per call.
    If cpu_us > the kernel's GPU time, the call is launch-bound: the GPU idles waiting for the next launch."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    cpu = (time.perf_counter() - t0) / n
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n
    return cpu * 1e6, wall * 1e6
