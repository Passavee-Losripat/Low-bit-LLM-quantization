"""Tests for decode_bench.py on a tiny randomly initialised Qwen2 (CPU OK; CUDA-graph test needs a GPU).

The key property for CUDA-graph capture is that one decode step never needs a value from the GPU on the CPU
(no .item(), no `if tensor:`), otherwise capture fails. test_decode_step_has_no_host_sync checks exactly that
by making those operations raise during a decode step.
"""
import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

import decode_bench as db                                   # noqa: E402
import ptq                                                  # noqa: E402
from test_ptq import _tiny_model                            # noqa: E402

DEV = "cuda" if HAS_CUDA else "cpu"


def _prompt(n=12, seed=0):
    return torch.randint(0, 512, (1, n), generator=torch.Generator().manual_seed(seed)).to(DEV)


def test_static_greedy_matches_generate():
    model = _tiny_model()
    ids = _prompt()
    ref, _ = db.generate_greedy(model, ids, 16, pad_token_id=0)
    toks, tps = db.static_greedy(model, ids, 16)
    assert toks.shape == (1, 16) and tps > 0
    assert torch.equal(toks, ref), (toks, ref)


def test_static_greedy_with_w4_kernels_matches_generate():
    model = _tiny_model()
    res = ptq.rtn_quantize_model(model, 128)
    ptq.swap_w4(model, res, 128)
    ids = _prompt(seed=3)
    ref, _ = db.generate_greedy(model, ids, 8, pad_token_id=0)
    toks, _ = db.static_greedy(model, ids, 8)
    assert torch.equal(toks, ref)


class _HostSync(Exception):
    pass


def test_decode_step_has_no_host_sync(monkeypatch):
    """A captured CUDA graph cannot contain GPU->CPU reads. Make every such read raise during one decode step."""
    model = _tiny_model()
    ids = _prompt()
    P = ids.shape[1]
    cache = db.make_static_cache(model, P + 4)
    with torch.no_grad():
        model(ids, past_key_values=cache, cache_position=torch.arange(P, device=DEV), use_cache=True)
    static_ids = torch.tensor([[5]], device=DEV)
    static_pos = torch.tensor([P], device=DEV)

    def boom(*_a, **_k):
        raise _HostSync("host sync during decode step")
    for name in ("item", "tolist", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    try:
        with torch.no_grad():
            model(static_ids, past_key_values=cache, cache_position=static_pos, use_cache=True)
    finally:
        monkeypatch.undo()


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA graphs need a GPU")
def test_cuda_graph_decode_matches_eager():
    model = _tiny_model()
    ids = _prompt()
    eager, _ = db.static_greedy(model, ids, 16)
    graph, _ = db.static_greedy(model, ids, 16, use_graph=True)
    assert torch.equal(eager, graph)
