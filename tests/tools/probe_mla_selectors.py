#!/usr/bin/env python3
"""Exact selector oracle, changing-metadata graph replay and real MLA consumer."""
import argparse
import json

import torch

torch.cuda.set_device(0)
from hcu_adapter.kernels import mla_selectors as fused
from hcu_adapter.kernels import mla_sparse_decode_dtk as decode
from triton.testing import do_bench_cudagraph


def check(actual, expected):
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def inputs(rows, width, csr):
    source = torch.randint(-16, 272, (rows, width), device='cuda', dtype=torch.int32)
    if width:
        source[:, ::5] = 17
        source[:, ::7] = -1
    lengths = torch.randint(0, width + 1, (rows,), device='cuda', dtype=torch.int32)
    if csr:
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        return [None, None, source.flatten(), ptr, rows, 256]
    return [source, lengths, None, None, rows, 256]


def mutate(args, width, replay):
    dense, lengths, flat, ptr, rows, _ = args
    source = flat if flat is not None else dense
    source.random_(-16, 272)
    source.reshape(-1)[::5] = 17
    if flat is not None:
        counts = torch.randint(0, width + 1, (rows,), device='cuda', dtype=torch.int32)
        if replay == 0:
            counts.zero_()
        elif replay == 1:
            counts.fill_(width)
        ptr[0] = 0
        ptr[1:].copy_(counts.cumsum(0, dtype=torch.int32))
    else:
        lengths.random_(0, width + 1)
        if replay == 0:
            lengths.fill_(-1)
        elif replay == 1:
            lengths.fill_(width + 7)


def capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function()
    return graph, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    opts = parser.parse_args()
    torch.manual_seed(194244)
    eager, graphs, timings, consumers = [], [], [], []
    for rows in (1, 4, 8, 16, 32, 40, 48, 64):
        for width in (0, 1, 63, 64, 127, 128, 511, 512, 513, 2048):
            for csr in (False, True):
                args = inputs(rows, width, csr)
                assert fused.supported(*args)
                for replay in range(3):
                    mutate(args, width, replay)
                    saved = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]
                    expected = decode._selectors_torch(*args)
                    actual = decode._selectors(*args)
                    check(actual, expected)
                    assert all(torch.equal(a, b) for a, b in zip(args, saved)
                               if isinstance(a, torch.Tensor))
                    eager.append(dict(rows=rows, width=width, csr=csr, replay=replay))
                graph, out = capture(lambda: decode._selectors(*args))
                for replay in range(3):
                    mutate(args, width, replay)
                    graph.replay()
                    check(out, decode._selectors_torch(*args))
                    graphs.append(dict(rows=rows, width=width, csr=csr, replay=replay))
                del graph
        for width in (128, 512):
            args = inputs(rows, width, True)
            for variant, function in [('torch', decode._selectors_torch), ('fused', decode._selectors)]:
                timings.append(dict(rows=rows, width=width, variant=variant,
                    us=do_bench_cudagraph(lambda: function(*args), rep=100) * 1000))

    # Real 584-byte cache, unchanged native FlashMLA. Compare whole decode
    # outputs while changing the selectors, Q, cache and RoPE positions.
    from vllm import _custom_ops  # noqa: F401
    from flash_mla import flash_mla_interface as native
    angle = torch.randn(8192, 32, device='cuda')
    cs = torch.cat((angle.cos(), angle.sin()), dim=1)
    kv = torch.randn(256, 512, device='cuda', dtype=torch.bfloat16)
    slot_ids = torch.arange(256, device='cuda')
    positions = slot_ids + 4096
    cache = torch.zeros(8, 32, 584, device='cuda', dtype=torch.uint8)
    def fill_cache():
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv, cache, slot_ids, positions, cs, 32, None, False)
    fill_cache()
    selected = decode._selectors
    for rows in (1, 8, 40, 48):
        for swa_only in (True, False):
            main_args, extra_args = inputs(rows, 128, True), inputs(rows, 512, True)
            q = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
            output = torch.empty_like(q)
            sink = torch.randn(64, device='cuda', dtype=torch.float32)
            kwargs = dict(q=q, kv_cache=cache, swa_k_cache=cache, swa_only=swa_only,
                topk_indices=None, topk_lens=None, swa_indices=None, swa_lens=None,
                swa_ragged_indices=main_args[2], swa_ragged_indptr=main_args[3],
                topk_ragged_indices=extra_args[2], topk_ragged_indptr=extra_args[3],
                attn_sink=sink, scale=512**-.5, head_dim=512, nope_head_dim=448,
                rope_head_dim=64, output=output, provider=native.flash_mla_with_kvcache,
                metadata_factory=native.get_mla_metadata, epilogue=None)
            assert decode.try_decode(**kwargs) == 0
            graph, _ = capture(lambda: decode.try_decode(**kwargs))
            for replay in range(5):
                mutate(main_args, 128, replay)
                mutate(extra_args, 512, replay)
                q.normal_()
                sink.normal_()
                kv.normal_()
                positions.random_(0, 8192)
                fill_cache()
                graph.replay()
                actual = output.clone()
                try:
                    decode._selectors = decode._selectors_torch
                    assert decode.try_decode(**kwargs) == 0
                    expected = output.clone()
                finally:
                    decode._selectors = selected
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                assert torch.isfinite(actual).all().item()
                consumers.append(dict(rows=rows, swa_only=swa_only, replay=replay))
            del graph
    assert len(eager) == len(graphs) == 480 and len(consumers) == 40
    result = dict(accepted=True, eager=eager, graphs=graphs, native_consumer=consumers, timings=timings)
    with open(opts.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(accepted=True, eager=len(eager), graphs=len(graphs),
                         native_consumer=len(consumers), timings=timings)))


if __name__ == '__main__':
    main()
