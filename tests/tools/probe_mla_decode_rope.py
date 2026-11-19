#!/usr/bin/env python3
"""Whole native decode versus an accepted bridge, plus all BF16 copy bits."""
import argparse
import importlib.util
import json

import torch

torch.cuda.set_device(0)
from hcu_adapter.kernels import mla_decode_rope_triton as io
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_rows_ as inverse
from hcu_adapter.kernels import mla_sparse_decode_dtk as decode
from probe_mla_selectors import inputs, mutate, capture
from triton.testing import do_bench_cudagraph


def exact(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), 'Tensor byte mismatch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='Accepted mla_sparse_decode_dtk.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('hcu_adapter.kernels._decode_rope_oracle', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.manual_seed(194450)
    masks, consumers, timings = [], [], []
    angle = torch.randn(8192, 32, device='cuda'); cs = torch.cat((angle.cos(), angle.sin()), 1)
    for rows in (1, 4, 8, 16, 32, 40, 48, 64):
        for heads in (64, 128):
            size = rows * heads * 512
            # Includes every BF16 bit pattern, signed zeros and NaN payloads.
            bits = torch.arange(size, device='cuda', dtype=torch.int32).to(torch.int16)
            result = bits.view(torch.bfloat16).view(rows, heads, 512)
            output = torch.empty_like(result)
            positions = torch.arange(rows, device='cuda')
            main = torch.empty(rows, device='cuda', dtype=torch.int32)
            extra = torch.empty_like(main)
            for with_extra in (False, True):
                def run():
                    assert io.try_mask_copy_rope(result, output, main, extra if with_extra else None, positions, cs)
                main.fill_(1); extra.zero_()
                graph, _ = capture(run)
                for replay in range(5):
                    result.view(torch.int16).copy_(bits.roll(replay * 17).view_as(result))
                    main.random_(-1, 3); extra.random_(-1, 3)
                    live = main + extra > 0 if with_extra else main > 0
                    positions.random_(0, 8192)
                    expected = torch.where(live[:, None, None], result, 0.)
                    inverse(expected, positions, cs, 64)
                    saved = [t.clone() for t in (result, main, extra, positions, cs)]
                    run(); exact(output, expected)
                    graph.replay(); exact(output, expected)
                    for t, before in zip((result, main, extra, positions, cs), saved):
                        exact(t, before)
                    masks.append(dict(rows=rows, heads=heads, extra=with_extra, replay=replay))
                del graph

    from vllm import _custom_ops  # noqa: F401
    from flash_mla import flash_mla_interface as native
    angle = torch.randn(8192, 32, device='cuda')
    cs = torch.cat((angle.cos(), angle.sin()), dim=1)
    kv = torch.randn(256, 512, device='cuda', dtype=torch.bfloat16)
    slots = torch.arange(256, device='cuda')
    positions = slots + 4096
    cache = torch.zeros(8, 32, 584, device='cuda', dtype=torch.uint8)
    def fill():
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv, cache, slots, positions, cs, 32, None, False)
    fill()
    for rows in (1, 4, 8, 16, 32, 40, 48, 64):
        for heads in (4, 8, 64, 128):
            for swa_only in (True, False):
                for layout in ('contiguous', 'strided', 'unaligned'):
                    main, extra = inputs(rows, 128, True), inputs(rows, 512, True)
                    backing = torch.empty(rows, heads * 2, 512, device='cuda', dtype=torch.bfloat16)
                    q = backing[:, ::2] if layout == 'strided' else backing[:, :heads].contiguous()
                    sink = torch.empty(heads * 2, device='cuda')[::2] if layout == 'strided' else torch.empty(heads, device='cuda')
                    if layout == 'unaligned':
                        q = torch.empty(q.numel() + 1, device='cuda', dtype=q.dtype)[1:].view_as(q)
                        sink = torch.empty(heads + 1, device='cuda')[1:]
                    output_storage = torch.empty(rows, heads * 2, 512, device='cuda', dtype=q.dtype)
                    output = output_storage[:, ::2] if layout == 'strided' else torch.empty_like(q)
                    kwargs = dict(q=q, kv_cache=cache, swa_k_cache=cache, swa_only=swa_only,
                        topk_indices=None, topk_lens=None, swa_indices=None, swa_lens=None,
                        swa_ragged_indices=main[2], swa_ragged_indptr=main[3],
                        topk_ragged_indices=extra[2], topk_ragged_indptr=extra[3],
                        attn_sink=sink, scale=512**-.5, head_dim=512, nope_head_dim=448,
                        rope_head_dim=64, output=output, inv_rope_positions=positions[:rows], inv_rope_cos_sin_cache=cs, provider=native.flash_mla_with_kvcache,
                        metadata_factory=native.get_mla_metadata, epilogue=None)
                    q.normal_(); sink.normal_()
                    def consume(function):
                        rotated = function(**kwargs)
                        assert rotated in (0, rows)
                        if function is decode.try_decode:
                            assert rotated == (rows if heads in (64, 128) and layout != 'strided' else 0)
                        if rotated < rows:
                            inverse(output[rotated:], positions[:rows][rotated:], cs, 64)
                    consume(decode.try_decode)
                    graph, _ = capture(lambda: consume(decode.try_decode))
                    for replay, magnitude in enumerate((0., .001, .25, 1., 8., 100.)):
                        mutate(main, 128, replay); mutate(extra, 512, replay)
                        q.normal_().mul_(magnitude); sink.normal_()
                        kv.normal_(); positions.random_(0, 8192); fill()
                        saved = [t.clone() for t in (q, sink, cache, positions, cs, main[2], main[3], extra[2], extra[3])]
                        consume(baseline.try_decode)
                        expected = output.clone()
                        consume(decode.try_decode)
                        exact(output, expected)
                        graph.replay(); exact(output, expected)
                        assert torch.isfinite(output).all().item()
                        for t, before in zip((q, sink, cache, positions, cs, main[2], main[3], extra[2], extra[3]), saved):
                            exact(t, before)
                        consumers.append(dict(rows=rows, heads=heads, swa_only=swa_only,
                                              layout=layout, magnitude=magnitude, eager=True, graph=True, fused_rope=heads in (64, 128) and layout != 'strided'))
                    del graph
                    if rows in (8, 40, 48, 64) and heads == 64 and layout == 'contiguous':
                        for name, function in (('accepted_io_plus_rope', baseline.try_decode), ('mask_rope_bridge', decode.try_decode)):
                            timings.append(dict(rows=rows, heads=heads, swa_only=swa_only, variant=name,
                                us=do_bench_cudagraph(lambda: consume(function), rep=100) * 1000))
    assert len(masks) == 160 and len(consumers) == 1152
    result = dict(accepted=True, mask_cases=masks, native_consumer=consumers, timings=timings,
                  gate='all BF16 output bytes exact in eager/dynamic graphs; all inputs immutable')
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(accepted=True, masks=len(masks), consumer=len(consumers), timings=timings)))


if __name__ == '__main__':
    main()
