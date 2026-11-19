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
    parser.add_argument('--block-size', type=int, choices=(32, 64), default=64)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('hcu_adapter.kernels._native_plan_oracle', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.manual_seed(194450)
    masks, consumers, timings = [], [], []
    angle = torch.randn(8192, 32, device='cuda'); cs = torch.cat((angle.cos(), angle.sin()), 1)
    from probe_native_plan_cases import check_plans
    masks = check_plans()

    from vllm import _custom_ops  # noqa: F401
    from flash_mla import flash_mla_interface as native
    angle = torch.randn(8192, 32, device='cuda')
    cs = torch.cat((angle.cos(), angle.sin()), dim=1)
    kv = torch.randn(256, 512, device='cuda', dtype=torch.bfloat16)
    slots = torch.arange(256, device='cuda')
    positions = slots + 4096
    cache = torch.zeros(256 // args.block_size, args.block_size, 584, device='cuda', dtype=torch.uint8)
    def fill():
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv, cache, slots, positions, cs, args.block_size, None, False)
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
                        for name, function in (('accepted_native_plan', baseline.try_decode), ('fresh_triton_plan', decode.try_decode)):
                            timings.append(dict(rows=rows, heads=heads, swa_only=swa_only, variant=name,
                                us=do_bench_cudagraph(lambda: consume(function), rep=100) * 1000))
    assert len(masks) >= 500 and len(consumers) == 1152
    result = dict(accepted=True, cache_block_size=args.block_size, mask_cases=masks, native_consumer=consumers, timings=timings,
                  gate='all BF16 output bytes exact in eager/dynamic graphs; all inputs immutable')
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(accepted=True, masks=len(masks), consumer=len(consumers), timings=timings)))


if __name__ == '__main__':
    main()
