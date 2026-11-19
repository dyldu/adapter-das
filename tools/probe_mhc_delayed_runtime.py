#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare actual delayed runtimes, native stage1/mixes/collapse/RMS consumers."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(93220)
    from aiter.ops.mhc import mhc_pre_gemm_sqrsum
    from aiter.ops.tilelang.mhc.pre_big_fuse_kernel import pre_big_fuse_tilelang
    from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime
    from hcu_adapter.kernels.mhc_delayed_stage2 import supported
    from vllm.model_executor.kernels.mhc.triton import mhc_pre_mix_triton, hc_collapse_triton
    from vllm import _custom_ops as ops
    from safetensors import safe_open
    from triton.testing import do_bench_cudagraph

    def native_stage2(*a, **kw):
        return pre_big_fuse_tilelang(*a, **kw)

    def rms(layer, weight, eps):
        out = torch.empty_like(layer)
        ops.rms_norm(out, layer, weight, eps)
        return out

    def runtime(stage2, splits):
        provider = MhcAiterProvider(project=mhc_pre_gemm_sqrsum,
            tilelang_pre=stage2, tilelang_enabled=True)
        return MhcAiterRuntime(provider, None, rms, mhc_pre_mix_triton,
                              hc_collapse_triton, split_k=str(splits))

    def exact(actual, expected, label):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, label
        assert torch.equal(actual.contiguous().view(torch.uint8),
                           expected.contiguous().view(torch.uint8)), label

    def capture(fn):
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fn()
        return graph, result

    index = args.checkpoint / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map']
    report = dict(unit_gate_passed=False, serving_admitted=False,
                  eager=[], graph_replays=[], timing=[], weights=[],
                  index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                  actual_consumers=['native AITER stage1', 'Core mhc_pre_mix_triton',
                                    'Core hc_collapse_triton', 'Core _custom_ops.rms_norm'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for layer in (0, 20, 39):
                for kind in ('attn', 'ffn'):
                    values = []
                    for suffix in ('fn', 'scale', 'base'):
                        key = f'layers.{layer}.hc_{kind}_{suffix}'
                        with safe_open(args.checkpoint / mapping[key], framework='pt', device='cpu') as f:
                            value = f.get_tensor(key)
                        report['weights'].append(dict(key=key, sha256=hashlib.sha256(
                            value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
                        values.append(value.to('cuda').contiguous())
                    fn, scale, base = values
                    for rows in (1, 8, 48, 64):
                        residual = torch.empty(rows, 4, 5120, device='cuda', dtype=torch.bfloat16)
                        previous = torch.empty(rows, 4, device='cuda', dtype=torch.float32)
                        weight = torch.empty(5120, device='cuda', dtype=torch.bfloat16)
                        for splits in (1, 32):
                            original, candidate = runtime(native_stage2, splits), runtime(pre_big_fuse_tilelang, splits)
                            dummy = torch.empty(splits, rows, 32, device='cuda')
                            sums = torch.empty(splits, rows, device='cuda')
                            assert supported(candidate.provider, residual, dummy, sums, scale, base, splits)
                            assert not supported(original.provider, residual, dummy, sums, scale, base, splits)
                            for has_previous in (False, True):
                                mix = previous if has_previous else None
                                def inputs(magnitude):
                                    residual.normal_().mul_(magnitude)
                                    previous.normal_().mul_(.25)
                                    weight.uniform_(.5, 1.5)
                                def run(owner):
                                    return owner.delayed(residual, fn, scale, base,
                                        1e-6, 1e-6, 1e-6, 1.0, 20, pre_mix=mix,
                                        norm_weight=weight, norm_eps=1e-6)
                                def compare(actual, expected):
                                    for i, (a, b) in enumerate(zip(actual, expected)):
                                        exact(a, b, f'L{layer}{kind}T{rows}S{splits}previous{has_previous} output{i}')
                                for magnitude in (0., .001, .25, 1., 8., 100.):
                                    inputs(magnitude)
                                    tensors = (residual, previous, weight, fn, scale, base)
                                    saved = [t.clone() for t in tensors]
                                    compare(run(candidate), run(original))
                                    for t, before in zip(tensors, saved):exact(t, before, 'input modified')
                                    report['eager'].append(dict(layer=layer, kind=kind, rows=rows,
                                        splits=splits, previous=has_previous, magnitude=magnitude, exact=True))
                                ref_graph, ref_out = capture(lambda: run(original))
                                new_graph, new_out = capture(lambda: run(candidate))
                                for replay, magnitude in enumerate((.001, .25, 1., 8., 100.)):
                                    inputs(magnitude); ref_graph.replay(); new_graph.replay()
                                    compare(new_out, ref_out)
                                    report['graph_replays'].append(dict(layer=layer, kind=kind, rows=rows,
                                        splits=splits, previous=has_previous, replay=replay, exact=True))
                                del ref_graph, new_graph
                                if layer == 0 and kind == 'attn' and has_previous:
                                    for repeat in range(3):
                                        order = [('native', original), ('candidate', candidate)]
                                        random.Random(93220 + repeat).shuffle(order)
                                        for label, owner in order:
                                            us = do_bench_cudagraph(lambda: run(owner), rep=40) * 1000
                                            report['timing'].append(dict(rows=rows, splits=splits,
                                                repeat=repeat, label=label, us=us))
            report['unit_gate_passed'] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(unit_gate_passed=report['unit_gate_passed'],
                         eager=len(report['eager']), graphs=len(report['graph_replays']))))


if __name__ == '__main__':
    main()
