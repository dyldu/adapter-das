#!/usr/bin/env python3
"""Compare launch geometries against official Post and changed-input replays.

This probe does not change serving policy. Run only on an idle authorized GPU.
"""
import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(93118)
    from triton.testing import do_bench_cudagraph
    from hcu_adapter.kernels.mhc_post_5120_triton import _mhc_post_5120, post_5120
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx938'
    configurations = [(block, warps) for block in (128, 256, 512, 1024)
                      for warps in (2, 4, 8)]
    report = dict(unit_gate_passed=False, serving_admitted=False, cases=[],
                  graph_replays=[], timing=[], reference_source_sha256=hashlib.sha256(
                      inspect.getsource(mhc_post_torch).encode()).hexdigest())

    def exact(actual, expected, label):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
        assert torch.equal(actual.contiguous().view(torch.uint8),
                           expected.contiguous().view(torch.uint8)), label

    def capture(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn()
        return graph, output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for rows in (1, 4, 8, 16, 32, 40, 48, 64, 256, 1024):
                x = torch.empty(rows, 5120, device='cuda', dtype=torch.bfloat16)
                residual = torch.empty(rows, 4, 5120, device='cuda', dtype=torch.bfloat16)
                post = torch.empty(rows, 4, 1, device='cuda', dtype=torch.float32)
                comb = torch.empty(rows, 4, 4, device='cuda', dtype=torch.float32)

                def inputs(magnitude):
                    x.normal_().mul_(magnitude)
                    residual.normal_().mul_(magnitude)
                    post.normal_()
                    comb.normal_()

                outputs = {}
                for block, warps in configurations:
                    outputs[block, warps] = torch.empty_like(residual)

                def execute(block, warps):
                    out = outputs[block, warps]
                    _mhc_post_5120[(rows, (5120 + block - 1) // block)](
                        x, residual, post, comb, out, block,
                        num_warps=warps, enable_fp_fusion=False)
                    return out

                for magnitude in (0., .001, .25, 1., 8., 100.):
                    inputs(magnitude)
                    saved = [t.clone() for t in (x, residual, post, comb)]
                    expected = mhc_post_torch(x, residual, post, comb)
                    exact(post_5120(x, residual, post, comb), expected, 'accepted Post')
                    for block, warps in configurations:
                        exact(execute(block, warps), expected,
                              f'eager rows{rows} magnitude{magnitude} block{block} warps{warps}')
                        report['cases'].append(dict(rows=rows, magnitude=magnitude,
                                                    block=block, warps=warps, exact=True))
                    for t, before in zip((x, residual, post, comb), saved):
                        exact(t, before, 'input modified')

                if rows <= 64:
                    for block, warps in configurations:
                        graph, output = capture(lambda: execute(block, warps))
                        for replay, magnitude in enumerate((.001, .25, 1., 8., 100.)):
                            inputs(magnitude)
                            graph.replay()
                            expected = mhc_post_torch(x, residual, post, comb)
                            exact(output, expected, f'graph rows{rows} block{block} warps{warps}')
                            report['graph_replays'].append(dict(rows=rows, block=block,
                                warps=warps, replay=replay, exact=True))
                        del graph
                    # Alternate order to avoid always timing one geometry first.
                    for repeat in range(3):
                        order = configurations + [('accepted', 4)]
                        random.Random(93118 + repeat).shuffle(order)
                        for block, warps in order:
                            function = (lambda: post_5120(x, residual, post, comb)) if block == 'accepted' else (
                                lambda block=block, warps=warps: execute(block, warps))
                            us = do_bench_cudagraph(function, rep=40) * 1000
                            report['timing'].append(dict(rows=rows, block=block, warps=warps,
                                                         repeat=repeat, us=us))
                del outputs
        report['unit_gate_passed'] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(unit_gate_passed=True, serving_admitted=False,
                         cases=len(report['cases']), graph_replays=len(report['graph_replays']))))


if __name__ == '__main__':
    main()
