#!/usr/bin/env python3
"""Probe removal of the unused delayed-Pre layer; serving stays unchanged."""
import argparse
import hashlib
import inspect
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
    torch.manual_seed(93219)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(':')[0] == 'gfx938'
    assert props.multi_processor_count == 64
    from aiter.ops.mhc import mhc_pre_gemm_sqrsum
    from aiter.ops.tilelang.mhc.pre_big_fuse_kernel import pre_big_fuse_tilelang
    from hcu_adapter.kernels.mhc_delayed_stage2_tilelang import post_comb_only
    from hcu_adapter.kernels.mhc_post_5120_triton import post_5120
    from safetensors import safe_open
    from triton.testing import do_bench_cudagraph

    native_path = Path(inspect.getfile(pre_big_fuse_tilelang))
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == (
        '2ac06a653f02491b6c319d1e46cebcb9fd64081b8499fb2763c02c2ff9bf50a9')
    index_path = args.checkpoint / 'model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map']
    report = dict(unit_gate_passed=False, serving_admitted=False, cases=[],
                  graph_replays=[], timing=[], weights=[], native_source_sha256=
                  hashlib.sha256(native_path.read_bytes()).hexdigest(),
                  index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest())

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
            fn()
        return graph

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for layer in (0, 20, 39):
                for kind in ('attn', 'ffn'):
                    weights = []
                    for suffix in ('fn', 'scale', 'base'):
                        key = f'layers.{layer}.hc_{kind}_{suffix}'
                        with safe_open(args.checkpoint / weight_map[key],
                                       framework='pt', device='cpu') as file:
                            value = file.get_tensor(key)
                        report['weights'].append(dict(key=key, shape=list(value.shape),
                            dtype=str(value.dtype), sha256=hashlib.sha256(
                                value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
                        weights.append(value.to(device='cuda', dtype=torch.float32).contiguous())
                    fn, scale, base = weights
                    assert fn.shape == (24, 20480)
                    assert scale.shape == (3,) and base.shape == (24,)
                    for rows in (1, 4, 8, 16, 32, 40, 48, 64):
                        residual = torch.empty(rows, 4, 5120, device='cuda', dtype=torch.bfloat16)
                        x = torch.empty(rows, 5120, device='cuda', dtype=torch.bfloat16)
                        for splits in (1, 32):
                            padded = torch.empty(splits, rows, 32, device='cuda', dtype=torch.float32)
                            partial = padded[:, :, :24]
                            sqrsum = torch.empty(splits, rows, device='cuda', dtype=torch.float32)
                            post = torch.empty(rows, 4, device='cuda', dtype=torch.float32)
                            comb = torch.empty(rows, 16, device='cuda', dtype=torch.float32)
                            old_post, old_comb = torch.empty_like(post), torch.empty_like(comb)
                            unused_layer = torch.empty_like(x)

                            def inputs(magnitude):
                                residual.normal_().mul_(magnitude)
                                x.normal_().mul_(magnitude)

                            def project():
                                mhc_pre_gemm_sqrsum(partial, sqrsum, residual, fn, 64, False)

                            def native():
                                pre_big_fuse_tilelang(partial.contiguous(), sqrsum, scale, base,
                                    residual, old_post, old_comb, unused_layer, 5120,
                                    1e-6, 1e-6, 1e-6, 1.0, 20, n_splits=splits, mhc_mult=4)

                            def candidate():
                                post_comb_only(padded, sqrsum, scale, base, post, comb,
                                    rms_eps=1e-6, pre_eps=1e-6, sinkhorn_eps=1e-6,
                                    post_mult=1.0, repeat=20)

                            def compare(label):
                                exact(post, old_post, label + ' post')
                                exact(comb, old_comb, label + ' comb')
                                a = post_5120(x, residual, post.view(rows, 4, 1),
                                             comb.view(rows, 4, 4))
                                b = post_5120(x, residual, old_post.view(rows, 4, 1),
                                             old_comb.view(rows, 4, 4))
                                exact(a, b, label + ' native Post consumer')

                            for magnitude in (0., .001, .25, 1., 8., 100.):
                                inputs(magnitude)
                                project()
                                saved = [t.clone() for t in (padded, sqrsum, scale, base, residual, x)]
                                native()
                                candidate()
                                compare(f'eager L{layer}{kind} rows{rows} splits{splits}')
                                for t, before in zip((padded, sqrsum, scale, base, residual, x), saved):
                                    exact(t, before, 'modified input')
                                report['cases'].append(dict(layer=layer, kind=kind, rows=rows,
                                    splits=splits, magnitude=magnitude, exact=True))

                            def full_native():
                                project()
                                native()

                            def full_candidate():
                                project()
                                candidate()

                            reference_graph = capture(full_native)
                            candidate_graph = capture(full_candidate)
                            for replay, magnitude in enumerate((.001, .25, 1., 8., 100.)):
                                inputs(magnitude)
                                reference_graph.replay()
                                candidate_graph.replay()
                                compare(f'graph L{layer}{kind} rows{rows} splits{splits}')
                                report['graph_replays'].append(dict(layer=layer, kind=kind, rows=rows,
                                    splits=splits, replay=replay, exact=True))
                            del reference_graph, candidate_graph
                            if layer == 0 and kind == 'attn':
                                for repeat in range(3):
                                    order = [('native_stage2', native), ('candidate_stage2', candidate),
                                             ('native_full', full_native), ('candidate_full', full_candidate)]
                                    random.Random(93219 + repeat).shuffle(order)
                                    for label, function in order:
                                        us = do_bench_cudagraph(function, rep=40) * 1000
                                        report['timing'].append(dict(rows=rows, splits=splits,
                                            repeat=repeat, label=label, us=us))
        report['unit_gate_passed'] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(unit_gate_passed=True, serving_admitted=False,
                         cases=len(report['cases']), graph_replays=len(report['graph_replays']))))


if __name__ == '__main__':
    main()
