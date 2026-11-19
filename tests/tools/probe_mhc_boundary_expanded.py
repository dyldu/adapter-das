# SPDX-License-Identifier: Apache-2.0
"""Validate target and DSpark boundary weights, nullable inputs, and live graphs."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--post-mult', type=float, default=2.)
    args = parser.parse_args()
    root, out = Path(args.model), Path(args.output)
    detail = out.with_suffix('.jsonl')
    assert not out.exists() and not detail.exists()
    from aiter.ops.mhc import mhc_post, mhc_pre_gemm_sqrsum
    from aiter.ops.tilelang.mhc.pre_big_fuse_kernel import pre_big_fuse_tilelang
    from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton, mhc_pre_mix_triton
    from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime
    from hcu_adapter.kernels.mhc_boundary_triton import boundary

    provider = MhcAiterProvider(post=mhc_post, project=mhc_pre_gemm_sqrsum,
                               tilelang_pre=pre_big_fuse_tilelang, tilelang_enabled=True)
    native = MhcAiterRuntime(provider, None, lambda x, *_: x,
                            mhc_pre_mix_triton, hc_collapse_triton)
    mapping = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    prefixes = [f'layers.{i}.hc_{kind}' for i in range(40) for kind in ('attn', 'ffn')]
    prefixes += [f'mtp.{i}.hc_{kind}' for i in range(3) for kind in ('attn', 'ffn')]
    report = dict(completed=False, passed=False, production_admitted=False, eager_checks=0,
                  graph_checks=0, completed_weights=[], timing=[],
                  post_mult=args.post_mult,
                  launch=dict(block_k=64, warps=4), details=str(detail),
                  contract='BF16 updated residual and collapsed layer byte-exact; FP32 post/comb/pre rtol2e-5,atol2e-6',
                  scope='All80target+6draft boundaryweights; original8capturebatches;4nullablepost/previousmodes;6eagermagnitudes+3changinggraphreplays')

    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')

    def load(key):
        with safe_open(str(root / mapping[key]), framework='pt', device='cpu') as handle:
            return handle.get_tensor(key).cuda().contiguous()

    def compare(actual, expected, context):
        assert len(actual) == len(expected) == 5
        values = []
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert a.shape == b.shape and a.dtype == b.dtype
            exact = bool(torch.equal(a, b))
            close = exact if i in (0, 3) else bool(torch.allclose(a, b, rtol=2.e-5, atol=2.e-6))
            values.append(dict(output=i, exact=exact, passed=close,
                               max_abs=float((a.float()-b.float()).abs().max())))
        record = dict(context, passed=all(value['passed'] for value in values), values=values)
        with detail.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        assert record['passed'], record

    torch.manual_seed(20261122)
    torch.set_float32_matmul_precision('highest')
    save()
    try:
        for prefix in prefixes:
            fn, scale, base = [load(prefix + '_' + suffix) for suffix in ('fn', 'scale', 'base')]
            originals = [value.clone() for value in (fn, scale, base)]
            assert fn.shape == (24, 20480)
            for m in (1, 4, 8, 16, 32, 40, 48, 64):
                residual = torch.empty((m, 4, 5120), device='cuda', dtype=torch.bfloat16)
                x = torch.empty((m, 5120), device='cuda', dtype=torch.bfloat16)
                previous = torch.empty((m, 4), device='cuda', dtype=torch.float32)
                post_in = torch.empty((m, 4, 1), device='cuda', dtype=torch.float32)
                comb_in = torch.empty((m, 4, 4), device='cuda', dtype=torch.float32)

                def inputs(magnitude):
                    residual.normal_().mul_(magnitude)
                    x.normal_().mul_(magnitude)
                    previous.uniform_(.1, .9)
                    post_in.uniform_(0., 2.)
                    comb_in.uniform_(0., .5)

                for has_post, has_previous in ((False, False), (False, True), (True, False), (True, True)):
                    prior = previous if has_previous else None
                    context = dict(weight=prefix, m=m, has_post=has_post, has_previous=has_previous)

                    def reference():
                        updated = native.post(x, residual, post_in, comb_in) if has_post else residual
                        return updated, *native.delayed(updated, fn, scale, base,
                            1.e-6, 1.e-6, 1.e-6, args.post_mult, 20, pre_mix=prior)

                    def candidate():
                        return boundary(residual, fn, scale, base, prior,
                            x if has_post else None, post_in if has_post else None,
                            comb_in if has_post else None, block_k=64, warps=4,
                            post_mult=args.post_mult)

                    for magnitude in (0., .001, .25, 1., 8., 100.):
                        inputs(magnitude)
                        values = (residual, x, previous, post_in, comb_in, fn, scale, base)
                        snapshots = [value.clone() for value in values]
                        compare(candidate(), reference(), dict(context, mode='eager', magnitude=magnitude))
                        assert all(torch.equal(a, b) for a, b in zip(values, snapshots))
                        report['eager_checks'] += 1
                    graphs, outputs = {}, {}
                    for name, function in [('native', reference), ('candidate', candidate)]:
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            function()
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            for _ in range(8):
                                result = function()
                        graphs[name], outputs[name] = graph, result
                    for iteration, magnitude in enumerate((.001, .25, 8.)):
                        inputs(magnitude)
                        fn.copy_(originals[0] * (.5 if iteration % 2 else 1.))
                        scale.copy_(originals[1] * (.9 if iteration % 2 else 1.))
                        base.copy_(originals[2] + (.001 if iteration % 2 else 0.))
                        for graph in graphs.values():
                            graph.replay()
                        compare(outputs['candidate'], outputs['native'], dict(context, mode='graph', replay=iteration))
                        report['graph_checks'] += 1
                    for value, original in zip((fn, scale, base), originals):
                        value.copy_(original)
                    if prefix == prefixes[0] and has_post and has_previous:
                        samples = {name: [] for name in graphs}
                        for repeat in range(3):
                            for name in (('native', 'candidate') if repeat % 2 else ('candidate', 'native')):
                                for _ in range(5):
                                    graphs[name].replay()
                                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                                a.record()
                                for _ in range(50):
                                    graphs[name].replay()
                                b.record()
                                b.synchronize()
                                samples[name].append(a.elapsed_time(b)*1000/400)
                        report['timing'].append(dict(m=m, samples_us=samples,
                            medians_us={name: statistics.median(values) for name, values in samples.items()}))
                    del graphs, outputs
                    report['last_completed'] = context
                    save()
            report['completed_weights'].append(prefix)
            save()
            print(json.dumps(dict(weight=prefix, completed=True, eager=report['eager_checks'], graphs=report['graph_checks'])), flush=True)
            del fn, scale, base, originals
        report.update(completed=True, passed=True)
        save()
    except Exception as error:
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
