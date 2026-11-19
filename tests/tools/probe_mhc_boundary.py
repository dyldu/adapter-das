# SPDX-License-Identifier: Apache-2.0
"""Compare the boundary prototype with the actual accepted delayed runtime."""
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
    args = parser.parse_args()
    root, out = Path(args.model), Path(args.output)
    config = json.loads((root / 'config.json').read_text())
    config = config.get('text_config', config)
    rms_eps, hc_eps = config['rms_norm_eps'], config['hc_eps']
    assert not out.exists()
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
    tensors = []
    for suffix in ('fn', 'scale', 'base'):
        key = 'layers.0.hc_attn_' + suffix
        with safe_open(str(root / mapping[key]), framework='pt', device='cpu') as handle:
            tensors.append(handle.get_tensor(key).cuda().contiguous())
    fn, scale, base = tensors
    original_fn = fn.clone()
    report = dict(completed=False, production_admitted=False, rows=[],
                  contract='BF16 updated residual and collapsed layer byte-exact; FP32 post/comb/pre rtol2e-5,atol2e-6; no threshold changes',
                  scope='Preliminary real layer0attn,BF16 Post+delayed projection; all candidate launch configurations')

    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')

    def compare(actual, expected):
        values = []
        for i, (a, b) in enumerate(zip(actual, expected)):
            exact = bool(torch.equal(a, b))
            close = exact if i in (0, 3) else bool(torch.allclose(a, b, rtol=2.e-5, atol=2.e-6))
            values.append(dict(output=i, exact=exact, passed=close,
                               max_abs=float((a.float()-b.float()).abs().max())))
        return dict(passed=all(value['passed'] for value in values), values=values)

    torch.manual_seed(20261121)
    torch.set_float32_matmul_precision('highest')
    save()
    for m in (8, 48, 64):
        residual = torch.empty((m, 4, 5120), device='cuda', dtype=torch.bfloat16)
        x = torch.empty((m, 5120), device='cuda', dtype=torch.bfloat16)
        previous = torch.rand((m, 4), device='cuda', dtype=torch.float32)
        post_in = torch.rand((m, 4, 1), device='cuda', dtype=torch.float32)
        comb_in = torch.rand((m, 4, 4), device='cuda', dtype=torch.float32)

        def reference():
            updated = native.post(x, residual, post_in, comb_in)
            return updated, *native.delayed(updated, fn, scale, base,
                rms_eps, hc_eps, hc_eps, 2., 20, pre_mix=previous)

        for block_k in (64, 128, 256):
            for warps in (2, 4):
                candidate = lambda: boundary(residual, fn, scale, base, previous,
                    x, post_in, comb_in, block_k=block_k, warps=warps,
                    rms_eps=rms_eps, eps=hc_eps, post_mult=2.)
                row = dict(m=m, block_k=block_k, warps=warps, eager=[], graphs=[])
                report['rows'].append(row)
                save()
                try:
                    for label, magnitude in (('zero', 0.), ('small', .001), ('normal', 1.), ('large', 8.)):
                        residual.normal_().mul_(magnitude)
                        x.normal_().mul_(magnitude)
                        saved_inputs = [t.clone() for t in (residual, x, previous, post_in, comb_in, fn, scale, base)]
                        value = compare(candidate(), reference())
                        value['pattern'] = label
                        row['eager'].append(value)
                        assert all(torch.equal(a, b) for a, b in zip(
                            (residual, x, previous, post_in, comb_in, fn, scale, base), saved_inputs))
                        save()
                    if not all(value['passed'] for value in row['eager']):
                        continue
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
                        residual.normal_().mul_(magnitude)
                        x.normal_().mul_(magnitude)
                        previous.uniform_(.1, .9)
                        fn.copy_(original_fn * (.5 if iteration % 2 else 1.))
                        for graph in graphs.values():
                            graph.replay()
                        row['graphs'].append(compare(outputs['candidate'], outputs['native']))
                        save()
                    fn.copy_(original_fn)
                    if all(value['passed'] for value in row['graphs']):
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
                        row.update(samples_us=samples,
                                   medians_us={name: statistics.median(values) for name, values in samples.items()})
                        print(json.dumps(dict(m=m, block_k=block_k, warps=warps, medians_us=row['medians_us'])), flush=True)
                    del graphs, outputs
                except Exception as error:
                    row['error'] = str(error)
                    fn.copy_(original_fn)
                    save()
                    raise
                save()
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
