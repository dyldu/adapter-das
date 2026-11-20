# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual public mHC OOT owner, registered fake, and graph replay."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 4, 8, 16, 32, 40, 48, 64])
    parser.add_argument('--rms-eps', type=float)
    parser.add_argument('--hc-eps', type=float)
    parser.add_argument('--post-mult', type=float, default=2.)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()
    root, out = Path(args.model), Path(args.output)
    config = json.loads((root / 'config.json').read_text())
    config = config.get('text_config', config)
    rms_eps, hc_eps = config['rms_norm_eps'], config['hc_eps']
    rms_eps = rms_eps if args.rms_eps is None else args.rms_eps
    hc_eps = hc_eps if args.hc_eps is None else args.hc_eps
    assert config['hc_mult'] == 4 and config['hc_sinkhorn_iters'] == 20
    assert not out.exists()
    from hcu_adapter.bridges import vllm_mhc as bridge
    from hcu_adapter.kernels import mhc_boundary_triton as kernel
    from hcu_adapter.kernels.mhc_aiter import MhcAiterRuntime, load_provider
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mhc import MHCPreDelayedOp
    from vllm.ir import ops as ir_ops
    from vllm.model_executor.kernels.mhc import torch as torch_kernels

    report = dict(passed=False, serving_admitted=False, eager=[], graphs=[], fake_checked=False,
                  rms_eps=rms_eps, hc_eps=hc_eps,
                  rms_reference='The public vllm.ir.ops.rms_norm used by the accepted delayed runtime')

    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')

    save()
    assert bridge.install()
    bridge.require_healthy()
    reference = MhcAiterRuntime(load_provider(), torch_kernels, ir_ops.rms_norm,
        torch.ops.vllm.mhc_pre_mix_triton, torch.ops.vllm.hc_collapse_triton)
    config = VllmConfig()
    with set_current_vllm_config(config):
        owner = MHCPreDelayedOp()
    assert type(owner).__name__ == 'HcuMHCPreDelayedOp'
    report['owner'] = type(owner).__qualname__
    report['schema'] = str(torch.ops.vllm.hcu_adapter_mhc_boundary.default._schema)
    save()
    mapping = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']

    def load(key):
        with safe_open(str(root / mapping[key]), framework='pt', device='cpu') as handle:
            return handle.get_tensor(key).cuda().contiguous()

    fn, scale, base = [load('layers.0.hc_attn_' + suffix) for suffix in ('fn', 'scale', 'base')]
    norm = load('layers.0.attn_norm.weight')
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(20261123)
    for m in args.rows:
        residual = torch.randn((m, 4, 5120), device='cuda', dtype=torch.bfloat16)
        x = torch.randn((m, 5120), device='cuda', dtype=torch.bfloat16)
        previous = torch.rand((m, 4), device='cuda', dtype=torch.float32)
        post_in = torch.rand((m, 4, 1), device='cuda', dtype=torch.float32)
        comb_in = torch.rand((m, 4, 4), device='cuda', dtype=torch.float32)
        assert kernel.supported(residual, fn, scale, base, previous, x, post_in, comb_in,
                                rms_eps, hc_eps, hc_eps, args.post_mult, args.repeat)

        def run():
            return owner.forward_hip(residual, fn, scale, base, rms_eps, hc_eps, hc_eps, args.post_mult, args.repeat,
                pre_mix=previous, norm_weight=norm, norm_eps=rms_eps,
                sublayer_out=x, post_layer_mix=post_in, comb_res_mix=comb_in)

        def check(actual):
            updated, post, comb, layer, pre = reference.boundary(
                residual, fn, scale, base, previous, x, post_in, comb_in,
                rms_eps=rms_eps, pre_eps=hc_eps, sinkhorn_eps=hc_eps,
                post_mult=args.post_mult, repeat=args.repeat)
            normed = ir_ops.rms_norm(layer, norm, rms_eps)
            expected = (updated, post, comb, normed, pre)
            assert len(actual) == len(expected) == 5
            checks = [dict(output=i, exact=bool(torch.equal(a, b)),
                           max_abs=float((a.float()-b.float()).abs().max()))
                      for i, (a, b) in enumerate(zip(actual, expected))]
            if not all(value['exact'] for value in checks):
                report['failed_outputs'] = checks
                save()
                raise AssertionError(checks)

        for magnitude in (0., .001, 8.):
            residual.normal_().mul_(magnitude)
            x.normal_().mul_(magnitude)
            check(run())
            report['eager'].append(dict(m=m, magnitude=magnitude, exact=True))
            save()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = run()
        for iteration, magnitude in enumerate((.001, .25, 8.)):
            residual.normal_().mul_(magnitude)
            x.normal_().mul_(magnitude)
            previous.uniform_(.1, .9)
            graph.replay()
            check(result)
            report['graphs'].append(dict(m=m, replay=iteration, exact=True))
            save()
        del graph
    # AOT tracing sees the real registered schema/fake, with static scalar arguments.
    with set_current_vllm_config(config):
        compiled = torch.compile(lambda a, b, c, d, e, f, g, h, i:
            torch.ops.vllm.hcu_adapter_mhc_boundary(a, b, c, d, e, f, g, h, i,
                rms_eps, rms_eps, hc_eps, args.post_mult, args.repeat),
            backend='aot_eager', fullgraph=True)
        traced = compiled(residual, x, post_in, comb_in, fn, scale, base, previous, norm)
        check(traced)
    report.update(passed=True, fake_checked=True)
    save()


if __name__ == '__main__':
    main()
