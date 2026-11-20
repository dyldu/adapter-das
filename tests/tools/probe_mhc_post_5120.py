#!/usr/bin/env python3
"""Compare the actual H5120 adapter path with Torch, including graph replay.

Run on an otherwise idle gfx938 GPU. This probe rejects any differing BF16
bit, checks input immutability, and reports isolated graph timings separately.
"""
import argparse
import json
from types import SimpleNamespace

import torch

torch.cuda.set_device(0)
torch.set_float32_matmul_precision("highest")

from triton.testing import do_bench_cudagraph
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime
from hcu_adapter.kernels.mhc_post_5120_triton import post_5120, supported


def reject_native(*args):
    raise AssertionError("Unvalidated native H5120 Post was selected")


runtime = MhcAiterRuntime(
    MhcAiterProvider(post=reject_native),
    SimpleNamespace(mhc_post_torch=mhc_post_torch), None, None, None)


def inputs(tokens, magnitude=1.0, layout=3):
    x = (torch.randn(tokens, 5120, device="cuda") * magnitude).bfloat16()
    residual = (torch.randn(tokens, 4, 5120, device="cuda") * magnitude).bfloat16()
    post = torch.rand(tokens, 4, 1, device="cuda") * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, device="cuda"), dim=-1)
    return (x, residual, post if layout == 3 else post.squeeze(-1), comb)


def reference(args):
    x, residual, post, comb = args
    return mhc_post_torch(x, residual, post.reshape(-1, 4, 1), comb)


def exact(actual, expected):
    assert torch.isfinite(actual).all().item()
    assert torch.isfinite(expected).all().item()
    different = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
    assert different == 0, (different, (actual.float() - expected.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8, 16, 32, 40, 48, 64, 128, 192, 256, 1024])
    opts = parser.parse_args()
    torch.manual_seed(194211)
    rows = []
    for tokens in opts.rows:
        for magnitude in (0.0, .001, .25, 1., 8., 100.):
            for layout in (2, 3):
                args = inputs(tokens, magnitude, layout)
                assert supported(*args)
                before = tuple(t.clone() for t in args)
                expected = reference(args)
                exact(post_5120(*args), expected)
                exact(runtime.post(*args), expected)
                for actual, old in zip(args, before):
                    assert torch.equal(actual.view(torch.uint8), old.view(torch.uint8))
                rows.append(dict(tokens=tokens, magnitude=magnitude, post_ndim=layout,
                                 different_bf16=0, direct_and_runtime=True))

    # Signed zero, exact cancellation, and BF16 halfway values. These also
    # exercise non-stochastic mixes rather than just softmax-generated weights.
    for kind in ("signed_zero", "cancellation", "halfway"):
        args = inputs(48)
        x, residual, post, comb = args
        if kind == "signed_zero":
            x.zero_(); residual.zero_(); x[:, ::2].neg_(); residual[:, :, ::2].neg_()
            post.zero_(); comb.zero_()
        elif kind == "cancellation":
            residual[:, 1].copy_(-residual[:, 0])
            residual[:, 3].copy_(-residual[:, 2])
            comb.fill_(1.)
        else:
            residual.fill_(1.); comb.zero_()
            comb[:, 0, :].fill_(1.)
            x.fill_(1.); post.fill_(1 / 256)
        exact(runtime.post(*args), reference(args))
        rows.append(dict(edge=kind, different_bf16=0))

    graph_cases = []
    for tokens in opts.rows:
        args = inputs(tokens)
        for _ in range(3):
            runtime.post(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = runtime.post(*args)
        for replay in range(5):
            changed = inputs(tokens, (0., .001, .25, 1., 8.)[replay])
            for dst, src in zip(args, changed):
                dst.copy_(src)
            graph.replay()
            exact(result, reference(args))
            graph_cases.append(dict(tokens=tokens, replay=replay, different_bf16=0))

    # Admission checks must keep unsupported layouts/dtypes/precision on Torch.
    args = inputs(4)
    for index in (0, 1):
        changed = list(args); changed[index] = changed[index].float()
        assert not supported(*changed)
    assert not supported(*inputs(0))
    changed = list(args); changed[1] = changed[1].transpose(0, 1)
    assert not supported(*changed)
    torch.set_float32_matmul_precision("high")
    assert not supported(*args)
    torch.set_float32_matmul_precision("highest")
    empty = inputs(0)
    assert runtime.post(*empty).shape == empty[1].shape

    timing = []
    for tokens in opts.rows:
        args = inputs(tokens)
        timing.append(dict(tokens=tokens,
            torch_us=do_bench_cudagraph(lambda: reference(args), rep=50) * 1000,
            adapter_us=do_bench_cudagraph(lambda: runtime.post(*args), rep=50) * 1000))
    result = dict(eager_cases=rows, graph_replays=graph_cases, timing=timing,
                  all_bf16_byte_exact=True, inputs_unchanged=True)
    with open(opts.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(eager_cases=len(rows), graph_replays=len(graph_cases),
                          all_bf16_byte_exact=True, timing=timing)), flush=True)


if __name__ == "__main__":
    main()
