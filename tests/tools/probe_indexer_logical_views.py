#!/usr/bin/env python3
"""Check logical-negative Tensor views use Torch, not raw-pointer kernels."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-native', required=True)
    p.add_argument('--original-torch', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from hcu_adapter.kernels.indexer_topk_lightop import select_topk
    from hcu_adapter.kernels.indexer_topk_post_triton import finish_topk
    from hcu_adapter.v41_fp8_indexer import build
    old_native = load('hcu_adapter.kernels._old_topk_negative_view', args.previous_native).select_topk
    old_torch = load('_old_torch_negative_view', args.original_torch).build()['local_topk']
    wrapper = build()['local_topk']
    report = dict(passed=False, previous_reproductions=[], wrapper=[], graphs=[], refusal=[])
    for rows, width in ((1, 1024), (8, 4096)):
        x = torch.arange(width, device='cuda', dtype=torch.float32).repeat(rows, 1)
        starts = torch.ones(rows, device='cuda', dtype=torch.int32)
        ends = torch.full_like(starts, width)
        negative = torch._neg_view(x)
        expected = old_torch(negative, starts, ends, 512)
        wrong = old_native(negative, starts, ends, 512)
        assert wrong is not None and not torch.equal(wrong, expected)
        report['previous_reproductions'].append(dict(rows=rows, width=width,
            differing_indices=int((wrong != expected).sum())))
        for name, inputs in [('scores', (negative, starts, ends)),
                             ('starts', (x, torch._neg_view(starts), ends)),
                             ('ends', (x, starts, torch._neg_view(ends)))]:
            assert select_topk(*inputs, 512) is None
            assert torch.equal(wrapper(*inputs, 512), old_torch(*inputs, 512))
            report['wrapper'].append(dict(rows=rows, width=width, negative_view=name))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = wrapper(*inputs, 512)
            x.neg_()
            starts.add_(1)
            ends.sub_(1)
            graph.replay()
            assert torch.equal(result, old_torch(*inputs, 512))
            x.neg_()
            starts.sub_(1)
            ends.add_(1)
            report['graphs'].append(dict(rows=rows, width=width, negative_view=name))
            del graph, result
        # Resolved negative values remain eligible and preserve exact unique
        # score indices. They are different from a deferred negative view.
        physical = negative.resolve_neg()
        assert select_topk(physical, starts, ends, 512) is not None
        assert torch.equal(wrapper(physical, starts, ends, 512), old_torch(physical, starts, ends, 512))
        values, indices = x.topk(512, dim=-1)
        original = (values, indices, starts, ends)
        for i, name in enumerate(('values', 'indices', 'starts', 'ends')):
            tensors = list(original)
            tensors[i] = torch._neg_view(tensors[i])
            assert finish_topk(*tensors, 512, width) is None
            report['refusal'].append(dict(rows=rows, width=width, negative_view=name))
    report['passed'] = True
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
