#!/usr/bin/env python3
"""GPU oracles for static wide storage with changing live decode lengths."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from hcu_adapter.kernels.indexer_topk_lightop import select_topk
from hcu_adapter.kernels.indexer_live_mask import mask_decode_candidates


def rank_key(x):
    bits = x.contiguous().view(torch.int32).long() & 0xffffffff
    keys = torch.where(bits & 0x80000000 != 0, (~bits) & 0xffffffff, bits ^ 0x80000000)
    keys = torch.where(x == 0, 0x80000000, keys)
    return torch.where(torch.isnan(x), 0xffffffff, keys)


def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(521116)
    report = dict(passed=False, selections=[], masks=[], timings=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    for width in (512, 32768, 1048576):
        rows = 8
        backing = torch.empty((rows, width + 17), device='cuda')
        x = backing[:, 3:width+3]
        starts = torch.zeros(rows, dtype=torch.int32, device='cuda')
        ends = torch.full_like(starts, 37)
        cols = torch.arange(width, device='cuda')[None, :]
        for k in (512, 1024, 2048):
            def run():
                result = select_topk(x, starts, ends, k, valid_lengths=ends)
                assert result is not None
                return result

            x.normal_()
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for pattern in ('unique', 'edges', 'short', 'holes'):
                lens = [0, 1, 511, 512, 513, 4096, 32769, width]
                if pattern == 'short':
                    lens = [7, 0, 19, 52, 0, 3, 37, 4]
                ends.copy_(torch.tensor([min(width, n) for n in lens], dtype=torch.int32, device='cuda'))
                if pattern == 'edges':
                    bits = torch.tensor([0, 0x80000000, 1, 0x80000001,
                                         0x7f7fffff, 0xff7fffff, 0x7f800000,
                                         0xff800000, 0x7f800001, 0xff800001],
                                        dtype=torch.int64, device='cuda').int()
                    x.copy_(bits.view(torch.float32).repeat((rows*width+9)//10)[:rows*width].reshape(rows, width))
                else:
                    x.copy_(torch.rand((rows, width), device='cuda').argsort(-1).float())
                    if pattern == 'holes':
                        x[:, ::3] = -float('inf')
                # Unwritten native logits tails may contain any bit pattern.
                x.masked_fill_(cols >= ends[:, None], float('nan'))
                saved = backing.view(torch.int32).clone()
                graph.replay()
                result = captured.clone()
                torch.cuda.synchronize()
                assert torch.equal(backing.view(torch.int32), saved)
                masked = x.masked_fill(cols >= ends[:, None], -float('inf'))
                values = masked.topk(min(k, width), dim=-1).values
                actual = x.gather(1, result.clamp_min(0).long())
                wanted_keys = torch.full((rows, k), -1, dtype=torch.int64, device='cuda')
                wanted_keys[:, :values.shape[1]] = torch.where(values == -float('inf'), -1, rank_key(values))
                actual_keys = torch.where(result >= 0, rank_key(actual), -1)
                assert torch.equal(actual_keys.sort(-1, descending=True).values,
                                   wanted_keys.sort(-1, descending=True).values), (width, k, pattern)
                for row in range(rows):
                    ids = result[row][result[row] >= 0]
                    assert torch.unique(ids).numel() == ids.numel()
                    assert (ids < ends[row]).all()
                    if int(ends[row]) <= k:
                        assert torch.equal(ids, ids.sort().values)
                report['selections'].append(dict(width=width, k=k, pattern=pattern))
                save()
            del graph, captured
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
        candidates = torch.tensor([[-1, 0, 2, width//8, width//8+7, 1, -1, 3]],
                                  dtype=torch.int32, device='cuda').expand(rows, -1).contiguous()
        ends.copy_(torch.tensor([0, 1, 7, 8, 37, min(width, 4097), width-1, width],
                               dtype=torch.int32, device='cuda'))
        x.normal_()
        expected = x.clone()
        apply_candidate_mask(expected, starts, ends, candidates, 8)
        assert mask_decode_candidates(x, ends, candidates, 8)
        valid = cols < ends[:, None]
        assert torch.equal(x[valid].view(torch.uint8), expected[valid].view(torch.uint8))
        report['masks'].append(dict(width=width, valid_scores_exact=True))
        save()
    # Hold row count and live length constant while changing only storage pitch.
    for width in (32768, 1048576):
        rows = 48
        x = torch.randn((rows, width), device='cuda')
        starts = torch.zeros(rows, dtype=torch.int32, device='cuda')
        ends = torch.full_like(starts, 4096)
        select_topk(x, starts, ends, 512, valid_lengths=ends)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(8):
                select_topk(x, starts, ends, 512, valid_lengths=ends)
        timings = []
        for _ in range(3):
            for _ in range(10):
                graph.replay()
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(30):
                graph.replay()
            b.record()
            b.synchronize()
            timings.append(a.elapsed_time(b)*1000/240)
        report['timings'].append(dict(width=width, rows=rows, live=4096,
                                      median_us=statistics.median(timings), repeats_us=timings))
        del graph
        save()
    report['passed'] = True
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
