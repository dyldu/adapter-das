# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent membership oracle, interpreter boundaries, and real HIP parity."""
import os

import pytest
import torch

from hcu_adapter.kernels.indexer_topk_compact_triton import (
    _compact_candidate_blocks, _compact_topk_scores,
    _compact_blocks_and_short_output, select_topk_compact,
)
from hcu_adapter.kernels.indexer_topk_lightop import _topk_native_output, select_topk


def _members(row, length, candidates, block_size):
    width = row.numel()
    nblocks = (width + block_size - 1) // block_size
    admitted = set()
    for candidate in candidates.tolist():
        candidate = min(candidate, nblocks)
        if candidate == nblocks and length == width:
            admitted.add(width - 1)
        elif candidate >= 0:
            admitted.update(range(candidate * block_size,
                                  min((candidate + 1) * block_size, length)))
    return sorted(admitted)


def _oracle(row, length, candidates, block_size, k):
    """FP32 value ranking, independent of the compact representation/bit map."""
    ids = _members(row, length, candidates, block_size)
    ids = [i for i in ids if row[i] != -float('inf')]
    ids.sort(key=lambda i: (bool(torch.isnan(row[i])),
                           0.0 if torch.isnan(row[i]) else float(row[i]), -i), reverse=True)
    ids = ids[:k]
    if length <= k:
        ids.sort()
    else:
        bits = row.view(torch.int32).long() & 0xffffffff

        def output_key(i):
            value = int(bits[i])
            if value & 0x7fffffff > 0x7f800000:
                key = 0xffffffff
            else:
                key = (~value & 0xffffffff) if value & 0x80000000 else value ^ 0x80000000
            return key, i

        ids.sort(key=output_key, reverse=True)
    return torch.tensor(ids + [-1] * (k - len(ids)), dtype=torch.int32)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='CPU Triton interpreter')
@pytest.mark.parametrize('width,block_size,count', [(257, 8, 64), (65539, 8, 2048), (65539, 17, 129)])
def test_short_prefix_direct_output_interpreted(width, block_size, count):
    rows, k = 3, 512
    torch.manual_seed(251125)
    x = torch.randn(rows, width)
    x[:, :4] = torch.tensor([float('nan'), -float('inf'), float('inf'), -0.])
    ends = torch.tensor([min(width, 257), 0, min(width, k)], dtype=torch.int32)
    candidates = torch.randint(-1, (width + block_size - 1)//block_size + 3, (rows, count), dtype=torch.int32)
    candidates[:, :4] = torch.tensor([0, 0, -1, 2**30], dtype=torch.int32)
    mapping = torch.empty_like(candidates)
    lengths, offsets = torch.empty(rows, dtype=torch.int32), torch.empty(rows, dtype=torch.int32)
    pitch = ((count * block_size + 3)//4)*4
    scores = torch.full((rows, pitch), float('nan'))
    output = torch.full((rows, k), -9876, dtype=torch.int32)
    _compact_blocks_and_short_output[(rows,)](
        x, candidates, ends, mapping, lengths, offsets, output,
        width, x.stride(0), block_size, count, 1 << (count - 1).bit_length(),
        *candidates.stride(), k, 1 << ((k + block_size - 1)//block_size).bit_length(), num_warps=4)
    assert torch.count_nonzero(lengths) == torch.count_nonzero(offsets) == 0
    assert torch.isnan(scores).all()
    for row in range(rows):
        torch.testing.assert_close(output[row], _oracle(x[row], int(ends[row]), candidates[row], block_size, k), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='CPU Triton interpreter')
@pytest.mark.parametrize('width,block_size,count', [
    (33, 8, 1), (33, 4, 6), (513, 8, 9), (2049, 17, 23), (65539, 8, 2048),
])
@pytest.mark.parametrize('turn', [0, 1, 2])
def test_compact_kernels_interpreted(width, block_size, count, turn):
    rows, k = 3, 512
    torch.manual_seed(251125 + turn)
    x = torch.empty((rows, width + 7))[:, :width]
    x.normal_()
    # Include deterministic ties and all IEEE exceptional ranks.
    x[:, :8] = torch.tensor([0, -2147483648, 0x7f800000, -8388608,
                             0x7fc00001, 1, -2147483647, 0x7fa12345],
                            dtype=torch.int32).view(torch.float32)
    ends = torch.tensor([width, (width - 1) if turn == 0 else 7, 0], dtype=torch.int32)
    x.masked_fill_(torch.arange(width)[None, :] >= ends[:, None], float('nan'))
    candidates = torch.empty((rows, count * 2), dtype=torch.int32)[:, ::2]
    candidates.random_(-1, (width + block_size - 1) // block_size + 3)
    candidates[:, :min(5, count)] = torch.tensor([0, 0, -1, 2**30, 1])[:min(5, count)]
    if turn == 1:
        candidates.fill_(-1)
        candidates[:, -1] = 2**30
    elif turn == 2:
        candidates.fill_(0)
        candidates[:, -1] = (width - 1) // block_size
    saved = x.view(torch.int32).clone()
    mapping = torch.full((rows, count), -7654321, dtype=torch.int32)
    lengths = torch.empty(rows, dtype=torch.int32)
    offsets = torch.full_like(lengths, -9876)
    compact_width = count * block_size
    pitch = ((compact_width + 3) // 4) * 4
    scores = torch.full((rows, pitch), float('nan'))
    _compact_candidate_blocks[(rows,)](
        candidates, ends, mapping, lengths, offsets, width, block_size,
        count, 1 << (count - 1).bit_length(), *candidates.stride(), num_warps=4)
    _compact_topk_scores[(rows, 2)](
        x, ends, mapping, lengths, scores, width, x.stride(0), count,
        block_size, pitch, 2048, num_warps=4)
    assert torch.count_nonzero(offsets) == 0
    # Model only the documented native core: ordered IEEE keys, ascending-ID
    # ties, and -1 padding. This does not claim the actual HIP core was run.
    selected = torch.full((rows, k), -1, dtype=torch.int32)
    for row in range(rows):
        length = int(lengths[row])
        bits = scores[row, :length].view(torch.int32).long() & 0xffffffff
        assert not torch.isnan(scores[row, :length]).any()
        keys = torch.where(bits & 0x80000000 != 0, ~bits & 0xffffffff, bits ^ 0x80000000)
        order = torch.argsort(keys, descending=True, stable=True)[:k]
        selected[row, :order.numel()] = order.to(torch.int32)
        # Every live source member appears once, despite repeated candidates.
        mapped = []
        for block in mapping[row, :length // block_size].tolist():
            nblocks = (width + block_size - 1) // block_size
            if block == nblocks:
                mapped.append(width - 1)
            else:
                mapped.extend(range(block * block_size,
                                    min((block + 1) * block_size, int(ends[row]))))
        assert mapped == _members(x[row], int(ends[row]), candidates[row], block_size)
        assert len(mapped) == len(set(mapped))
    actual = torch.empty((rows, k), dtype=torch.int32)
    _topk_native_output[(rows,)](
        x, selected, torch.zeros_like(ends), ends, actual, mapping,
        width, x.stride(0), k, False, (width + block_size - 1) // block_size,
        block_size, True, count, num_warps=4)
    for row in range(rows):
        expected = _oracle(x[row], int(ends[row]), candidates[row], block_size, k)
        torch.testing.assert_close(actual[row], expected, rtol=0, atol=0)
    assert torch.equal(saved, x.view(torch.int32))


@pytest.mark.parametrize('rows,width,block_size,count,k', [
    (1, 4097, 8, 32, 512), (7, 65536, 8, 2048, 512),
    (17, 32769, 8, 2048, 512), (35, 131072, 8, 2048, 512),
    (42, 65539, 8, 2048, 512), (48, 65536, 8, 2048, 1024),
    (65, 65536, 8, 2048, 2048), (129, 8193, 17, 129, 512),
])
def test_hip_compact_matches_full_chain_and_replays(rows, width, block_size, count, k):
    if os.environ.get('TRITON_INTERPRET') == '1':
        pytest.skip('actual HIP test, not interpreter')
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires gfx938 and LightOp')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx938':
        pytest.skip('requires gfx938')
    device = torch.device('cuda', 0)
    x = torch.empty((rows, width + 7), device=device)[:, :width]
    candidates = torch.empty((rows, count * 2), device=device, dtype=torch.int32)[:, ::2]
    ends = torch.empty(rows, device=device, dtype=torch.int32)
    starts = torch.zeros_like(ends)

    def fill(turn):
        torch.manual_seed(251125 + turn)
        x.normal_()
        ends.copy_((torch.arange(rows, device=device, dtype=torch.int32) * 131
                    + turn * 103) % (width + 1))
        ends[0] = width
        candidates.random_(-1, (width + block_size - 1) // block_size + 3)
        candidates[:, :4] = torch.tensor([0, 0, -1, 2**30], device=device)
        if rows > 1:
            ends[-1] = 0
        if turn == 2:
            x.copy_((torch.arange(width, device=device) % 3).float()[None, :])
        if turn == 3:
            candidates.fill_(-1)
            candidates[:, -1] = 2**30
        x[:, :8] = torch.tensor([0, -2147483648, 0x7f800000, -8388608,
                                 0x7fc00001, 1, -2147483647, 0x7fa12345],
                                dtype=torch.int32, device=device).view(torch.float32)
        x.masked_fill_(torch.arange(width, device=device)[None, :] >= ends[:, None], float('nan'))

    def before():
        return select_topk(x, starts, ends, k, valid_lengths=ends,
                           candidates=candidates, block_size=block_size)

    def after():
        return select_topk_compact(x, starts, ends, k, candidates=candidates, block_size=block_size)

    fill(0)
    torch.testing.assert_close(after(), before(), rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        after()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = after()
    for turn in range(1, 4):
        fill(turn)
        saved = x.view(torch.int32).clone()
        graph.replay()
        torch.testing.assert_close(actual, before(), rtol=0, atol=0)
        assert torch.equal(saved, x.view(torch.int32))


def test_cpu_declines():
    x = torch.randn(3, 65536)
    saved = x.clone()
    ends = torch.tensor([65536, 17, 0], dtype=torch.int32)
    assert select_topk_compact(x, ends * 0, ends, 512,
        candidates=torch.zeros(3, 2048, dtype=torch.int32), block_size=8) is None
    assert torch.equal(x, saved)
