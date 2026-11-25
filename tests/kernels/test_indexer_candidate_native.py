# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Candidate producer oracle independent of native IEEE key mapping."""
import os

import pytest
import torch
import triton

from hcu_adapter.kernels.indexer_candidate_native_triton import (
    _candidate_native_output, _candidate_native_scores, select_candidate_blocks_native,
)


def block_oracle(x, end, block_size):
    width = x.numel()
    scores = []
    for block in range((max(0, min(end, width)) + block_size - 1) // block_size):
        value = x[block * block_size:min((block + 1) * block_size, end, width)].max()
        if end > 0 and block == (end - 1) // block_size:
            value = torch.tensor(float("inf"))
        scores.append(value)
    return torch.stack(scores) if scores else torch.empty(0)


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="CPU Triton interpreter")
@pytest.mark.parametrize("width,block_size", [(17, 8), (1025, 3), (8193, 8), (65539, 8)])
def test_interpreted_candidate_scores_and_mapping(width, block_size):
    torch.manual_seed(251125)
    rows, k = 3, 512
    x = torch.randn(rows, width + 11)[:, :width]
    x[:, :8] = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), 1., -1., 2.])
    blocks = (width + block_size - 1) // block_size
    pitch = (blocks + 3) // 4 * 4
    scores = torch.empty((rows, pitch))
    lengths = torch.empty(rows, dtype=torch.int32)
    offsets = torch.empty_like(lengths)
    selected = torch.empty((rows, k), dtype=torch.int32)
    out = torch.empty((rows, k * 2), dtype=torch.int32)[:, ::2]
    for turn in range(3):
        ends = torch.tensor([width, 8 - turn, 0], dtype=torch.int32)
        x.masked_fill_(torch.arange(width)[None, :] >= ends[:, None], float("nan"))
        saved = x.view(torch.int32).clone()
        _candidate_native_scores[(rows, 2)](x, ends, scores, lengths, offsets, width,
            x.stride(0), block_size, pitch, 128, triton.next_power_of_2(block_size))
        assert torch.count_nonzero(offsets) == 0
        for row in range(rows):
            expected = block_oracle(x[row], int(ends[row]), block_size)
            length = int(lengths[row])
            assert length == expected.numel()
            assert not torch.isnan(scores[row, :length]).any()
            # Simulate only native rank selection; compare the reconstructed
            # candidate membership to independent FP32 block maxima.
            keys = scores[row, :length].view(torch.int32).long() & 0xffffffff
            ordered = torch.where(keys & 0x80000000 != 0, ~keys & 0xffffffff, keys ^ 0x80000000)
            indices = torch.argsort(ordered, descending=True, stable=True)[:k]
            selected[row].fill_(-1)
            selected[row, :indices.numel()] = indices.int()
            want = torch.argsort(expected, descending=True, stable=True)[:k]
            want = want[expected[want] != -float("inf")]
            selected_ids = indices[expected[indices] != -float("inf")]
            assert torch.equal(want, selected_ids)
        _candidate_native_output[(rows, 2)](scores, lengths, selected, out, pitch, k, *out.stride(), 256)
        for row in range(rows):
            expected = block_oracle(x[row], int(ends[row]), block_size)
            want = torch.argsort(expected, descending=True, stable=True)[:k]
            want = want[expected[want] != -float("inf")]
            actual = out[row][out[row] >= 0]
            assert torch.equal(actual.long(), want)
        assert torch.equal(x.view(torch.int32), saved)


def test_cpu_declines_without_output_write():
    x = torch.randn(3, 8193)
    lengths = torch.tensor([8193, 17, 0], dtype=torch.int32)
    out = torch.full((3, 2048), -123, dtype=torch.int32)
    assert not select_candidate_blocks_native(x, lengths, out, 8)
    assert torch.all(out == -123)


@pytest.mark.parametrize("rows,width,k,block_size", [(3, 17, 512, 8), (1, 4097, 512, 8), (7, 65536, 2048, 8),
                                                    (42, 65539, 2048, 8), (65, 32769, 1024, 17)])
def test_hip_exact_ranks_and_graph_replay(rows, width, k, block_size):
    if os.environ.get("TRITON_INTERPRET") == "1" or not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("actual HIP + LightOp test")
    x = torch.empty((rows, width), device="cuda")
    ends = torch.empty(rows, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")

    def fill(turn):
        torch.manual_seed(251125 + turn)
        x.normal_()
        x[:, :8] = torch.tensor(
            [0., -0., float('nan'), float('inf'), -float('inf'), 1., -1., 2.],
            device='cuda')
        ends.copy_((torch.arange(rows, device="cuda") * 37 + width - turn) % (width + 1))
        ends[0] = width
        if rows > 1:
            ends[-1] = 0
        x.masked_fill_(torch.arange(width, device="cuda")[None, :] >= ends[:, None], float("nan"))

    fill(0)
    assert select_candidate_blocks_native(x, ends, out, block_size)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert select_candidate_blocks_native(x, ends, out, block_size)
    for turn in range(3):
        fill(turn)
        graph.replay()
        for row in range(rows):
            expected = block_oracle(x[row].cpu(), int(ends[row]), block_size)
            selected = out[row].cpu()
            selected = selected[selected >= 0].long()
            assert selected.unique().numel() == selected.numel()
            assert selected.numel() == min(k, (expected != -float("inf")).count_nonzero())
            wanted = expected.topk(min(k, expected.numel())).values
            wanted = wanted[wanted != -float("inf")]
            torch.testing.assert_close(expected[selected].sort(descending=True).values,
                                       wanted, rtol=0, atol=0, equal_nan=True)
