# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare fused decode masking with the existing complete mask/TopK chain."""
import pytest
import torch

from hcu_adapter.kernels.indexer_live_mask_triton import mask_decode_candidates
from hcu_adapter.kernels.indexer_topk_lightop import select_topk


@pytest.fixture
def hcu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires gfx938 and LightOp')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx938':
        pytest.skip('requires gfx938')
    return torch.device('cuda', 0)


@pytest.mark.parametrize('rows,width,block_size', [
    (1, 1, 1), (3, 513, 128), (7, 4097, 64), (17, 32768, 128),
    (35, 65536, 128), (42, 131072, 128), (48, 65539, 256),
    (65, 2049, 17), (129, 8192, 64),
])
@pytest.mark.parametrize('k', [512, 1024, 2048])
def test_fused_matches_existing_chain(hcu, rows, width, block_size, k):
    # Noncontiguous rows and candidate columns are part of the contract.
    x = torch.empty((rows, width + 7), dtype=torch.float32, device=hcu)[:, :width]
    candidates = torch.empty((rows, 14), dtype=torch.int32, device=hcu)[:, ::2]
    lengths = torch.empty(rows, dtype=torch.int32, device=hcu)
    starts = torch.zeros_like(lengths)

    def fill(turn):
        torch.manual_seed(251125 + turn)
        x.normal_()
        lengths.copy_((torch.arange(rows, device=hcu, dtype=torch.int32)*131
                       + turn*103) % (width + 1))
        lengths[0] = width
        if rows > 1:
            lengths[-1] = 0
        candidates.random_(-1, (width+block_size-1)//block_size + 2)
        candidates[:, 0] = 0
        candidates[:, 1] = -1
        candidates[:, 2] = 2**30  # existing clamp-to-edge semantics
        if width > 8:
            special = torch.tensor([0, 0x80000000, 0x7f800000, 0xff800000,
                                    0x7fc00001, 1, 0x80000001, 0x7fa12345],
                                   dtype=torch.uint32, device=hcu).view(torch.float32)
            x[:, :8] = special
        # Never read unwritten score tails, including NaN-poisoned storage.
        x.masked_fill_(torch.arange(width, device=hcu)[None, :] >= lengths[:, None],
                       float('nan'))

    def before():
        scores = x.clone()
        assert mask_decode_candidates(scores, lengths, candidates, block_size)
        return select_topk(scores, starts, lengths, k, valid_lengths=lengths)

    def after():
        return select_topk(x, starts, lengths, k, valid_lengths=lengths,
                           candidates=candidates, block_size=block_size)

    fill(0)
    saved = x.view(torch.int32).clone()
    actual, reference = after(), before()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.equal(saved, x.view(torch.int32))
    # Independent finite-mask oracle: every emitted index belongs to a live,
    # admitted block, and an underfilled row contains every admissible score.
    cols = torch.arange(width, device=hcu)[None, :]
    nblocks = (width+block_size-1)//block_size
    admitted = torch.zeros((rows, width), device=hcu, dtype=torch.bool)
    for i in range(candidates.shape[1]):
        block = candidates[:, i:i+1].clamp_max(nblocks)
        admitted |= ((block >= 0) & (cols//block_size == block))
        admitted |= ((block == nblocks) & (cols == width-1))
    admitted &= (cols < lengths[:, None]) & (x != -float('inf'))
    valid = actual >= 0
    assert bool(admitted.gather(1, actual.clamp_min(0).long())[valid].all())
    assert torch.equal(valid.sum(1), admitted.sum(1).clamp_max(k))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        after()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = after()
    for turn in range(1, 4):
        fill(turn)
        graph.replay()
        torch.testing.assert_close(captured, before(), rtol=0, atol=0)


def test_cpu_declines_without_mutation():
    x = torch.randn(3, 513)
    saved = x.clone()
    ends = torch.tensor([513, 17, 0], dtype=torch.int32)
    assert select_topk(x, ends*0, ends, 512, valid_lengths=ends,
                       candidates=torch.zeros(3, 2, dtype=torch.int32),
                       block_size=128) is None
    assert torch.equal(x, saved)
