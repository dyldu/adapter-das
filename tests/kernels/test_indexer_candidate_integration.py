# SPDX-License-Identifier: Apache-2.0
"""Exercise the published selector seam, its overrides, and graph replay."""
import pytest
import torch


@pytest.fixture
def selector(monkeypatch):
    from hcu_adapter.features.dsv41 import v41_fp8_indexer as reference
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask, select_candidate_blocks,
    )
    monkeypatch.setattr(reference, '_resolvers', {
        'candidate_select': lambda: select_candidate_blocks,
        'candidate_mask': lambda: apply_candidate_mask,
    })
    return reference, reference.build()['select']


@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('write', [False, True])
def test_external_helpers_keep_their_selection_contract(selector, monkeypatch, bounded, write):
    reference, select = selector
    from hcu_adapter.kernels import indexer_candidate_native_triton as producer
    from hcu_adapter.kernels import indexer_topk_compact_triton as consumer
    def forbidden(*args, **kwargs):
        pytest.fail('A custom collaborator must retain ownership')
    monkeypatch.setattr(producer, 'select_candidate_blocks_native', forbidden)
    monkeypatch.setattr(consumer, 'select_topk_compact', forbidden)
    calls = []
    def helper(*args):
        calls.append(args)
        if write:
            args[-1].fill_(0)
        else:
            args[0][:, 2:] = -float('inf')
    reference._resolvers['candidate_select' if write else 'candidate_mask'] = lambda: helper
    x = torch.tensor([[1., 3., 2., 4.]])
    starts, ends = torch.tensor([0], dtype=torch.int32), torch.tensor([4], dtype=torch.int32)
    out = torch.full((1, 4), -123, dtype=torch.int32)
    candidates = torch.full((1, 2), -456, dtype=torch.int32)
    select(x, starts, ends, out, candidates, 2, write, bounded=bounded)
    assert len(calls) == 1
    wanted = torch.tensor([[0, 1, 2, 3] if write else [0, 1, -1, -1]], dtype=torch.int32)
    assert torch.equal(out, wanted)
    if write:
        assert torch.count_nonzero(candidates) == 0


@pytest.mark.parametrize('rows,width,count', [(1, 4097, 512), (7, 65536, 2048), (42, 65539, 2048)])
def test_hip_published_producer_consumer_graph(selector, rows, width, count):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('actual HCU selector integration')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx938':
        pytest.skip('requires gfx938')
    _, select = selector
    from hcu_adapter.kernels.indexer_topk_lightop import select_topk
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask, select_candidate_blocks,
    )
    x = torch.empty((rows, width + 7), device='cuda')[:, :width]
    starts = torch.zeros(rows, device='cuda', dtype=torch.int32)
    ends = torch.empty_like(starts)
    candidates = torch.empty((rows, count), device='cuda', dtype=torch.int32)
    # A strided output also proves that copy-back leaves the reserved tail alone.
    storage = torch.full((rows, 520), -7654, device='cuda', dtype=torch.int32)
    produced, consumed = storage[:, :512], torch.empty_like(storage[:, :512])

    def fill(turn):
        torch.manual_seed(251125 + turn)
        x.normal_()
        ends.copy_((torch.arange(rows, device='cuda') * 137 + width - turn) % (width + 1))
        ends[0] = width
        if rows > 1:
            ends[-1] = 0
        x.masked_fill_(torch.arange(width, device='cuda')[None, :] >= ends[:, None], float('nan'))

    def chain():
        select(x, starts, ends, produced, candidates, 8, True, bounded=True)
        select(x, starts, ends, consumed, candidates, 8, False, bounded=True)

    fill(0)
    chain()
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        chain()
    for turn in range(3):
        fill(turn)
        saved = x.view(torch.int32).clone()
        graph.replay()
        baseline_candidates = torch.empty_like(candidates)
        # Native block reduction reads storage tails, so give the old path a
        # fully materialized reference instead of poisoned unwritten memory.
        full = x.masked_fill(torch.arange(width, device='cuda')[None, :] >= ends[:, None], -float('inf'))
        select_candidate_blocks(full, starts, ends, count, 8, baseline_candidates)
        padded = torch.nn.functional.pad(full, (0, (-width) % 8), value=-float('inf'))
        scores = padded.reshape(rows, -1, 8).amax(-1)
        scores.scatter_(1, ((ends - 1).clamp_min(0) // 8)[:, None].long(), float('inf'))
        scores.masked_fill_(ends[:, None] == 0, -float('inf'))
        values = []
        for ids in (baseline_candidates, candidates):
            selected = scores.gather(1, ids.long().clamp_min(0))
            values.append(selected.masked_fill(ids < 0, -float('inf')).sort().values)
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        apply_candidate_mask(full, starts, ends, candidates, 8)
        expected = select_topk(full, starts, ends, 512, valid_lengths=ends)
        torch.testing.assert_close(consumed, expected, rtol=0, atol=0)
        torch.testing.assert_close(produced, select_topk(x, starts, ends, 512, valid_lengths=ends), rtol=0, atol=0)
        assert torch.equal(saved, x.view(torch.int32))
        assert torch.all(storage[:, 512:] == -7654)
