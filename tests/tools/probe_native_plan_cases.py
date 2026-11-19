"""Actual native scheduling oracle, defined fields and whole attention output."""
import torch

from flash_mla import flash_mla_interface as native
from hcu_adapter.kernels.mla_native_plan_triton import prepare_native_plan
from probe_mla_selectors import capture


def check_plans():
    from vllm import _custom_ops  # noqa: F401
    angle = torch.randn(8192, 32, device='cuda')
    cs = torch.cat((angle.cos(), angle.sin()), 1)
    kv = torch.randn(256, 512, device='cuda', dtype=torch.bfloat16)
    caches = {}
    for b, padding in ((32, 320), (64, 64)):
        backing = torch.full((256 // b, b * 584 + padding), 0x71, device='cuda', dtype=torch.uint8)
        cache = backing.as_strided((256 // b, b, 584), (backing.stride(0), 584, 1))
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv, cache,
            torch.arange(256, device='cuda'), torch.arange(256, device='cuda'), cs, b, None, False)
        caches[b] = cache
    cases = [(rows, mw, ew, 32 if (i + j) % 2 else 64, 32 if (j + k) % 2 else 64)
             for i, rows in enumerate((1, 4, 8, 16, 32, 40, 48, 64))
             for j, mw in enumerate((64, 128, 512, 2048))
             for k, ew in enumerate((0, 512, 2048))]
    cases += [(rows, 128, 512, mb, eb) for rows in (8, 48) for mb in (32, 64) for eb in (32, 64)]
    records = []
    for rows, mw, ew, mb, eb in cases:
        q = torch.randn(rows, 1, 64, 512, device='cuda', dtype=torch.bfloat16)
        sink = torch.randn(64, device='cuda')
        ids = (torch.arange(mw, device='cuda', dtype=torch.int32) % 256).repeat(rows, 1, 1)
        extra_ids = (torch.arange(ew, device='cuda', dtype=torch.int32) % 256).repeat(rows, 1, 1) if ew else None
        main = torch.ones(rows, device='cuda', dtype=torch.int32)
        extra = torch.ones_like(main) if ew else None
        def plan():
            return prepare_native_plan(main, extra, 64, mw, ew,
                native.flash_mla_with_kvcache, native.get_mla_metadata, (mb, eb if ew else 0))
        def attention(metadata):
            result, _ = native.flash_mla_with_kvcache(
                q=q, k_cache=caches[mb].unsqueeze(-2), block_table=None,
                cache_seqlens=None, head_dim_v=512, tile_scheduler_metadata=metadata,
                softmax_scale=512**-.5, causal=False, is_fp8_kvcache=True,
                indices=ids, topk_length=main, attn_sink=sink,
                extra_k_cache=caches[eb].unsqueeze(-2) if ew else None,
                extra_indices_in_kvcache=extra_ids, extra_topk_length=extra)
            live = main > 0 if extra is None else main + extra > 0
            return torch.where(live[:, None, None, None], result, 0.)
        graph, captured = capture(plan)
        full_graph, full_output = capture(lambda: attention(plan()))
        for replay, pattern in enumerate(('empty', 'one', 'full', 'ramp', 'alternating', 'random', 'reverse', 'edge')):
            for values, width in ((main, mw), (extra, ew)):
                if values is None:
                    continue
                row = torch.arange(rows, device='cuda', dtype=torch.int32)
                ramp = row * width // max(rows - 1, 1)
                if pattern == 'empty':
                    values.zero_()
                elif pattern == 'one':
                    values.fill_(1)
                elif pattern == 'full':
                    values.fill_(width)
                elif pattern == 'ramp':
                    values.copy_(ramp)
                elif pattern == 'alternating':
                    values.copy_((row % 2) * width)
                elif pattern == 'random':
                    values.random_(0, width + 1)
                elif pattern == 'reverse':
                    values.copy_(width - ramp)
                else:
                    edges = torch.tensor([0, 1, 63, 64, min(65, width), width - 1, width], device='cuda', dtype=torch.int32)
                    values.copy_(edges[row % 7])
            saved = [x.clone() for x in (main, extra) if x is not None]
            oracle = native.get_mla_metadata()[0]
            expected = attention(oracle)
            eager = plan()
            assert eager.tile_scheduler_metadata is not None, 'fast plan not selected'
            actual = attention(eager)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), 'attention output mismatch'
            graph.replay(); full_graph.replay()
            assert torch.equal(full_output.view(torch.uint8), expected.view(torch.uint8)), 'attention graph mismatch'
            for candidate in (eager, captured):
                op = oracle.tile_scheduler_metadata
                cp = candidate.tile_scheduler_metadata
                active = op[:, 0] < rows
                assert torch.equal(cp[:, 0], op[:, 0]), 'partition activation mismatch'
                assert torch.equal(cp[active, :7], op[active, :7]), 'active descriptor mismatch'
                assert torch.equal(candidate.num_splits, oracle.num_splits), 'cumulative split mismatch'
            assert all(torch.equal(x, y) for x, y in zip((x for x in (main, extra) if x is not None), saved))
            records.append(dict(rows=rows, main_width=mw, extra_width=ew, main_block=mb,
                extra_block=eb if ew else 0, pattern=pattern, defined_fields_exact=True,
                native_attention_bytes_exact=True, dynamic_graph=True))
        del graph, full_graph
    return records
