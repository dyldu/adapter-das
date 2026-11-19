"""The native ABI keeps padded heads/sinks isolated and inputs unchanged."""
import pytest
import torch

from hcu_adapter.kernels.mla_decode_io_triton import native_query_and_sink, try_mask_copy


@pytest.mark.parametrize('heads', [4, 8, 16, 64, 96, 128])
@pytest.mark.parametrize('strided', [False, True])
def test_native_query_sink_values_and_alias_ownership(heads, strided):
    native_heads = 64 if heads <= 64 else 128
    backing = torch.randn(3, heads * (2 if strided else 1), 512).bfloat16()
    q = backing[:, ::2] if strided else backing
    # A full-size source sink must still pad with -inf for a partial head set.
    source = torch.randn(native_heads * (2 if strided else 1))
    sink = source[::2] if strided else source
    before = [x.clone() for x in (q, sink)]
    queries, actual_sink = native_query_and_sink(q, sink, native_heads)
    assert queries.shape == (3, 1, native_heads, 512)
    assert torch.equal(queries[:, 0, :heads], q)
    assert torch.equal(actual_sink[:heads], sink[:heads])
    assert torch.count_nonzero(queries[:, :, heads:]) == 0
    assert torch.isneginf(actual_sink[heads:]).all()
    direct = heads == native_heads and not strided
    assert (queries.data_ptr() == q.data_ptr()) == direct
    assert (actual_sink.data_ptr() == sink.data_ptr()) == direct
    assert all(torch.equal(x, y) for x, y in zip((q, sink), before))


def test_cpu_output_mask_leaves_caller_fallback_untouched():
    result = torch.randn(2, 64, 512).bfloat16()
    output = torch.full_like(result, -7.)
    saved = output.clone()
    assert not try_mask_copy(result, output, torch.ones(2, dtype=torch.int32), None)
    assert torch.equal(output, saved)
