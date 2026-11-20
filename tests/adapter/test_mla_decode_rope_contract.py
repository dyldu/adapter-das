"""Unknown native code and unsupported inputs retain the caller's rotation."""
import hashlib
import sys
import types

import torch

from hcu_adapter.kernels import mla_decode_rope_triton as rope


def test_cpu_and_missing_positions_leave_output_untouched():
    result = torch.randn(4, 64, 512).bfloat16()
    output = torch.full_like(result, -7.)
    saved = output.clone()
    counts = torch.ones(4, dtype=torch.int32)
    positions = torch.arange(4)
    cs = torch.randn(16, 64)
    assert not rope.try_mask_copy_rope(result, output, counts, None, positions, cs)
    assert not rope.try_mask_copy_rope(result, output, counts, None, None, cs)
    assert not rope.try_mask_copy_rope(result, output, counts, None, positions, None)
    assert torch.equal(output, saved)


def test_native_source_and_function_replacement_reject(monkeypatch, tmp_path):
    source = tmp_path / 'native.py'
    source.write_bytes(b'known native contract')
    native = types.SimpleNamespace(__file__=str(source),
        _inverse_rope_gptj_kernel=object(), rocm_inverse_rope_rows_=object())
    parent = types.ModuleType('vllm.v1.attention.ops')
    parent.rocm_aiter_mla_sparse = native
    monkeypatch.setitem(sys.modules, 'vllm.v1.attention.ops', parent)
    monkeypatch.setattr(rope, '_NATIVE_SOURCE_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(rope, '_native_contract', None)
    assert rope._native_matches()
    native._inverse_rope_gptj_kernel = object()
    assert not rope._native_matches()
    monkeypatch.setattr(rope, '_native_contract', None)
    source.write_bytes(b'new native contract')
    assert not rope._native_matches()
