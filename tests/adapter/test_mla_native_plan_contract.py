"""Plan fallback creates fresh native objects; changed native code is rejected."""
import hashlib
import sys
import types

import torch

from hcu_adapter.kernels import mla_native_plan_triton as plan


def test_cpu_fallback_is_fresh_and_preserves_counts():
    main = torch.arange(8, dtype=torch.int32)
    before = main.clone()
    calls = []
    def factory():
        result = object()
        calls.append(result)
        return result, None
    a = plan.prepare_native_plan(main, None, 64, 128, 0, None, factory, (32, 0))
    b = plan.prepare_native_plan(main, None, 128, 128, 0, None, factory, (32, 0))
    assert a is calls[0] and b is calls[1] and a is not b
    assert torch.equal(main, before)


def test_unknown_native_binary_and_replaced_binding_reject(monkeypatch, tmp_path):
    interface, binary = tmp_path / 'interface.py', tmp_path / 'native.so'
    interface.write_bytes(b'known interface'); binary.write_bytes(b'known binary')
    metadata_type = type('Metadata', (), {})
    cuda = types.SimpleNamespace(__file__=str(binary), sparse_decode_fwd=object())
    native = types.SimpleNamespace(__file__=str(interface), flash_mla_cuda=cuda,
        flash_mla_with_kvcache=object(), get_mla_metadata=object(), FlashMLASchedMeta=metadata_type)
    parent = types.ModuleType('flash_mla')
    parent.flash_mla_interface = native
    monkeypatch.setitem(sys.modules, 'flash_mla', parent)
    monkeypatch.setattr(plan, '_INTERFACE_SHA', hashlib.sha256(interface.read_bytes()).hexdigest())
    monkeypatch.setattr(plan, '_BINARY_SHA', hashlib.sha256(binary.read_bytes()).hexdigest())
    monkeypatch.setattr(plan, '_contract', None)
    assert plan._native_matches(native.flash_mla_with_kvcache, native.get_mla_metadata, metadata_type())
    cuda.sparse_decode_fwd = object()
    assert not plan._native_matches(native.flash_mla_with_kvcache, native.get_mla_metadata, metadata_type())
    monkeypatch.setattr(plan, '_contract', None)
    binary.write_bytes(b'new binary')
    assert not plan._native_matches(native.flash_mla_with_kvcache, native.get_mla_metadata, metadata_type())
