# SPDX-License-Identifier: Apache-2.0
"""The omitted output is private to delayed(); public Pre retains its layer."""
import hashlib
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from hcu_adapter.kernels import mhc_delayed_stage2 as selected
from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime


def test_cpu_rejects_before_native_import_or_cuda_init(monkeypatch):
    monkeypatch.setattr(selected, '_native_matches', Mock(side_effect=AssertionError))
    args = [torch.empty(8, 4, 5120, dtype=torch.bfloat16),
            torch.empty(32, 8, 32), torch.empty(32, 8),
            torch.empty(3), torch.empty(24)]
    assert not selected.supported(NS(tilelang_enabled=True), *args, 32)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def eligible(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: NS(gcnArchName='gfx938', multi_processor_count=64))
    monkeypatch.setattr(selected, '_native_matches', lambda *args: True)
    provider = NS(tilelang_enabled=True, tilelang_pre=lambda: None)
    with FakeTensorMode():
        args = [torch.empty(8, 4, 5120, device='cuda', dtype=torch.bfloat16),
                torch.empty(32, 8, 32, device='cuda'), torch.empty(32, 8, device='cuda'),
                torch.empty(3, device='cuda'), torch.empty(24, device='cuda')]
        yield provider, args


def test_qualified_fake_device_never_initialises_cuda(eligible):
    provider, args = eligible
    assert selected.supported(provider, *args, 32)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['autocast', 'grad', 'x', 'split', 'pitch',
                                  'device', 'dtype', 'hardware', 'provider'])
def test_unsupported_contract_keeps_native(eligible, monkeypatch, case):
    provider, args = eligible
    splits, x = 32, None
    if case == 'autocast':
        monkeypatch.setattr(torch, 'is_autocast_enabled', lambda *a: True)
    elif case == 'grad':
        args[3].requires_grad_(True)
    elif case == 'x':
        x = args[0][:, 0]
    elif case == 'split':
        splits = 16
    elif case == 'pitch':
        args[1] = args[1][:, :, :24]
    elif case == 'device':
        args[3] = torch.empty(3, device='cuda:1')
    elif case == 'dtype':
        args[4] = args[4].to(torch.bfloat16)
    elif case == 'hardware':
        monkeypatch.setattr(torch.cuda, 'get_device_properties',
                            lambda d: NS(gcnArchName='gfx938', multi_processor_count=128))
    elif case == 'provider':
        monkeypatch.setattr(selected, '_native_matches', lambda *a: False)
    assert not selected.supported(provider, *args, splits, x=x)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('unhashable', [False, True])
def test_native_source_and_live_function_changes_are_rejected(tmp_path, monkeypatch, unhashable):
    source = b'def pre_big_fuse_tilelang():\n    return 1\n'
    path = tmp_path / 'provider.py'; path.write_bytes(source)
    namespace = {'__name__': 'aiter.ops.tilelang.mhc.pre_big_fuse_kernel'}
    exec(compile(source, str(path), 'exec'), namespace)
    fn = namespace['pre_big_fuse_tilelang']; factory = [] if unhashable else object()
    key = selected._Identity(factory)
    module = NS(pre_big_fuse_tilelang=fn, _mhc_pre_big_fuse=factory)
    monkeypatch.setattr(selected, 'import_module', lambda name: module)
    monkeypatch.setattr(selected, '_NATIVE_SHA256', hashlib.sha256(source).hexdigest())
    selected._native_matches.cache_clear()
    try:
        assert selected._native_matches(fn, fn.__code__, key)
        assert selected._native_matches(fn, fn.__code__, selected._Identity(factory))
        assert not selected._native_matches(fn, fn.__code__, selected._Identity(object()))
        code = compile('def pre_big_fuse_tilelang():\n    return 2\n', str(path), 'exec').co_consts[0]
        fn.__code__ = code
        assert not selected._native_matches(fn, code, key)
    finally:
        selected._native_matches.cache_clear()


@pytest.mark.parametrize('entry', ['pre', 'delayed'])
@pytest.mark.parametrize('admit', [False, True])
@pytest.mark.parametrize('previous', [False, True])
def test_runtime_only_skips_discarded_layer_in_delayed(monkeypatch, entry, admit, previous):
    import hcu_adapter.kernels.mhc_aiter as mhc
    monkeypatch.setattr(mhc, '_gpu_tensors', lambda ts: True)
    monkeypatch.setattr(selected, 'supported', lambda *a: admit)
    events = []
    residual = torch.ones(8, 4, 5120, dtype=torch.bfloat16)
    def project(gemm, sqrsum, *args):
        events.append('project'); gemm.fill_(1); sqrsum.fill_(2)
    def native(gemm, sqrsum, scale, base, residual, post, comb, layer, *a, **kw):
        events.append('native'); assert gemm.is_contiguous()
        post.fill_(3); comb.fill_(4); layer.fill_(5)
    def candidate(padded, sqrsum, scale, base, post, comb, *a):
        events.append('candidate'); assert padded.shape == (32, 8, 32)
        post.fill_(3); comb.fill_(4)
    monkeypatch.setattr(selected, 'run', candidate)
    collapse = Mock(return_value=torch.full((8, 5120), 7, dtype=torch.bfloat16))
    norm = Mock(side_effect=lambda layer, weight, eps: layer)
    pre_mix = Mock(return_value=torch.full((8, 4), 6.0))
    runtime = MhcAiterRuntime(MhcAiterProvider(project=project, tilelang_pre=native,
                             tilelang_enabled=True), None, norm, pre_mix, collapse)
    args = (residual, torch.ones(24, 20480), torch.ones(3), torch.ones(24),
            1e-6, 1e-6, 1e-6, 1.0, 20)
    if entry == 'pre':
        result = runtime.pre(*args, n_splits=32, norm_weight=torch.ones(5120))
        assert torch.all(result[2] == 5)
        pre_mix.assert_not_called(); collapse.assert_not_called()
    else:
        mix = torch.ones(8, 4) if previous else None
        result = runtime.delayed(*args, pre_mix=mix, norm_weight=torch.ones(5120))
        assert torch.all(result[2] == (7 if previous else 1))
        pre_mix.assert_called_once()
        assert collapse.call_count == int(previous)
    assert events == ['project', 'candidate' if entry == 'delayed' and admit else 'native']
    norm.assert_called_once()
    assert torch.all(result[0] == 3) and torch.all(result[1] == 4)
    assert not torch.cuda.is_initialized()
