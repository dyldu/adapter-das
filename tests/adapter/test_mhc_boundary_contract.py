# SPDX-License-Identifier: Apache-2.0
"""Admission and owned-seam regression checks without initializing CUDA."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from hcu_adapter.kernels import mhc_boundary_triton as kernel
from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider


def inputs(device='cpu', m=8):
    return [torch.empty((m, 4, 5120), dtype=torch.bfloat16, device=device),
            torch.empty((24, 20480), dtype=torch.float32, device=device),
            torch.empty(3, device=device), torch.empty(24, device=device),
            torch.empty((m, 4), device=device),
            torch.empty((m, 5120), dtype=torch.bfloat16, device=device),
            torch.empty((m, 4, 1), device=device), torch.empty((m, 4, 4), device=device)]


def admitted(values, **changes):
    scalars = dict(rms_eps=1.e-20, pre_eps=1.e-6, sinkhorn_eps=1.e-6, post_mult=2., repeat=20)
    scalars.update(changes)
    return kernel.supported(*values, **scalars)


def test_cpu_guard_does_not_initialize_cuda():
    with torch.inference_mode():
        assert not admitted(inputs())
    assert not torch.cuda.is_initialized()


@pytest.fixture
def fake_cuda(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    precision = torch.get_float32_matmul_precision()
    enabled = torch.is_autocast_enabled('cuda')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda _: NS(gcnArchName='gfx938', multi_processor_count=64))
    try:
        torch.set_float32_matmul_precision('highest')
        torch.set_autocast_enabled('cuda', False)
        with FakeTensorMode(), torch.inference_mode():
            yield inputs('cuda')
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.set_autocast_enabled('cuda', enabled)


def test_actual_service_contract_and_optional_previous(fake_cuda):
    assert admitted(fake_cuda)
    assert admitted(fake_cuda, rms_eps=1.e-6)
    fake_cuda[4] = None
    assert admitted(fake_cuda)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change', [dict(rms_eps=-1.), dict(pre_eps=1.e-5),
                                  dict(sinkhorn_eps=1.e-5), dict(post_mult=float('nan')), dict(repeat=0)])
def test_other_scalar_contracts_fall_back(fake_cuda, change):
    assert not admitted(fake_cuda, **change)


@pytest.mark.parametrize('m', [0])
def test_other_batch_sizes_fall_back(fake_cuda, m):
    assert not admitted(inputs('cuda', m))


@pytest.mark.parametrize('index', range(8))
def test_wrong_dtype_falls_back(fake_cuda, index):
    fake_cuda[index] = fake_cuda[index].to(torch.float64)
    assert not admitted(fake_cuda)


@pytest.mark.parametrize('index', range(8))
def test_logical_negative_view_falls_back(fake_cuda, index):
    fake_cuda[index] = torch._neg_view(fake_cuda[index])
    assert not admitted(fake_cuda)


def test_autocast_grad_and_lower_precision_fall_back(fake_cuda):
    torch.set_autocast_enabled('cuda', True)
    assert not admitted(fake_cuda)
    torch.set_autocast_enabled('cuda', False)
    with torch.enable_grad():
        assert not admitted(fake_cuda)
    torch.set_float32_matmul_precision('high')
    assert not admitted(fake_cuda)


def declarations(monkeypatch, split='auto', project=True):
    from hcu_adapter.bridges import vllm_mhc as bridge
    from hcu_adapter.kernels import mhc_aiter as backend

    class Base:
        def _maybe_post(self, residual, *_):
            return residual

    mhc = NS(MHCPreOp=type('Pre', (Base,), {}), MHCFusedPostPreOp=type('Fused', (Base,), {}),
             MHCPreDelayedOp=type('Delayed', (Base,), {}), MHCPostOp=type('Post', (Base,), {}))
    monkeypatch.setattr(bridge, 'require_healthy', lambda: None)
    monkeypatch.setattr(torch.ops.vllm, 'mhc_pre_mix_triton', Mock(), raising=False)
    monkeypatch.setattr(torch.ops.vllm, 'hc_collapse_triton', Mock(), raising=False)
    provider = MhcAiterProvider(project=Mock() if project else None,
                               tilelang_pre=Mock(), tilelang_enabled=True)
    return bridge._declare_aiter(mhc, Mock(), NS(ops=NS(rms_norm=Mock())),
                                backend, provider, split)


def test_admitted_seam_bypasses_post_and_delayed_reference(monkeypatch):
    values = inputs()
    residual, fn, scale, base, previous, x, post, comb = values
    monkeypatch.setattr(kernel, 'supported', lambda *a: True)
    functions = declarations(monkeypatch)
    expected = tuple(object() for _ in range(5))
    fused = Mock(return_value=expected)
    fallback = Mock(side_effect=AssertionError('delayed fallback must not run'))
    monkeypatch.setattr(torch.ops.vllm, 'hcu_adapter_mhc_boundary', fused, raising=False)
    monkeypatch.setattr(torch.ops.vllm, 'hcu_adapter_mhc_delayed', fallback, raising=False)
    owner = functions[2][2]()
    owner._maybe_post = Mock(side_effect=AssertionError('separate Post must not run'))
    result = owner.forward_hip(residual, fn, scale, base, 1.e-20, 1.e-6, 1.e-6, 2., 20,
        pre_mix=previous, sublayer_out=x, post_layer_mix=post, comb_res_mix=comb)
    assert result is expected
    assert fused.call_args.args[-4:] == (1.e-20, 1.e-6, 2., 20)
    assert all(a is b for a, b in zip(fused.call_args.args[:8],
               (residual, x, post, comb, fn, scale, base, previous)))
    fallback.assert_not_called()


@pytest.mark.parametrize('reason', ['unsupported', 'split_override', 'missing_project', 'direct_x'])
def test_unadmitted_seam_keeps_existing_post_and_delayed(monkeypatch, reason):
    residual, fn, scale, base, previous, x, post, comb = inputs()
    monkeypatch.setattr(kernel, 'supported', lambda *a: reason != 'unsupported')
    functions = declarations(monkeypatch, split='16' if reason == 'split_override' else 'auto',
                             project=reason != 'missing_project')
    expected = tuple(object() for _ in range(4))
    fallback = Mock(return_value=expected)
    fused = Mock(side_effect=AssertionError('unadmitted fusion must not run'))
    monkeypatch.setattr(torch.ops.vllm, 'hcu_adapter_mhc_delayed', fallback, raising=False)
    monkeypatch.setattr(torch.ops.vllm, 'hcu_adapter_mhc_boundary', fused, raising=False)
    owner = functions[2][2]()
    owner._maybe_post = Mock(return_value=residual)
    result = owner.forward_hip(residual, fn, scale, base, 1.e-6, 1.e-6, 1.e-6, 2., 20,
        pre_mix=previous, x=x if reason == 'direct_x' else None,
        sublayer_out=x, post_layer_mix=post, comb_res_mix=comb)
    assert result[0] is residual and all(a is b for a, b in zip(result[1:], expected))
    owner._maybe_post.assert_called_once()
    fallback.assert_called_once()
    fused.assert_not_called()


def test_boundary_fake_has_five_distinct_outputs(monkeypatch):
    residual, fn, scale, base, previous, x, post, comb = inputs()
    fake = declarations(monkeypatch)[4][1]
    out = fake(residual, x, post, comb, fn, scale, base, previous, None, 1.e-6, 1.e-20, 1.e-6, 2., 20)
    assert [tuple(value.shape) for value in out] == [(8,4,5120),(8,4,1),(8,4,4),(8,5120),(8,4)]
    assert [value.dtype for value in out] == [torch.bfloat16,torch.float32,torch.float32,
                                            torch.bfloat16,torch.float32]
    assert len({value.data_ptr() for value in out} | {residual.data_ptr()}) == 6
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('m', [17, 65, 96, 128, 256])
def test_dynamic_rows_and_scalar_parameters(fake_cuda, m):
    assert admitted(inputs('cuda', m), rms_eps=1.e-5, pre_eps=1.e-4,
                    sinkhorn_eps=1.e-4, post_mult=1.5, repeat=7)
