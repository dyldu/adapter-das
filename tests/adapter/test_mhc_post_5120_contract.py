# SPDX-License-Identifier: Apache-2.0
"""Keep unsupported inputs on the existing Torch/native paths without CUDA init."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
import torch
from hcu_adapter.kernels import mhc_post_5120_triton as kernel
from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime


def inputs():
    return (torch.zeros(2,5120,dtype=torch.bfloat16),
        torch.zeros(2,4,5120,dtype=torch.bfloat16),
        torch.zeros(2,4,1),torch.zeros(2,4,4))


def test_cpu_inputs_rejected_without_cuda_initialization():
    args=inputs()
    assert not kernel.supported(*args)
    with pytest.raises(ValueError):kernel.post_5120(*args)
    assert not torch.cuda.is_initialized()


def test_cpu_post_keeps_torch_reference_and_native_width_guard():
    args=inputs(); expected=object()
    native=Mock(side_effect=AssertionError('unvalidated native H5120 Post'))
    reference=Mock(return_value=expected)
    runtime=MhcAiterRuntime(MhcAiterProvider(post=native),NS(mhc_post_torch=reference),None,None,None)
    assert runtime.post(*args) is expected
    reference.assert_called_once();native.assert_not_called()
    assert not torch.cuda.is_initialized()


def test_admitted_path_consumes_original_tensors_and_bypasses_native(monkeypatch):
    args=inputs(); expected=object(); observed=[]
    monkeypatch.setattr(kernel,'supported',lambda *actual:all(a is b for a,b in zip(actual,args)))
    def run(*actual):
        assert all(a is b for a,b in zip(actual,args));observed.append(actual);return expected
    monkeypatch.setattr(kernel,'post_5120',run)
    native=Mock(side_effect=AssertionError('native cast must not run'))
    reference=Mock(side_effect=AssertionError('Torch fallback must not run'))
    runtime=MhcAiterRuntime(MhcAiterProvider(post=native),NS(mhc_post_torch=reference),None,None,None)
    assert runtime.post(*args) is expected and len(observed)==1
    native.assert_not_called();reference.assert_not_called()
    assert not torch.cuda.is_initialized()


def test_missing_provider_preserves_torch_contract(monkeypatch):
    args=inputs(); expected=object()
    monkeypatch.setattr(kernel,'supported',lambda *a:pytest.fail('missing provider must not select Triton'))
    reference=Mock(return_value=expected)
    runtime=MhcAiterRuntime(MhcAiterProvider(),NS(mhc_post_torch=reference),None,None,None)
    assert runtime.post(*args) is expected


@pytest.fixture
def fake_gfx938_post(monkeypatch):
    """Exercise CUDA admission with real autocast state and no GPU allocation."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    precision = torch.get_float32_matmul_precision()
    enabled = torch.is_autocast_enabled('cuda')
    dtype = torch.get_autocast_dtype('cuda')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: NS(gcnArchName='gfx938'))
    try:
        torch.set_float32_matmul_precision('highest')
        torch.set_autocast_enabled('cuda', False)
        with FakeTensorMode():
            yield (torch.empty((2, 5120), device='cuda', dtype=torch.bfloat16),
                   torch.empty((2, 4, 5120), device='cuda', dtype=torch.bfloat16),
                   torch.empty((2, 4, 1), device='cuda', dtype=torch.float32),
                   torch.empty((2, 4, 4), device='cuda', dtype=torch.float32))
    finally:
        torch.set_autocast_enabled('cuda', enabled)
        torch.set_autocast_dtype('cuda', dtype)
        torch.set_float32_matmul_precision(precision)


def test_validated_input_is_admitted_without_autocast(fake_gfx938_post):
    assert kernel.supported(*fake_gfx938_post)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_cuda_autocast_preserves_torch_post(fake_gfx938_post, dtype, monkeypatch):
    args = fake_gfx938_post
    torch.set_autocast_dtype('cuda', dtype)
    torch.set_autocast_enabled('cuda', True)
    assert torch.get_float32_matmul_precision() == 'highest'
    assert not kernel.supported(*args)
    with pytest.raises(ValueError, match='autocast disabled'):
        kernel.post_5120(*args)

    fused = Mock(side_effect=AssertionError('FP32 fusion must not run under autocast'))
    native = Mock(side_effect=AssertionError('unvalidated native H5120 Post'))
    expected = object()

    def original(x, residual, post, comb):
        assert torch.is_autocast_enabled('cuda')
        assert torch.get_autocast_dtype('cuda') == dtype
        assert x is args[0] and residual is args[1] and comb is args[3]
        assert post.shape == args[2].shape and post.dtype == args[2].dtype
        return expected

    reference = Mock(side_effect=original)
    monkeypatch.setattr(kernel, 'post_5120', fused)
    runtime = MhcAiterRuntime(MhcAiterProvider(post=native),
                            NS(mhc_post_torch=reference), None, None, None)
    assert runtime.post(*args) is expected
    reference.assert_called_once()
    fused.assert_not_called()
    native.assert_not_called()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('rows', [17, 65, 1025, 2048])
def test_rows_are_derived_from_input(fake_gfx938_post, rows):
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        args = (torch.empty((rows, 5120), device='cuda', dtype=torch.bfloat16),
                torch.empty((rows, 4, 5120), device='cuda', dtype=torch.bfloat16),
                torch.empty((rows, 4, 1), device='cuda'),
                torch.empty((rows, 4, 4), device='cuda'))
        assert kernel.supported(*args)
