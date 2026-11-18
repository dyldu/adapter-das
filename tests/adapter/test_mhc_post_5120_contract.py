# SPDX-License-Identifier: Apache-2.0
"""Keep unsupported inputs on the existing Torch/native paths without CUDA init."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
import torch
from hcu_adapter.kernels import mhc_post_5120 as kernel
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
