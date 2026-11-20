# SPDX-License-Identifier: Apache-2.0
"""The handoff must not outlive, alias, or escape an owned decoder call."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from hcu_adapter import rms_int8_scope as scope


def payload():
    return (torch.empty((2, 5120), dtype=torch.bfloat16),
            torch.empty((2, 5120), dtype=torch.int8),
            torch.empty((2, 1), dtype=torch.float32))


def test_handoff_matches_identity_and_is_cleared_after_call():
    y, q, s = payload()
    assert scope.lookup(y) is None
    with scope.attention_norm_scope():
        scope.publish(y, q, s)
        assert scope.lookup(y) == (q, s)
        assert scope.lookup(y.view_as(y)) is None
        assert scope.lookup(y.clone()) is None
    assert not scope.active() and scope.lookup(y) is None
    with scope.attention_norm_scope():
        assert scope.lookup(y) is None


def test_nested_exception_restores_parent_and_finally_clears():
    y, q, s = payload()
    other, oq, os = payload()
    with pytest.raises(RuntimeError, match='outer'):
        with scope.attention_norm_scope():
            scope.publish(y, q, s)
            with pytest.raises(RuntimeError, match='inner'):
                with scope.attention_norm_scope():
                    assert scope.lookup(y) is None
                    scope.publish(other, oq, os)
                    raise RuntimeError('inner')
            assert scope.lookup(y) == (q, s)
            assert scope.lookup(other) is None
            raise RuntimeError('outer')
    assert not scope.active() and scope.lookup(y) is None


def test_foreign_python_thread_cannot_consume_handoff():
    y, q, s = payload()
    with scope.attention_norm_scope():
        scope.publish(y, q, s)
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(scope.lookup, y).result() is None
            assert not executor.submit(scope.active).result()
        assert scope.lookup(y) == (q, s)


def test_requires_scope_and_valid_payload_metadata():
    y, q, s = payload()
    with pytest.raises(RuntimeError, match='owned decoder'):
        scope.publish(y, q, s)
    with scope.attention_norm_scope():
        with pytest.raises(ValueError, match='metadata mismatch'):
            scope.publish(y, q, s.double())
        assert scope.lookup(y) is None


@pytest.mark.parametrize('rows', [1, 17, 65, 96, 256, 1025])
@pytest.mark.parametrize('eps', [0., 1.e-20, 1.e-6, 1.e-5])
def test_fusion_accepts_dynamic_rows_and_valid_epsilon(monkeypatch, rows, eps):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    monkeypatch.setattr(FakeTensor, 'data_ptr', lambda _: 16)
    with FakeTensorMode(), torch.inference_mode():
        x = torch.empty((rows, 5120), device='cuda', dtype=torch.bfloat16)
        weight = torch.empty(5120, device='cuda', dtype=torch.bfloat16)
        assert scope.supported(x, weight, eps)
        assert not scope.supported(x, weight, -1.)
        assert not scope.supported(x, weight, float('nan'))
        assert not scope.supported(x, weight, float('inf'))
