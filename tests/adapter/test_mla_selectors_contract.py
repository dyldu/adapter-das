# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from hcu_adapter.kernels import mla_selectors_triton as mla_selectors
from hcu_adapter.kernels import mla_sparse_decode_dtk


@pytest.mark.parametrize('csr', [False, True])
def test_cpu_retains_torch_integer_reference(csr):
    dense = torch.tensor([[7, -1, 2, 7], [3, 99, 1, 2]], dtype=torch.int32)
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    args = ((None, None, dense.flatten(), torch.tensor([0, 4, 6], dtype=torch.int32), 2, 10)
            if csr else (dense, lengths, None, None, 2, 10))
    assert mla_selectors.try_selectors(*args) is None
    actual = mla_sparse_decode_dtk._selectors(*args)
    reference = mla_sparse_decode_dtk._selectors_torch(*args)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    assert actual[1].tolist() == [3, 1]
    assert actual[0][0, 0, :3].tolist() == [2, 7, 7]
