# SPDX-License-Identifier: Apache-2.0
"""FP64 oracle and changed-operand graphs for the auxiliary projection."""
import pytest
import torch

from hcu_adapter.kernels import aux_bf16_splitk_triton as kernel


@pytest.mark.parametrize('shape', [(0,128,33),(1,128,33),(65,128,33),(7,96,1)])
def test_cpu_contract_is_not_admitted(shape):
    m,k,n=shape
    a=torch.empty((m,k),dtype=torch.bfloat16)
    w=torch.empty((n,k),dtype=torch.bfloat16)
    assert not kernel.supported(a,w)
    with pytest.raises(ValueError):kernel.project(a,w)


@pytest.mark.parametrize('shape',[(1,32,1),(7,96,33),(17,128,64),
    (42,768,33),(48,4096,257),(64,5120,32),(7,16384,64),(35,5120,257)])
@pytest.mark.parametrize('scenario',['zeros','alternating_cancel','near_cancel','exponent_spread'])
def test_oracle_and_changing_graph(shape,scenario):
    if not torch.cuda.is_available():pytest.skip('GPU qualification requires the target device')
    torch.set_num_threads(4);torch.manual_seed(231124)
    m,k,n=shape
    a=torch.empty((m,k),dtype=torch.bfloat16,device='cuda')
    w=torch.empty((n,k),dtype=torch.bfloat16,device='cuda')
    if not kernel.supported(a,w):pytest.skip('Unqualified device')

    def change(turn):
        a.normal_(std=.25+turn);w.normal_(std=.25+turn)
        if scenario=='zeros':a.zero_()
        elif scenario in ('alternating_cancel','near_cancel'):
            a[:,::2].fill_(1+turn);a[:,1::2].fill_(-1-turn)
            w[:,1::2].copy_(w[:,::2])
            if scenario=='near_cancel':a[:,-1].add_(.03125)
        else:
            exponents=(torch.arange(k,device='cuda')%21-10).int()
            a.copy_(torch.ldexp(a.float(),exponents).bfloat16())
            w.copy_(torch.ldexp(w.float(),exponents.flip(0)).bfloat16())

    def verify(actual):
        oracle=a.cpu().double()@w.cpu().double().T
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu().double(),oracle,rtol=2e-5,atol=2e-6)

    change(0);weights_before=w.clone()
    verify(kernel.project(a,w));torch.testing.assert_close(w,weights_before,rtol=0,atol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):kernel.project(a,w)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=kernel.project(a,w)
    for turn in range(1,4):
        change(turn);weights_before=w.clone();graph.replay()
        verify(out)
        torch.testing.assert_close(out,kernel.project(a,w),rtol=0,atol=0)
        torch.testing.assert_close(w,weights_before,rtol=0,atol=0)
