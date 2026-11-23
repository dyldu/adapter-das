# SPDX-License-Identifier: Apache-2.0
"""Independent rank oracle for the actual native live-prefix TopK consumer."""
import numpy as np
import pytest
import torch

from hcu_adapter.kernels.indexer_topk_lightop import select_topk


@pytest.mark.parametrize('m,width',[(1,513),(7,4096),(17,32768),(35,131072),
    (42,1048576),(48,131072),(64,65536),(129,4096),(257,4096)])
@pytest.mark.parametrize('k',[512,1024,2048])
@torch.inference_mode()
def test_native_live_topk_oracle_and_graph(m,width,k):
    if not torch.cuda.is_available():pytest.skip('HCU required')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx938':
        pytest.skip('Qualified gfx938 device required')
    torch.set_num_threads(4)
    x=torch.empty((m,width),device='cuda',dtype=torch.float32)
    lengths=torch.empty((m,),device='cuda',dtype=torch.int32)
    starts=torch.zeros_like(lengths)
    short=min(width,65536)

    def change(turn):
        x.fill_(torch.nan)  # Unwritten tails must never enter selection.
        lens=[min(short,max(0,(row*653+turn*131)%(short+1))) for row in range(m)]
        lens[0]=max(0,short-turn)
        if m>1:lens[-1]=turn
        expected=np.full((m,k),-1,dtype=np.int32)
        oracles=[]
        for row,length in enumerate(lens):
            if not length:
                oracles.append(None)
                continue
            values=(((np.arange(length,dtype=np.int64)*32771+turn*19)%65537)-32768).astype(np.float32)
            bits=values.view(np.uint32)
            special=np.array([0x7fc00001,0x7f800000,0xff800000,0x80000000,
                              0,1,0x80000001],dtype=np.uint32)
            bits[:min(length,len(special))]=special[:min(length,len(special))]
            x[row,:length].copy_(torch.from_numpy(values))
            # Independent integer IEEE rank + index ordering; NaNs rank first.
            ranks=np.where(bits&0x80000000,~bits,bits^0x80000000).astype(np.uint32)
            ranks[(bits&0x7fffffff)>0x7f800000]=0xffffffff
            keys=(ranks.astype(np.uint64)<<np.uint64(32))|np.arange(length,dtype=np.uint64)
            selection_ranks=ranks.copy()
            selection_ranks[(bits&0x7fffffff)==0]=0x7fffffff
            oracles.append((length,keys,selection_ranks))
            ids=np.argsort(keys)[::-1][:min(k,length)].astype(np.int32)
            ids=ids[bits[ids]!=0xff800000]
            if length<=k:ids=np.sort(ids)
            expected[row,:len(ids)]=ids
        lengths.copy_(torch.tensor(lens,dtype=torch.int32))
        return expected,oracles

    def verify(actual,oracle):
        expected,rows=oracle
        actual=actual.cpu().numpy()
        for row,details in enumerate(rows):
            if details is None or details[0]<=k:
                np.testing.assert_array_equal(actual[row],expected[row])
                continue
            length,keys,ranks=details
            ids=actual[row][actual[row]>=0]
            wanted=expected[row][expected[row]>=0]
            assert len(ids)==len(wanted) and len(np.unique(ids))==len(ids)
            assert np.all(ids<length)
            np.testing.assert_array_equal(np.sort(ranks[ids]),np.sort(ranks[wanted]))
            # Native selection merges the two zero signs. At a cutoff zero,
            # either member is legal; postprocessing still orders the selected
            # original IEEE scores and ids. Require that exact ordering too.
            assert np.all(keys[ids][:-1]>=keys[ids][1:])
            assert np.all(actual[row][len(ids):]==-1)

    expected=change(0)
    actual=select_topk(x,starts,lengths,k,valid_lengths=lengths)
    assert actual is not None
    verify(actual,expected)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):select_topk(x,starts,lengths,k,valid_lengths=lengths)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        out=select_topk(x,starts,lengths,k,valid_lengths=lengths)
    for turn in range(1,4):
        expected=change(turn);graph.replay()
        verify(out,expected)
