#!/usr/bin/env python3
"""Validate native selection ranks, exact unique-score indices and real MLA."""
import argparse
import importlib.util
import json
import random
import statistics
import traceback
from pathlib import Path

import torch


def run(args, report):
    from hcu_adapter.kernels.indexer_topk_lightop import select_topk
    spec=importlib.util.spec_from_file_location('native_topk_oracle',args.original)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    old=oracle.build()['local_topk']
    torch.manual_seed(112096)
    report.update(eager=[],graphs=[],consumer=[],timings=[])
    def save():Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    def keys(values):
        bits=values.contiguous().view(torch.int32).to(torch.int64)&0xffffffff
        ordered=torch.where((bits&0x80000000)!=0,(~bits)&0xffffffff,bits^0x80000000)
        ordered=torch.where(values==0,0x80000000,ordered)
        ordered=torch.where(torch.isnan(values),0xffffffff,ordered)
        return torch.where(values==-float('inf'),-1,ordered)
    for rows in (1,8,48,64):
        for width in (512,1024,4096,32768):
            backing=torch.empty((rows,width+17),device='cuda',dtype=torch.float32)
            x=backing[:,3:width+3]
            starts=torch.zeros((rows,),device='cuda',dtype=torch.int32)
            ends=torch.full_like(starts,width)
            columns=torch.arange(width,device='cuda')[None]
            def setup(pattern):
                starts.copy_(torch.arange(rows,device='cuda',dtype=torch.int32)%13)
                lens=torch.tensor([min(width,[0,1,511,512,513,width][i%6]) for i in range(rows)],device='cuda',dtype=torch.int32)
                ends.copy_(starts+lens)
                if pattern in ('unique','holes'):
                    x.copy_(torch.rand((rows,width),device='cuda').argsort(-1).float()-width//2)
                    x.masked_fill_(columns>=lens[:,None],-float('inf'))
                    if pattern=='holes':x[:,::3]=-float('inf')
                elif pattern=='ties':x.copy_((columns%5).float().expand_as(x))
                elif pattern=='masked':x.fill_(-float('inf'))
                elif pattern=='nonfinite':
                    x.normal_();x[:,::7]=float('nan');x[:,::13]=float('inf');x[:,::17]=-float('inf')
                else:
                    # Both signed zeros/subnormals,largest finite,Inf and
                    # signed signaling/quiet NaNs exercise the monotone map.
                    bits=torch.tensor([0,0x80000000,1,2,0x80000001,0x80000002,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xff800001],device='cuda',dtype=torch.int64).to(torch.int32)
                    x.copy_(bits.view(torch.float32).repeat((rows*width+14)//15)[:rows*width].reshape(rows,width))
            def check(result,pattern):
                assert result is not None and result.dtype==torch.int32 and result.shape==(rows,512)
                expected=old(x,starts,ends,512)
                if pattern in ('unique','holes','masked'):
                    assert torch.equal(result,expected),(rows,width,pattern,'unique indices')
                ids=result.long()+starts[:,None]
                valid=result!=-1
                assert torch.all((~valid)|((ids>=0)&(ids<width)))
                selected=x.gather(1,ids.clamp(0,width-1))
                actual_keys=torch.where(valid,keys(selected),-1).sort(-1,descending=True).values
                wanted_keys=keys(x.topk(512,dim=-1).values).sort(-1,descending=True).values
                assert torch.equal(actual_keys,wanted_keys),(rows,width,pattern,'score ranks')
                # Returned local indices can legitimately be negative when
                # callers choose a start beyond a selected global index. Keep
                # actual indexer inputs realistic here: scores before starts
                # are masked below, so -1 remains solely invalid padding.
                return dict(rows=rows,width=width,pattern=pattern,ranks_exact=True,indices_exact=pattern in ('unique','holes','masked'))
            for pattern in ('unique','holes','ties','nonfinite','edges','masked'):
                setup(pattern)
                # The global selection domain matches torch. Mask the prefix
                # just as the production score providers do before selection.
                x.masked_fill_(columns<starts[:,None],-float('inf'))
                saved=backing.view(torch.int32).clone()
                result=select_topk(x,starts,ends,512)
                record=check(result,pattern)
                assert torch.equal(backing.view(torch.int32),saved)
                report['eager'].append(record)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=select_topk(x,starts,ends,512)
            for pattern in ('unique','holes','ties','nonfinite','edges','masked'):
                setup(pattern);x.masked_fill_(columns<starts[:,None],-float('inf'));graph.replay()
                report['graphs'].append(check(result,pattern))
            del graph,result
            if rows in (8,48) and width in (4096,32768):
                setup('unique');x.masked_fill_(columns<starts[:,None],-float('inf'))
                gs={}
                for name,fn in [('torch',old),('native',select_topk)]:
                    fn(x,starts,ends,512);g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(8):fn(x,starts,ends,512)
                    gs[name]=g
                times={name:[] for name in gs}
                for repeat in range(3):
                    order=list(gs);random.Random(728+repeat).shuffle(order)
                    for name in order:
                        for _ in range(10):gs[name].replay()
                        a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
                        for _ in range(100):gs[name].replay()
                        b.record();b.synchronize();times[name].append(a.elapsed_time(b)*1000/800)
                report['timings'].append(dict(rows=rows,width=width,repeats_us=times,medians_us={k:statistics.median(v) for k,v in times.items()}))
                del gs
            save()
            del x,backing,starts,ends
            torch.cuda.empty_cache()
    from vllm import _custom_ops  # noqa:F401
    from flash_mla import flash_mla_interface as native
    from hcu_adapter.kernels.mla_sparse_decode_dtk import try_decode
    kv=torch.randn((1024,512),device='cuda',dtype=torch.bfloat16)
    cache=torch.zeros((16,64,584),device='cuda',dtype=torch.uint8)
    slots=torch.arange(1024,device='cuda');pos=slots.clone()
    angle=torch.randn((2048,32),device='cuda');cs=torch.cat((angle.cos(),angle.sin()),1)
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv,cache,slots,pos,cs,64,None,False)
    for rows in (8,48):
        q=torch.randn((rows,64,512),device='cuda',dtype=torch.bfloat16)
        starts=torch.zeros(rows,device='cuda',dtype=torch.int32);ends=torch.full_like(starts,1024)
        x=torch.rand((rows,1024),device='cuda').argsort(-1).float()
        for empty in (False,True):
            if empty:x.fill_(-float('inf'))
            a,b=old(x,starts,ends,512),select_topk(x,starts,ends,512)
            assert torch.equal(a,b)
            outputs=[]
            for ids in (a,b):
                out=torch.empty_like(q)
                count=(ids>=0).sum(-1,dtype=torch.int32)
                assert try_decode(q=q,kv_cache=cache,swa_k_cache=cache,swa_only=True,topk_indices=None,topk_lens=None,
                    swa_indices=ids,swa_lens=count,swa_ragged_indices=None,swa_ragged_indptr=None,topk_ragged_indices=None,topk_ragged_indptr=None,
                    attn_sink=None,scale=512**-.5,head_dim=512,nope_head_dim=448,rope_head_dim=64,output=out,
                    provider=native.flash_mla_with_kvcache,metadata_factory=native.get_mla_metadata,epilogue=None)==0
                outputs.append(out)
            assert torch.equal(outputs[0].view(torch.int16),outputs[1].view(torch.int16))
            assert torch.isfinite(outputs[1]).all()
            report['consumer'].append(dict(rows=rows,empty=empty,MLA_exact=True))
    report['passed']=True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',required=True);p.add_argument('--output',required=True);a=p.parse_args();r={'passed':False}
    try:run(a,r)
    except Exception:r['error']=traceback.format_exc();raise
    finally:Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
