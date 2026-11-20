#!/usr/bin/env python3
"""Byte-exact indexer postprocessing and changing-graph/full-wrapper timing."""
import argparse
import importlib.util
import json
import random
import statistics
import traceback
from pathlib import Path
from unittest import mock

import torch


def run(args, report):
    from hcu_adapter.v41_fp8_indexer import build
    from hcu_adapter.kernels import indexer_topk_post_triton as post_module
    finish_topk = post_module.finish_topk
    spec = importlib.util.spec_from_file_location('topk_before_fusion', args.original)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    old, new = baseline.build()['local_topk'], build()['local_topk']
    torch.manual_seed(8720)
    report.update(eager=[], graphs=[], timings=[], refusal=[], oracle_sha256=__import__('hashlib').sha256(Path(args.original).read_bytes()).hexdigest())
    def save(): Path(args.output).write_text(json.dumps(report, indent=2)+'\n')

    for rows in args.rows:
        for width in (512, 1024, 4096, 32768):
            # An offset, padded outer stride checks that torch-owned topk output,
            # rather than input-layout assumptions, defines the fused ABI.
            backing = torch.empty((rows, width+19), dtype=torch.float32, device='cuda')
            logits = backing[:, 7:7+width]
            starts = torch.empty(rows, dtype=torch.int32, device='cuda')
            ends = torch.empty_like(starts)
            columns = torch.arange(width, device='cuda')[None]
            def setup(pattern):
                logits.normal_()
                starts.copy_(torch.arange(rows, device='cuda', dtype=torch.int32) % 17)
                lengths = torch.tensor([min(width-int(i%17), [0, 1, 511, 512, 513, width][i%6]) for i in range(rows)], dtype=torch.int32, device='cuda')
                ends.copy_(starts+lengths)
                logits.masked_fill_((columns<starts[:,None]) | (columns>=ends[:,None]), -float('inf'))
                if pattern == 'holes': logits[:, ::3] = -float('inf')
                elif pattern == 'ties': logits.masked_fill_(torch.isfinite(logits), 1.)
                elif pattern == 'nonfinite':
                    logits[:, ::7] = float('nan'); logits[:, ::13] = float('inf')
                elif pattern == 'allmasked': logits.fill_(-float('inf'))
            def post_reference(v, i):
                invalid = v == -float('inf')
                selected = (i-starts[:,None]).to(torch.int32).masked_fill(invalid, -1)
                ordered = i.masked_fill(invalid, width).sort(-1).values
                shifted = (ordered-starts[:,None]).to(torch.int32).masked_fill(ordered==width,-1)
                return torch.where((ends-starts<=512)[:,None], shifted, selected)
            for pattern in ('random', 'holes', 'ties', 'nonfinite', 'allmasked'):
                setup(pattern)
                before = backing.view(torch.int32).clone()
                top = logits.topk(512, dim=-1)
                assert torch.equal(post_reference(top.values,top.indices),finish_topk(top.values,top.indices,starts,ends,512,width)), (rows,width,pattern,'post')
                # Native multiblock top-k may pick different tied NaNs across
                # identical calls. Compare the transformation of the SAME
                # native selection, retaining its exact order and membership.
                with mock.patch.object(torch.Tensor,'topk',return_value=top):
                    assert torch.equal(old(logits,starts,ends,512),new(logits,starts,ends,512)), (rows,width,pattern,'wrapper same selection')
                assert torch.equal(backing.view(torch.int32),before), 'source changed'
                report['eager'].append(dict(rows=rows,width=width,pattern=pattern,exact=True))
            setup('random'); new(logits,starts,ends,512)
            g=torch.cuda.CUDAGraph()
            selected={}
            def retain_selection(values,indices,*a):
                selected.update(values=values,indices=indices)
                return finish_topk(values,indices,*a)
            # Retain references to actual torch top-k outputs inside the full
            # graph. No extra GPU operation, selection or data copy is added.
            with mock.patch.object(post_module,'finish_topk',side_effect=retain_selection):
                with torch.cuda.graph(g): result=new(logits,starts,ends,512)
            for pattern in ('holes','random','nonfinite','ties','allmasked'):
                setup(pattern); starts.add_(1); ends.sub_(1)
                g.replay()
                assert torch.equal(result,post_reference(selected['values'],selected['indices'])), (rows,width,pattern,'changing graph same selection')
                report['graphs'].append(dict(rows=rows,width=width,pattern=pattern,exact=True))
            del g,result,selected
            if rows in (8,48) and width in (1024,4096,32768):
                setup('random')
                gs={}
                for name,fn in [('old',old),('new',new)]:
                    fn(logits,starts,ends,512)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(8): fn(logits,starts,ends,512)
                    gs[name]=graph
                measurements={'old':[],'new':[]}
                for repeat in range(3):
                    order=['old','new']; random.Random(127+repeat).shuffle(order)
                    for name in order:
                        for _ in range(10):gs[name].replay()
                        start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        for _ in range(100):gs[name].replay()
                        end.record();end.synchronize()
                        measurements[name].append(start.elapsed_time(end)*1000/800)
                report['timings'].append(dict(rows=rows,width=width,repeats_us=measurements,medians_us={k:statistics.median(v) for k,v in measurements.items()}))
                del gs
            save()
            del backing,logits,starts,ends
            torch.cuda.empty_cache()
    # Unsupported K is still handled by the original torch branch.
    for rows,width,k in ((0,512,512),(8,0,512),(8,127,512),(8,1024,128),(8,1024,1024)):
        x=torch.randn((rows,width),device='cuda');starts=torch.zeros(rows,dtype=torch.int32,device='cuda');ends=torch.full_like(starts,width)
        assert torch.equal(old(x,starts,ends,k),new(x,starts,ends,k))
        report['refusal'].append(dict(rows=rows,width=width,k=k,exact=True))
    report['passed']=True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',required=True);p.add_argument('--output',required=True);p.add_argument('--rows',type=int,nargs='+',default=[1,8,48,64]);args=p.parse_args()
    report={'passed':False}
    # Isolate postprocessing; native selection has its own rank/MLA probe.
    try:
        with mock.patch('hcu_adapter.kernels.indexer_topk_lightop.select_topk',return_value=None):
            run(args,report)
    except Exception:
        report['error']=traceback.format_exc();raise
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
