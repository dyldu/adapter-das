# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Small-row BF16 -> FP32 projection with cancellation refinement.

Independent short FP32 dot products avoid a long accumulation chain. Their
FP64 reduction is rounded once to FP32. Weights retain their checkpoint dtype.
The Plugin owns consumer admission and the native fallback.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(A, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr,
             BK: tl.constexpr):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.program_id(2)*SPLIT+tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for _ in range(SPLIT//BK):
        a=tl.load(A+rm[:,None].to(tl.int64)*K+rk[None,:],
                  (rm[:,None]<M)&(rk[None,:]<K),other=0)
        b=tl.load(W+rn[None,:].to(tl.int64)*K+rk[:,None],
                  (rn[None,:]<N)&(rk[:,None]<K),other=0)
        acc=tl.dot(a,b,acc)
        rk+=BK
    offsets=(tl.program_id(2).to(tl.int64)*M+rm[:,None])*N+rn[None,:]
    tl.store(P+offsets,acc,(rm[:,None]<M)&(rn[None,:]<N))


@triton.jit
def _reduce(P, Y, Refine, ELEMENTS: tl.constexpr, PARTS: tl.constexpr,
            BS: tl.constexpr, BLOCK: tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    part=tl.arange(0,BS)
    values=tl.load(P+part[:,None].to(tl.int64)*ELEMENTS+idx[None,:],
                   (part[:,None]<PARTS)&(idx[None,:]<ELEMENTS),other=0)
    values64=values.to(tl.float64)
    result=tl.sum(values64,axis=0)
    absolute=tl.sum(tl.abs(values64),axis=0)
    # Keep at least eight bits of cancellation headroom. Refine those
    # dot products directly; no device-dependent condition reaches the host.
    refine=(absolute>0)&(tl.abs(result)<=absolute/256)
    tl.store(Y+idx,result.to(tl.float32),idx<ELEMENTS)
    tl.store(Refine+idx,refine,idx<ELEMENTS)


@triton.jit
def _refine_cancelled(A, W, Y, Refine, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, BLOCK: tl.constexpr, BK: tl.constexpr,
                      FORCE: tl.constexpr=False):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    if FORCE:
        flags=offsets<M*N
    else:
        flags=tl.load(Refine+offsets,offsets<M*N,other=0)
    if tl.sum(flags.to(tl.int32),axis=0)>0:
        for lane in tl.static_range(BLOCK):
            out=tl.program_id(0)*BLOCK+lane
            if FORCE:
                flag=out<M*N
            else:
                flag=tl.load(Refine+out,out<M*N,other=0)
            if flag:
                row,col=out//N,out%N
                rk=tl.arange(0,BK)
                total=tl.full((),0,tl.float64)
                for part in range(tl.cdiv(K,BK)):
                    kk=part*BK+rk
                    a=tl.load(A+row.to(tl.int64)*K+kk,kk<K,other=0).to(tl.float64)
                    w=tl.load(W+col.to(tl.int64)*K+kk,kk<K,other=0).to(tl.float64)
                    total+=tl.sum(a*w,axis=0)
                tl.store(Y+out,total.to(tl.float32))


def supported(a,w):
    if (not isinstance(a,torch.Tensor) or not isinstance(w,torch.Tensor)
            or a.ndim!=2 or w.ndim!=2 or a.dtype!=torch.bfloat16
            or w.dtype!=a.dtype or a.device.type!='cuda' or w.device!=a.device
            or not a.is_contiguous() or not w.is_contiguous()
            or not 0<a.shape[0]<=64 or not 0<a.shape[1]<=16384
            or a.shape[1]!=w.shape[1] or a.shape[1]%32
            or not 0<w.shape[0]<=4096):
        return False
    props=torch.cuda.get_device_properties(a.device)
    return (torch.version.hip is not None
            and props.gcnArchName.split(':')[0]=='gfx938'
            and props.multi_processor_count==64)


def project(a,w):
    """Return FP32 [M,N]; caller must check the explicit metadata contract."""
    if not supported(a,w):
        raise ValueError('Unqualified BF16 auxiliary projection contract')
    m,k=a.shape;n=w.shape[0]
    if k<=128:
        # A single partial cannot reveal cancellation within its own dot.
        # Short reductions are inexpensive to evaluate directly in FP64.
        result=torch.empty((m,n),device=a.device,dtype=torch.float32)
        _refine_cancelled[(m*n,)](
            a,w,result,result,m,n,k,1,triton.next_power_of_2(k),True,num_warps=4)
        return result
    split=128;parts=triton.cdiv(k,split)
    partial=torch.empty((parts,m,n),device=a.device,dtype=torch.float32)
    result=torch.empty((m,n),device=a.device,dtype=torch.float32)
    refine=torch.empty((m,n),device=a.device,dtype=torch.bool)
    _partial[(triton.cdiv(m,16),triton.cdiv(n,64),parts)](
        a,w,partial,m,n,k,16,64,split,32,num_warps=4)
    _reduce[(triton.cdiv(m*n,128),)](
        partial,result,refine,m*n,parts,triton.next_power_of_2(parts),128,num_warps=4)
    _refine_cancelled[(triton.cdiv(m*n,16),)](
        a,w,result,refine,m,n,k,16,1024,num_warps=4)
    return result
