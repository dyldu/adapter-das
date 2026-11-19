# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Experimental delayed mHC stage2; not installed in serving.

Adapted from aiter/ops/tilelang/mhc/pre_big_fuse_kernel.py, SHA256
2ac06a653f02491b6c319d1e46cebcb9fd64081b8499fb2763c02c2ff9bf50a9.
Keep the exact native Post/Sinkhorn arithmetic and128-thread reduction layout.
Read stage1's32-column pitch directly and omit its unused current-Pre collapse.
Delayed serving consumes a collapse using the previous Pre mix instead.
"""
import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def _delayed_post_comb_5120(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, 32), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
    ) -> None:
        threads = 128

        n_splits_aligned = tilelang.math.next_power_of_2(n_splits)

        if n_splits >= 4:
            split_groups = threads // 32
            # assert n_splits % split_groups == 0
            group_rows = n_splits // split_groups
        with T.Kernel(num_tokens, threads=threads) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            tx = T.get_thread_binding()
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            rms = T.alloc_fragment(1, T.float32)

            if n_splits >= 4 and n_splits % split_groups == 0:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                mixes_pre = T.alloc_fragment((split_groups, 32), T.float32)
                mixes_aligned = T.alloc_fragment(32, T.float32)
                T.clear(mixes_pre)
                for r in T.serial(group_rows):
                    for i, j in T.Parallel(split_groups, 32):
                        if j < mhc_mult3:
                            mixes_pre[i, j] += gemm_out_mul[i * group_rows + r, pid, j]
                T.reduce_sum(mixes_pre, mixes_aligned, dim=0)
                for i in T.Parallel(32):
                    if i < mhc_mult3:
                        mixes_shared[i] = mixes_aligned[i] * rms[0]
            elif n_splits >= 2:
                sqrsum = T.alloc_fragment(n_splits_aligned, T.float32)
                T.copy(gemm_out_sqrsum[:, pid], sqrsum)
                T.reduce_sum(sqrsum, rms)
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)
            else:
                rms[0] = gemm_out_sqrsum[0, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = gemm_out_mul[0, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if tx < 64:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2] + mhc_base[j * mhc_mult + k + mhc_mult * 2]

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
    return mhc_pre_big_fuse


def post_comb_only(padded, sqrsum, scale, base, post, comb, *, rms_eps,
                   pre_eps, sinkhorn_eps, post_mult, repeat):
    """Write only Post/comb mixes; this API has no uninitialized layer output."""
    if padded.ndim != 3:
        raise ValueError('delayed stage2 requires padded split/token/32 data')
    splits, tokens, width = padded.shape
    tensors = (padded, sqrsum, scale, base, post, comb)
    if (splits not in (1, 32) or not 0 < tokens <= 64 or width != 32
            or sqrsum.shape != (splits, tokens) or scale.shape != (3,)
            or base.shape != (24,) or post.shape != (tokens, 4)
            or comb.shape != (tokens, 16)
            or any(t.dtype != torch.float32 or not t.is_cuda
                   or not t.is_contiguous() or t.device != padded.device
                   for t in tensors)):
        raise ValueError('delayed stage2 requires native H5120/HC4 FP32 pitches')
    _delayed_post_comb_5120(5120, rms_eps, pre_eps, sinkhorn_eps, post_mult,
                           repeat, n_splits=splits, mhc_mult=4)(
        padded, sqrsum, scale, base, post, comb)
