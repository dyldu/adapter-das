# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decode candidate producer: block maximum and native exact-rank TopK.

Fuse block scoring, causal bounds, newest-block pinning and IEEE key mapping.
Only live blocks enter the ragged native selection. Candidate order is free;
equal-score membership follows the native exact-rank contract, as in Top512.
Other window and provider contracts retain the original implementation.
"""
import torch
import triton
import triton.language as tl

from .indexer_topk_post_triton import _supported_device


@triton.jit
def _maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_native_scores(X, Ends, Scores, Lengths, Offsets,
                              WIDTH: tl.constexpr, XROW: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr, PITCH: tl.constexpr,
                              TILE: tl.constexpr, INNER: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    end = tl.load(Ends + row)
    visible = tl.maximum(tl.minimum(end, WIDTH), 0)
    blocks = tl.cdiv(visible, BLOCK_SIZE)
    if tl.program_id(1) == 0:
        tl.store(Lengths + row, blocks)
        tl.store(Offsets + row, 0)
    offset = tl.arange(0, INNER)
    for tile in range(tl.program_id(1), tl.cdiv(blocks, TILE), tl.num_programs(1)):
        block = tile * TILE + tl.arange(0, TILE)
        column = block[:, None] * BLOCK_SIZE + offset[None, :]
        values = tl.load(X + row * XROW + column,
                         (offset[None, :] < BLOCK_SIZE) & (column < visible),
                         other=-float("inf"))
        value = tl.reduce(values, 1, _maximum_nan)
        value = tl.where((end > 0) & (block == (end - 1) // BLOCK_SIZE), float("inf"), value)
        bits = value.to(tl.uint32, bitcast=True)
        nan = (bits & 0x7fffffff) > 0x7f800000
        zero = (bits & 0x7fffffff) == 0
        mapped = tl.where(nan, 0x7f800000,
                          tl.where(zero, 0x80000000,
                                   tl.where((bits & 0x80000000) != 0, bits, bits - 1)))
        tl.store(Scores + row * PITCH + block, mapped.to(tl.float32, bitcast=True), block < blocks)


@triton.jit
def _candidate_native_output(Scores, Lengths, Selected, Out,
                              PITCH: tl.constexpr, K: tl.constexpr,
                              OROW: tl.constexpr, OCOL: tl.constexpr,
                              TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * TILE + tl.arange(0, TILE)
    index = tl.load(Selected + row * K + column, column < K, other=-1)
    length = tl.load(Lengths + row)
    valid = (index >= 0) & (index < length)
    value = tl.load(Scores + row * PITCH + index, valid, other=-float("inf"))
    tl.store(Out + row * OROW + column * OCOL,
             tl.where(valid & (value != -float("inf")), index, -1), column < K)


def select_candidate_blocks_native(logits, ends, out, block_size):
    """Zero-based decode producer; decline other ABI/shape families."""
    if (torch.version.hip is None or not logits.is_cuda or logits.ndim != 2
            or logits.dtype != torch.float32 or logits.stride(1) != 1
            or logits.shape[0] <= 0 or logits.shape[1] <= 0
            or logits.shape[1] > 2147483647 or logits.stride(0) < logits.shape[1]
            or type(block_size) is not int or not 0 < block_size <= 128
            or ends.shape != logits.shape[:1] or ends.dtype != torch.int32
            or ends.device != logits.device or not ends.is_contiguous()
            or out.ndim != 2 or out.shape[0] != logits.shape[0]
            or out.shape[1] not in (512, 1024, 2048) or out.dtype != torch.int32
            or out.device != logits.device or any(s <= 0 for s in out.stride())
            or out.stride(0) < (out.shape[1] - 1) * out.stride(1) + 1
            or any(t.is_neg() for t in (logits, ends, out))):
        return False
    if not _supported_device(logits.device.index):
        return False
    try:
        from lightop.fuse_topk_transform import topk_transform_ragged_v2
    except ImportError:
        return False
    rows, width = logits.shape
    blocks = triton.cdiv(width, block_size)
    pitch = triton.cdiv(blocks, 4) * 4
    scores = torch.empty((rows, pitch), device=logits.device, dtype=torch.float32)
    lengths = torch.empty(rows, device=logits.device, dtype=torch.int32)
    offsets = torch.empty_like(lengths)
    indices = torch.empty((rows, out.shape[1]), device=logits.device, dtype=torch.int32)
    sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
    _candidate_native_scores[(rows, min(triton.cdiv(blocks, 128), triton.cdiv(sms, rows)))](
        logits, ends, scores, lengths, offsets, width, logits.stride(0), block_size,
        pitch, 128, triton.next_power_of_2(block_size), num_warps=4)
    topk_transform_ragged_v2(scores, lengths, out_offsets=offsets, out_indices=indices)
    _candidate_native_output[(rows, triton.cdiv(out.shape[1], 256))](
        scores, lengths, indices, out, pitch, out.shape[1], *out.stride(), 256, num_warps=4)
    return True
