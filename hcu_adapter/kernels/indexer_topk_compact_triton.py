# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-only exact TopK for zero-based bounded decode windows.

Sort and deduplicate candidate blocks on device, gather their live logits into
a bounded score bank, and map native TopK output back to original columns.
Ascending block order preserves the native core's logical-ID tie ordering.
No GPU metadata is read on the host, including during graph replay.
"""
import torch
import triton
import triton.language as tl

from .indexer_topk_post_triton import _supported_device
from .indexer_topk_lightop import _topk_native_output


@triton.jit
def _compact_candidate_blocks(Candidates, Ends, Blocks, Lengths, Offsets,
                               WIDTH: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                               COUNT: tl.constexpr, TILE: tl.constexpr,
                               CROW: tl.constexpr, CCOL: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, TILE)
    end = tl.minimum(tl.load(Ends + row), WIDTH)
    nblocks = tl.cdiv(WIDTH, BLOCK_SIZE)
    candidate = tl.load(Candidates + row * CROW + lane * CCOL,
                        lane < COUNT, other=-1)
    candidate = tl.minimum(candidate, nblocks)
    normal = (candidate >= 0) & (candidate < tl.cdiv(end, BLOCK_SIZE))
    last_included = tl.sum(((lane < COUNT) & normal
                            & (candidate == nblocks - 1)).to(tl.int32), 0) > 0
    edge = ((candidate == nblocks) & (end == WIDTH) & ~last_included)
    valid = (lane < COUNT) & (normal | edge)
    ordered = tl.sort(tl.where(valid, candidate, 2147483647), descending=False)
    previous = tl.gather(ordered, tl.maximum(lane - 1, 0), 0)
    unique = (ordered != 2147483647) & ((lane == 0) | (ordered != previous))
    position = tl.cumsum(unique.to(tl.int32), 0) - 1
    tl.store(Blocks + row * COUNT + position, ordered, unique)
    tl.store(Lengths + row, tl.sum(unique.to(tl.int32), 0) * BLOCK_SIZE)
    tl.store(Offsets + row, 0)


@triton.jit
def _compact_topk_scores(X, Ends, Blocks, Lengths, Scores,
                         WIDTH: tl.constexpr, XROW: tl.constexpr,
                         COUNT: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                         PITCH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(Lengths + row)
    end = tl.minimum(tl.load(Ends + row), WIDTH)
    nblocks = tl.cdiv(WIDTH, BLOCK_SIZE)
    for tile in range(tl.program_id(1), tl.cdiv(length, TILE), tl.num_programs(1)):
        column = tile * TILE + tl.arange(0, TILE)
        block = tl.load(Blocks + row * COUNT + column // BLOCK_SIZE,
                        column < length, other=-1).to(tl.int64)
        offset = column % BLOCK_SIZE
        source = tl.where(block == nblocks, WIDTH - 1, block * BLOCK_SIZE + offset)
        valid = ((column < length) & (block >= 0) & (source < end)
                 & ((block != nblocks) | (offset == 0)))
        value = tl.load(X + row * XROW + source, valid, other=-float('inf'))
        bits = value.to(tl.uint32, bitcast=True)
        negative = (bits & 0x80000000) != 0
        zero = (bits & 0x7fffffff) == 0
        nan = (bits & 0x7fffffff) > 0x7f800000
        mapped = tl.where(nan, 0x7f800000,
                  tl.where(zero, 0x80000000, tl.where(negative, bits, bits - 1)))
        tl.store(Scores + row * PITCH + column,
                 mapped.to(tl.float32, bitcast=True), column < length)


@triton.jit
def _compact_blocks_and_short_output(X, Candidates, Ends, Blocks, Lengths,
                                     Offsets, Out,
                                     WIDTH: tl.constexpr, XROW: tl.constexpr,
                                     BLOCK_SIZE: tl.constexpr, COUNT: tl.constexpr,
                                     TILE: tl.constexpr, CROW: tl.constexpr,
                                     CCOL: tl.constexpr,
                                     K: tl.constexpr, SHORT_BINS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, TILE)
    end = tl.minimum(tl.load(Ends + row), WIDTH)
    nblocks = tl.cdiv(WIDTH, BLOCK_SIZE)
    candidate = tl.load(Candidates + row * CROW + lane * CCOL,
                        lane < COUNT, other=-1)
    candidate = tl.minimum(candidate, nblocks)
    tl.store(Offsets + row, 0)
    if end <= K:
        # Every live column fits in the output. Preserve ascending short-row
        # semantics directly, without sorting the much larger candidate bank
        # or invoking TopK on those rows. Membership remains exact, including
        # duplicate candidates and the native final-column edge sentinel.
        histogram = tl.histogram(candidate, SHORT_BINS,
            mask=(lane < COUNT) & (candidate >= 0) & (candidate < SHORT_BINS))
        col = tl.arange(0, K)
        admitted = tl.gather(histogram, col // BLOCK_SIZE, 0) != 0
        bins = tl.arange(0, SHORT_BINS)
        short_edge = tl.sum(tl.where(bins == nblocks, histogram, 0), 0) != 0
        admitted |= (col == WIDTH - 1) & short_edge
        short_value = tl.load(X + row * XROW + col, col < end, other=-float('inf'))
        short_valid = (col < end) & admitted & (short_value != -float('inf'))
        short_ordered = tl.sort(tl.where(short_valid, col, 2147483647))
        tl.store(Out + row * K + col, tl.where(short_ordered == 2147483647, -1, short_ordered))
        tl.store(Lengths + row, 0)
    else:
        normal = (candidate >= 0) & (candidate < tl.cdiv(end, BLOCK_SIZE))
        last_included = tl.sum(((lane < COUNT) & normal
                                & (candidate == nblocks - 1)).to(tl.int32), 0) > 0
        edge = (candidate == nblocks) & (end == WIDTH) & ~last_included
        ordered = tl.sort(tl.where((lane < COUNT) & (normal | edge),
                                   candidate, 2147483647))
        previous = tl.gather(ordered, tl.maximum(lane - 1, 0), 0)
        unique = (ordered != 2147483647) & ((lane == 0) | (ordered != previous))
        position = tl.cumsum(unique.to(tl.int32), 0) - 1
        tl.store(Blocks + row * COUNT + position, ordered, unique)
        tl.store(Lengths + row, tl.sum(unique.to(tl.int32), 0) * BLOCK_SIZE)


@triton.jit
def _compact_output_long_rows(X, Indices, Starts, Ends, Out, Blocks,
                               WIDTH: tl.constexpr, XROW: tl.constexpr,
                               K: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                               COUNT: tl.constexpr):
    if tl.minimum(tl.load(Ends + tl.program_id(0)), WIDTH) > K:
        _topk_native_output(X, Indices, Starts, Ends, Out, Blocks,
            WIDTH, XROW, K, False, tl.cdiv(WIDTH, BLOCK_SIZE), BLOCK_SIZE, True, COUNT)


def select_topk_compact(logits, starts, ends, k, *, candidates, block_size):
    """Select a zero-based live prefix, declining shapes without reduced work.

    The caller establishes that starts are zero from decode metadata. Prefill
    windows, including windows with nonzero starts, retain the original path.
    """
    if (torch.version.hip is None or not logits.is_cuda or logits.ndim != 2
            or logits.dtype != torch.float32 or logits.stride(1) != 1
            or logits.stride(0) < logits.shape[1] or k not in (512, 1024, 2048)
            or logits.shape[0] <= 0 or logits.shape[1] <= 0
            or logits.shape[1] > 2147483647
            or type(block_size) is not int or not 0 < block_size <= 128
            or starts.shape != logits.shape[:1] or ends.shape != starts.shape
            or starts.dtype != torch.int32 or ends.dtype != torch.int32
            or not starts.is_contiguous() or not ends.is_contiguous()
            or starts.device != logits.device or ends.device != logits.device
            or candidates.ndim != 2 or candidates.shape[0] != logits.shape[0]
            or not 0 < candidates.shape[1] <= 4096
            or candidates.dtype != torch.int32 or candidates.device != logits.device
            or any(s <= 0 for s in candidates.stride())
            or any(t.is_neg() for t in (logits, starts, ends, candidates))):
        return None
    rows, width = logits.shape
    count = candidates.shape[1]
    compact_width = count * block_size
    if compact_width >= width or not _supported_device(logits.device.index):
        return None
    sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
    if rows < triton.cdiv(sms, 16):
        # A handful of row-wise sorts leaves the device idle. The native full
        # prefix preparation distributes these rows over the available SMs.
        from .indexer_topk_lightop import select_topk
        return select_topk(logits, starts, ends, k, valid_lengths=ends,
                           candidates=candidates, block_size=block_size)
    try:
        from lightop.fuse_topk_transform import topk_transform_ragged_v2
    except ImportError:
        return None
    pitch = triton.cdiv(compact_width, 4) * 4
    blocks = torch.empty((rows, count), dtype=torch.int32, device=logits.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=logits.device)
    offsets = torch.empty_like(lengths)
    scores = torch.empty((rows, pitch), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
    out = torch.empty_like(indices)
    _compact_blocks_and_short_output[(rows,)](
        logits, candidates, ends, blocks, lengths, offsets, out,
        width, logits.stride(0), block_size, count, triton.next_power_of_2(count),
        *candidates.stride(), k, triton.next_power_of_2(triton.cdiv(k, block_size) + 1),
        num_warps=4)
    _compact_topk_scores[(rows, min(triton.cdiv(pitch, 2048), triton.cdiv(sms, rows)))](
        logits, ends, blocks, lengths, scores, width, logits.stride(0),
        count, block_size, pitch, 2048, num_warps=4)
    topk_transform_ragged_v2(scores, lengths, out_offsets=offsets, out_indices=indices)
    _compact_output_long_rows[(rows,)](
        logits, indices, starts, ends, out, blocks, width, logits.stride(0), k,
        block_size, count, num_warps=4)
    return out
