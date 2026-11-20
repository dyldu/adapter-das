# SPDX-License-Identifier: Apache-2.0
"""Exact integer compaction/sort for the gfx938 sparse decode selector ABI."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def supported(dense, lengths, flat, indptr, rows, num_slots):
    if triton is None or rows <= 0 or not 0 <= num_slots <= 2147483647:
        return False
    if flat is not None or indptr is not None:
        if (flat is None or indptr is None or flat.ndim != 1
                or indptr.shape != (rows + 1,) or flat.numel() % rows):
            return False
        source, metadata = flat, indptr
        width = flat.numel() // rows
    else:
        if (dense is None or lengths is None or dense.ndim not in (2, 3)
                or dense.shape[0] != rows or lengths.shape != (rows,)
                or (dense.ndim == 3 and dense.shape[1] != 1)):
            return False
        source, metadata = dense, lengths
        width = dense.shape[-1]
    if (width > 2048 or source.dtype != torch.int32 or metadata.dtype != torch.int32
            or not source.is_cuda or metadata.device != source.device
            or not source.is_contiguous() or not metadata.is_contiguous()):
        return False
    return getattr(torch.cuda.get_device_properties(source.device), 'gcnArchName', '').split(':')[0] == 'gfx938'


if triton is not None:
    @triton.jit
    def _compact_selectors(Source, Metadata, Ids, Counts,
                           WIDTH: tl.constexpr, CAPACITY: tl.constexpr,
                           SLOTS: tl.constexpr, PADDED: tl.constexpr,
                           CSR: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK)
        if CSR:
            begin = tl.load(Metadata + row)
            end = tl.load(Metadata + row + 1)
            offsets = begin.to(tl.int64) + col
            live = (col < WIDTH) & (col < end - begin)
            load = live & (offsets >= 0) & (offsets < CAPACITY)
        else:
            length = tl.load(Metadata + row)
            offsets = row * WIDTH + col
            live = (col < WIDTH) & (col < length)
            load = live
        values = tl.load(Source + offsets, load, other=-1)
        valid = load & (values >= 0) & (values < SLOTS)
        count = tl.sum(valid.to(tl.int32), 0)
        keys = tl.where(valid, values, 2147483647)
        ordered = tl.sort(keys, descending=False)
        ids = tl.where(ordered == 2147483647, -1, ordered)
        tl.store(Ids + row * PADDED + col, ids, col < PADDED)
        tl.store(Counts + row, count)


def try_selectors(dense, lengths, flat, indptr, rows, num_slots):
    if not supported(dense, lengths, flat, indptr, rows, num_slots):
        return None
    csr = flat is not None
    source, metadata = (flat, indptr) if csr else (dense, lengths)
    width = source.numel() // rows
    padded = max(64, triton.cdiv(width, 64) * 64)
    block = triton.next_power_of_2(padded)
    ids = torch.empty((rows, 1, padded), dtype=torch.int32, device=source.device)
    counts = torch.empty((rows,), dtype=torch.int32, device=source.device)
    _compact_selectors[(rows,)](
        source, metadata, ids, counts, width, source.numel(), num_slots, padded,
        csr, block, num_warps=4 if block <= 1024 else 8)
    return ids, counts
