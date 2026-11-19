# SPDX-License-Identifier: Apache-2.0
"""Admission for the byte-exact delayed H5120/HC4 stage2 specialization."""
from functools import lru_cache
import hashlib
from importlib import import_module
import inspect
from pathlib import Path
from types import CodeType

import torch

_NATIVE_SHA256 = '2ac06a653f02491b6c319d1e46cebcb9fd64081b8499fb2763c02c2ff9bf50a9'


class _Identity:
    """Keep an unhashable TileLang JIT object alive in an identity cache key."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        return isinstance(other, _Identity) and self.value is other.value


@lru_cache(maxsize=4)
def _native_matches(function, code, factory_key):
    """Only replace the native arithmetic qualified by the numerical probe.

    The wrapper and JIT factory identities are part of the key: replacing
    either symbol cannot inherit a cached admission for an earlier provider.
    """
    if not inspect.isfunction(function):
        return False
    name = 'aiter.ops.tilelang.mhc.pre_big_fuse_kernel'
    if function.__module__ != name:
        return False
    module = import_module(name)
    if (function is not getattr(module, 'pre_big_fuse_tilelang', None)
            or factory_key.value is not getattr(module, '_mhc_pre_big_fuse', None)):
        return False
    path = inspect.getsourcefile(function)
    if path is None:
        return False
    try:
        source = Path(path).read_bytes()
    except OSError:
        return False
    if hashlib.sha256(source).hexdigest() != _NATIVE_SHA256:
        return False
    # Also reject a changed live wrapper even if its on-disk file is intact.
    definitions = compile(source, code.co_filename, 'exec').co_consts
    return any(isinstance(value, CodeType) and value.co_name == function.__name__
               and value == code for value in definitions)


def supported(provider, residual, padded, sqrsum, scale, base, splits, x=None):
    """Reject unknown layouts/providers before importing or launching TileLang."""
    tensors = (residual, padded, sqrsum, scale, base)
    if (x is not None or not provider.tilelang_enabled
            or not all(isinstance(t, torch.Tensor) for t in tensors)
            or residual.ndim != 3 or residual.shape[1:] != (4, 5120)
            or not 0 < residual.shape[0] <= 64 or splits not in (1, 32)
            or padded.shape != (splits, residual.shape[0], 32)
            or sqrsum.shape != (splits, residual.shape[0])
            or scale.shape != (3,) or base.shape != (24,)
            or residual.dtype != torch.bfloat16
            or any(t.dtype != torch.float32 for t in tensors[1:])
            or not all(t.is_cuda and t.device == residual.device and t.is_contiguous()
                       for t in tensors)
            or torch.is_autocast_enabled('cuda')
            or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))):
        return False
    props = torch.cuda.get_device_properties(residual.device)
    if (getattr(props, 'gcnArchName', '').split(':')[0] != 'gfx938'
            or getattr(props, 'multi_processor_count', None) != 64):
        return False
    function = provider.tilelang_pre
    if not inspect.isfunction(function):
        return False
    return _native_matches(function, function.__code__,
                           _Identity(function.__globals__.get('_mhc_pre_big_fuse')))


def run(padded, sqrsum, scale, base, post, comb, rms_eps, pre_eps,
        sinkhorn_eps, post_mult, repeat):
    from .mhc_delayed_stage2_tilelang import post_comb_only
    post_comb_only(padded, sqrsum, scale, base, post.squeeze(-1),
                   comb.flatten(1), rms_eps=rms_eps, pre_eps=pre_eps,
                   sinkhorn_eps=sinkhorn_eps, post_mult=post_mult, repeat=repeat)
