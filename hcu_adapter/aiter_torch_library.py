# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Keep AITER's qualified schemas compatible with Torch library cleanup."""
from __future__ import annotations

from functools import wraps
from importlib import import_module
from pathlib import Path
import sys


def _repair_library(guard, torch):
    library = guard.aiter_lib
    if library is None:
        library = torch.library.Library("aiter", "FRAGMENT")
        guard.aiter_lib = library
    if library.ns != "aiter":
        raise RuntimeError("AITER Torch library has an unexpected namespace")
    if getattr(library, "_hcu_relative_schemas", False):
        return library

    # AITER passes "aiter::op(...)" to Library("aiter"). C++ accepts it,
    # but Torch 2.13 records "aiter::aiter::op" in the Python cleanup sets.
    # Mutate the existing sets: Library's finalizer holds those same objects.
    for name in tuple(library._op_defs):
        if name.startswith("aiter::aiter::"):
            relative = name[len("aiter::"):]
            library._op_defs.remove(name)
            library._op_defs.add(relative)
            torch.library._defs.discard(name)
            torch.library._defs.add(relative)

    original = library.define

    @wraps(original)
    def define(schema, *args, **kwargs):
        if schema.startswith("aiter::"):
            schema = schema[len("aiter::"):]
        return original(schema, *args, **kwargs)

    library.define = define
    library._hcu_relative_schemas = True
    return library


def install() -> None:
    import torch

    try:
        guard = import_module("aiter.jit.utils.torch_guard")
    except ModuleNotFoundError as error:
        if error.name in ("aiter", "aiter.jit", "aiter.jit.utils",
                          "aiter.jit.utils.torch_guard"):
            return
        raise
    if hasattr(guard, "aiter_lib"):
        _repair_library(guard, torch)
    # The installed AITER also loads this file under the bare name
    # "torch_guard"; most decorators belong to that separate module/library.
    # Admit only the same source file, never an unrelated top-level module.
    alias = sys.modules.get("torch_guard")
    if (alias is not None and alias is not guard
            and getattr(alias, "__file__", None)
            and getattr(guard, "__file__", None)
            and Path(alias.__file__).resolve() == Path(guard.__file__).resolve()
            and hasattr(alias, "aiter_lib")):
        _repair_library(alias, torch)
