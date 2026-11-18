"""Real CPU dispatcher/cache lifecycle; no accelerator is initialized."""
from types import SimpleNamespace
import gc
import weakref
import sys

import pytest
import torch

from hcu_adapter.aiter_torch_library import _repair_library


@pytest.mark.parametrize("already_registered", [False, True])
def test_qualified_aiter_schema_cleanup(already_registered):
    library = torch.library.Library("aiter", "FRAGMENT")
    guard = SimpleNamespace(aiter_lib=library)
    name = "hcu_schema_existing" if already_registered else "hcu_schema_new"
    schema = f"aiter::{name}(Tensor x) -> Tensor"
    bad = f"aiter::aiter::{name}"
    if already_registered:
        library.define(schema)
    try:
        assert _repair_library(guard, torch) is library
        define = library.define
        assert _repair_library(guard, torch).define is define
        if not already_registered:
            library.define(schema)
        library.impl(name, lambda x: x + 1, "CPU")
        assert torch.equal(getattr(torch.ops.aiter, name)(torch.tensor([2])),
                           torch.tensor([3]))
        assert library._op_defs == {f"aiter::{name}"}
        assert bad not in torch.library._defs
    finally:
        library._destroy()
    assert name not in vars(torch.ops.aiter)
    reference = weakref.ref(library)
    del define, guard, library
    gc.collect()
    assert reference() is None
    assert f"aiter::{name}" not in torch.library._defs


def test_lazy_aiter_library_created_before_registration():
    guard = SimpleNamespace(aiter_lib=None)
    library = _repair_library(guard, torch)
    try:
        assert guard.aiter_lib is library
        library.define("hcu_schema_relative(Tensor x) -> Tensor")
        assert library._op_defs == {"aiter::hcu_schema_relative"}
    finally:
        library._destroy()


def test_packaged_and_bare_module_libraries_are_repaired(monkeypatch):
    from hcu_adapter import aiter_torch_library as compatibility

    modules = []
    for name in ("canonical", "alias"):
        library = torch.library.Library("aiter", "FRAGMENT")
        library.define(f"aiter::hcu_schema_{name}(Tensor x) -> Tensor")
        modules.append(SimpleNamespace(aiter_lib=library, __file__=__file__))
    monkeypatch.setattr(compatibility, "import_module", lambda _: modules[0])
    monkeypatch.setitem(sys.modules, "torch_guard", modules[1])
    try:
        compatibility.install()
        for module in modules:
            assert module.aiter_lib._hcu_relative_schemas
            assert all(name.count("::") == 1 for name in module.aiter_lib._op_defs)
    finally:
        for module in modules:
            module.aiter_lib._destroy()
