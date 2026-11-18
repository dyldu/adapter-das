# AITER schema cleanup with Torch 2.13

Torch 2.13 records schemas supplied to Library("aiter") by prefixing the library
namespace. The installed AITER passes qualified schemas such as
"aiter::mhc_post(...)". C++ registers the correct operator, but Python records
"aiter::aiter::mhc_post"; Torch's cache finalizer then raises ValueError while
splitting that name. This obscures the original worker error during shutdown.

The adapter normalizes schemas on AITER's own Library instance before define,
and repairs already recorded names in the sets retained by Torch's finalizer.
Other libraries and Torch's Library class stay unchanged. The installed AITER
loads the same source as both torch_guard and aiter.jit.utils.torch_guard; both
own libraries, so both must be repaired. The bare alias is admitted only when
its source path matches the packaged module. Missing AITER remains optional.

Validation environment: duyl-033-1112 on nmz18, Torch
2.13.0+dtk2610.2611091606.ge8aa84, AITER
0.1.8+dtk2610.torch2130.2611111922.g2c9e26. Four CPU cases cover existing/new
qualified schemas, CPU dispatch, cache destruction/finalization, lazy library
creation, idempotence and the two module identities. A separate real AITER
CPU decorator/call probe reports result=[3.0], malformed_defs=[],
cuda_initialized=false and exits without the cleanup exception.

Run:

    python3 -m pytest -q tests/test_aiter_torch_library.py \
      tests/adapter/test_vllm_bridge.py tests/adapter/test_vllm_runtime_abi.py

The guard-ordering fixture isolates provider registration and health checks,
which have separate contract tests. Before this fixture correction, the same
extra version-check failure reproduces on the unmodified35c397a baseline.

Serving validation is tracked separately in the DeepSeek-V4.1 Channel-FP8
experiment under /public/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118.
This cleanup fix changes registration bookkeeping; it makes no arithmetic or
kernel scheduling change. CPU evidence alone does not establish a serving pass.
