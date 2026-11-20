# W4A8 mHC Post and sparse-MLA selectors

The H5120 mHC Post fallback launches FP32 Torch products, a small-K batched
GEMM and BF16 conversion separately. Sparse-MLA decode similarly materializes
many Torch operations to validate, sort and pad integer selectors. This series
reuses the prior exact fusion implementations on HYGON main ef63bbb, above the
FP4 owner graph predicate 3bf4b68. The selected native MLA kernel and cache
format remain unchanged.

The Post path admits contiguous gfx938 BF16 x/residual, FP32 mixing tensors,
H5120, four residual streams and 1..1024 rows, with highest FP32 matmul
precision and CUDA autocast disabled. It preserves the measured native FP32 bmm
forward K-order and +0 accumulator with explicit FMA instructions; the separate
post product and final sum retain their rounding boundary before BF16 storage.
Unsupported layouts or precision settings take the existing implementation.

Selector fusion admits contiguous gfx938 INT32 metadata/storage, at most64
rows and2048 columns. It uses device CSR offsets, preserves duplicates, filters
invalid slots, sorts ascending and pads with -1. Unsupported cases, including
192/256 rows, use the pre-change Torch implementation. Fullgraph target verify
and native attention consumers are tested; no host metadata reads or persistent
attention-plan cache are introduced.

## Exact operator validation

Tested Adapter3b6ecb8f4bf138406e7ed68c0e72e59dd7b8d198 with plugin
f9af645a85040f26427e6f5f222632d3f92d9322, W4A8 checkpoint and the original
MoE expected-M scheduling, FULL_AND_PIECEWISE, FP4 tailmask and graph-admission
changes retained. Published documentation commits do not alter tested code.

- CPU:77 passed, no skips.
- Post:147 eager cases and50 changed-input graph replays; active BF16 bytes
  match the actual pre-change Torch implementation, including zero,
  cancellation and halfway cases and rows through1024.
- Selectors:480 eager and480 changed-metadata graph checks; exact IDs/counts,
  dense/CSR layouts, duplicate and invalid slots, zero width and fallback.
  Additional50 native MLA consumer checks use block64 and padded page strides,
  both cache paths, changed Q/KV/RoPE inputs and actual native outputs.
- The original selector fixture passed the integer cases but used a mismatched
  native insert block32 for a block64 cache. Its failure is retained;3b6ecb8
  fixes only that test fixture, with production code unchanged from f8060fa.
- Example standalone timings on this setup: Post rows40 162.9607->6.4719us;
  selector rows48/width128 128.4585->3.8313us. These are operator benchmarks,
  not service-level improvements.

## Real-KV model accuracy

The exact candidate completes HumanEval32 with32/32 and full164 with
159/164 (96.9512%), first32 32/32, workflow0, no API errors, missing outputs
or truncation. The same-node release baseline is157/164 (95.7317%). All
prompt hashes match. Candidate passes91/93/163 newly and loses97; remaining
candidate failures are32/83/97/132/145. The configured minimum is156/164.
Passing sampled exact operators and this score gate does not establish whole
model numerical equivalence; the changed generated-code outcomes remain
unexplained. Both related MRs remain drafts for review.

## Increment above MoE plus full graph on nmz18

Both sides use the same96GiB node, native2993-file vLLM hashes, package
versions, checkpoint, serving argv including FULL_AND_PIECEWISE, plugin and
workload. Only Adapter source changes from3bf4b68 to3b6ecb8. The minimum2%
incremental gain and median/p99 tolerances were frozen before candidate runs.

| Unprofiled client TPOT | MoE+Graph control | Add Post/selectors |
| --- | ---: | ---: |
| Repeat1 median | 52.1898ms | 38.3444ms |
| Repeat2 median | 52.1243ms | 38.1159ms |
| Mean of medians | 52.1571ms | 38.2301ms |
| Repeat1 p99 | 58.7738ms | 43.2175ms |
| Repeat2 p99 | 59.3973ms | 43.4676ms |

The combined Post/selector increment is26.70%; both repeats exceed2% and pass
median/p99 ceilings. Each completes64/64 with0 errors and actual4096 input /
128 output tokens. This measures DecodeBench fake/zero KV and synthetic
DSpark acceptance, with two repeats only. Real-KV production latency and
separate attribution to each of the two operators remain unmeasured.

All8 exports pass gzip CRC, and all8 ranks retain one FULL graph launch per
target span (80 masked GEMMs). DP0 confirms830 Post and780 selector calls
inside execution spans. Median profiled target wall119.0877->85.3593ms,
busy union116.3954->83.9415ms, gap2.7107->1.4062ms; draft plus sampling
wall9.6312->7.9817ms. These profiler durations are independent from the formal
TPOT pair and cannot be treated as unprofiled latency. The supplied SGLang
reference has different FP8 work, batch, hardware/memory and context evidence;
it provides phase guidance, not a controlled cross-framework speedup claim.

## Evidence and reproduction

Evidence root: run/dsv4.1/033/w4a8_reverify_20261119/.

- Frozen reuse protocols: POST_SELECTORS_REUSE_EXPERIMENT.json,
  POST_SELECTORS_REUSE_PROBE_AMENDMENT.json, REUSE_INCREMENTAL_EXPERIMENT.json.
- Raw probes: nmz18/reuse_mhc_post_5120.json,
  nmz16/reuse_mla_selectors.retry.json, nmz18/cpu_post_selectors_bridge.retry.log.
- HE32: nmz16/post_selectors_he32/20261119_144416/.
- Full164: nmz16/post_selectors_he164/20261119_145348/;
  nmz16/post_selectors_full164.gate.json includes all outcomes.
- Graph control: nmz18/fullgraph_control_performance/20261119_143559_prof_w4_PcmJrt/.
- Candidate: nmz18/post_selectors_performance/20261119_145143_prof_w4_lc8gZf/;
  nmz18/post_selectors_incremental.audit.json;
  nmz18/post_selectors_all_ranks_trace.audit.json.
- Latest DP0 trace in that candidate directory:
  torch_profiler/dp0_pp0_tp0_dcp0_ep0_rank0.1795071704034911327.pt.trace.json.gz.

Run only on an authorized idle node, using the documented existing image in
container duyl-033-w4a8-1119. The immutable accuracy_w4a8.sh and
profile_w4a8_paired.sh retain the complete environment and command snapshots
with each result. Set W4_NODE, W4_RUN_ROLE=candidate,
HCU_VLLM_DIR=/duyl/vllm-hcu033-wt-w4a8-fullgraph-20261119,
HCU_ADAPTER_SRC=/duyl/adapter-das-wt-w4a8-post-selectors-20261119,
CUDAGRAPH_MODE=FULL_AND_PIECEWISE, RUN_ROOT to a new directory, and
HE_LIMIT=32 or164 for the accuracy harness. Frozen model/capacity arguments
remain TP1/DP8/EP, deep_gemm/deepep_low_latency, utilization0.85, block64,
max-seqs32, max-model-len32768, max-batched-tokens256 and capture buckets
1/4/8/16/32/40/48/64. Draft still computes with DeepGEMM using AG/RS transport;
no W4A8 draft LL/OOM fix has been implemented or measured in this series.

## User-requested pause and publication

The follow-up CI fix renames the two implementation modules to
`mhc_post_5120_triton.py` and `mla_selectors_triton.py`, as required by the
repository's operator-tier naming rules. Imports, probe imports and the mHC
source-fingerprint path follow the new names. Both implementation files retain
identical source bytes. The GPU results above refer to the pre-rename source;
no additional GPU run was started for this module-naming fix.

The user stopped nmz16 immediately during its separate composition-vs-release
performance run, then explicitly requested pushing the verified commits.
That run has formal medians38.1442/37.9644ms but no profile exports or completed
workflow exit; it is interrupted evidence and is excluded from accepted
performance claims. The nmz18 incremental experiment above completed both
formal repeats, all8 exports and workflow0 before the stop request. Its
acceptance is independent of the interrupted nmz16 performance run. The
full164 and HE32 accuracy runs on nmz16 completed before this stop.

No subsequent DSpark GPU probe or serving experiment was started. This series
is pushed to the existing Adapter MR1 with the user's explicit instruction;
no incomplete nmz16 workflow is represented as having passed its release SLA.

## Review follow-up: CUDA autocast admission

The review on Adapter MR1 identified that highest FP32 matmul precision does
not disable CUDA autocast. Under autocast the Torch reference can select a
different bmm precision, while the fused kernel keeps its explicit FP32 FMA.
Admission now rejects CUDA autocast and lets the existing runtime call the
Torch reference with the caller's autocast state intact. The Triton compute
body is unchanged, and the measured serving scripts do not enable autocast.

CPU-only FakeTensor regression tests exercise real CUDA autocast thread state
for BF16 and FP16, check fallback to the original Torch entry, and retain a
positive admission case with autocast disabled. Both autocast cases fail
before the guard and pass after it without CUDA initialization. The Post
contract suite is included in the CPU workflow with its Triton import
dependency. No GPU benchmark or HumanEval run was resumed for this fix.
