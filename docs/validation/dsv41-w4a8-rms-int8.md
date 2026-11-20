# Owned W4A8 RMS / INT8 handoff validation

Paired with Plugin MR8, the Adapter fuses the native attention RMS reduction,
BF16 storage rounding and dynamic INT8 quantization. It preserves the native
CT weight loader/GEMM and the BF16 value used by the indexer. INT8 codes and
FP32 scales are scoped to one decoder call and looked up by exact tensor
identity. Unsupported inputs use the existing norm/quantization path.

Admission is target attention only: aligned contiguous BF16, M=1..64,
H5120, epsilon1e-20, gfx938/64CU, native RMS and symmetric dynamic channel CT
INT8. The Plugin excludes autocast, gradient mode, compilation rewrites and
unsupported decoder/consumer contracts. Draft norms are unchanged. No serving
flag or feature environment variable is added.

Tested clean Plugin `f1d530c` + Adapter `56206d3`: 2,752 normalized real-weight
cases,544 edge cases,256 native QKV consumers,256 changing-input graph replays,
64 BF16 consumers and20 fallbacks pass byte-exact BF16/INT8/FP32 checks.
The paired CPU suite passes53 tests; local L0 is green with30 existing
warnings. M48 norm/quant/native-QKV median24.2127 to20.4720 us (15.45%).

Original real-prefill/rejection HumanEval passes159/164 (96.95%), first32 all;
the profiler-only repeat passes160/164. A batch64 coverage run initially
scored83/164 with78 execution timeouts. Official EvalScope cached review of
those same164 answers with four review workers passes159/164, first32 all,
zero timeouts; the four-second program limit is unchanged. The prediction
SHA256 is `dfa2b4965c1e6435acf4c886edae137c7aad718309a1123a7c063e0410c43b66`.
No new generation/GPU work is performed; all164 pass/fail outcomes match the
same candidate's original batch16 run. The original result is retained.
EvalScope1.9.1's eval_batch_size also controls review concurrency; deprecated
judge_worker_num did not limit it independently. The paired Plugin provides
the review-only repair tool and detailed evidence.

All eight original accuracy traces show40 target fused kernels per complete
span; native RMS41/quant170, retained mHC/MLA/FP4/MoE/DSpark paths are verified.
Supplementary batch64 sampled spans also cover the RMS and WO_A paths.
This is fixed-workload evidence, not a complete concurrency/context sweep.

Two unprofiled64-request repeats: TPOT medians17.8993/17.3512 ms,
p99s20.9428/21.5518 ms, zero request failures, frozen regression limits pass.
Mean median17.6252 versus17.7456 ms differs0.68%; >=2% gain in both repeats
is not established. Independent profiled complete step71.0260 versus71.3833 ms.
Performance uses fake zero KV and five literal unconditional0.6 acceptance
rates; accuracy uses real prefill/rejection. Server capacities are unchanged.

Workspace evidence: `RMS_INT8_REJUDGED_ACCEPTANCE.json`,
`RMS_INT8_UNIT_RESULTS.json`,
`nmz18/rms_int8_owned_service_20261120/acceptance.json`,
`nmz18/rms_int8_owned_accuracy_dispatch_20261120_v2/acceptance.json` and
`nmz18/batch64_evalscope_rejudge_20261120/rejudge_summary.json`, under
`run/dsv4.1/033/w4a8_reverify_20261119`.
