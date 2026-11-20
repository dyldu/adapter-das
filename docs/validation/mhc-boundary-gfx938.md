# Fused mHC boundary on gfx938

The vLLM delayed mHC entry previously launched Post, projection and collapsed
input work separately. The owned `hcu_adapter_mhc_boundary` entry combines Post,
the previous pre-mix collapse, and projection/RMS partials in one Triton kernel,
then reduces the partials and runs Sinkhorn in a second kernel. The existing
public IR RMSNorm still normalizes the collapsed input.

The idea and the reduction kernel derive from SGLang's Apache-2.0
`python/sglang/kernels/ops/layernorm/mhc_boundary_hip.py`, source SHA256
`58ef20e278484a8d8b47f522aa742b8b1757c0fe5bf3e574e100f1bdd2b6f602`.
This implementation preserves the current vLLM Post's zero-initialized forward
FMA order, separately rounded FP32 post product/addition, and BF16 store before
projection and collapse. Those rounding boundaries differ from the source.

Admission requires gfx938 with 64 CUs, inference without autocast, highest FP32
matmul precision, contiguous tensors on one device, BF16 residual/output,
FP32 projection/mixes, HC=4, H=5120, M=1..64, Post multiplier 2,
RMS epsilon 1e-20 or 1e-6, pre/Sinkhorn epsilon 1e-6, 20 Sinkhorn iterations,
and the existing automatic split policy. The W4A8 checkpoint uses RMS epsilon
1e-20. It is forwarded separately from the pre/Sinkhorn epsilon.
Direct-x calls, other shapes/scalars and explicit split overrides retain their
existing entry. No launch option or environment variable is added.

## Operator validation

On nmz18 in `duyl-033-w4a8-1119`, the expanded probe covered all 80 target and 6
draft attention/FFN weights, M=1/4/8/16/32/40/48/64, all four Post/previous-mix
nullable combinations, six magnitudes, and changing inputs and weights during
graph replay. All 16,512 eager and 8,256 graph checks passed. Updated residual
and collapsed input are byte exact BF16; FP32 mixes meet the existing
rtol=2e-5, atol=2e-6 contract. The probe reads RMS epsilon 1e-20 and HC epsilon
1e-6 from the actual checkpoint config; Post multiplier is 2.

The registered owner passed 24 eager and 24 changing-input graph checks,
including exact comparison with the same public IR RMSNorm as the existing
runtime. Its five-output fake schema also passed a fullgraph `aot_eager`
compile. CPU contract/bridge regressions: 82 passed. L0: GREEN, 30 existing
warnings. Initial RMS probe used a different private custom-op oracle; the
corrected probe uses the production public IR operation without changing the
runtime or tolerance. The initial 1e-6 RMS scalar matrix is retained as a
diagnostic. Its first full-model launch was stopped before evaluation because
that guard would retain the previous path. The corrected matrix above and the
full-service result below use the checkpoint's actual 1e-20 RMS epsilon.

Complete operator medians, native versus fused, in microseconds:

| M | Native | Fused |
|---|---:|---:|
| 8 | 33.1512 | 20.2304 |
| 48 | 42.7720 | 30.0316 |
| 64 | 46.7296 | 35.6760 |

## Full service validation

The immutable tested composition is Adapter `b191b7b` and Plugin `b50d7e3`.
The full workflow exited 0. Real-prefill/rejection HumanEval passed 159/164
(96.95%), first32 32/32, with zero errors, missing outputs or truncations.

Two unprofiled performance repeats each completed 64 requests with zero
failures. TPOT medians were 18.3430 / 17.4224 ms, p99 21.3137 / 21.6903 ms.
Mean of medians is 17.8827 ms versus the preceding accepted 18.1628 ms, a 1.54%
reduction. The frozen median +2%, p99 +5% and mean +2% regression ceilings
passed. The separate >=2%-gain-in-each-repeat claim did not pass.

All eight traces pass gzip CRC and actual dispatch checks: each target step
uses 77 fused boundaries plus three native entry/Engram seams; DSpark uses five
fused boundaries plus its native entry. Separate Post calls reduce from 83 to
6 in target and from 6 to 1 in draft. The Engram injection boundary remains
intact. FULL target execution, 40/3 DeepEP and activation layers, FP4 staging,
native exact-rank Top512 and WO_A output ownership are retained.

DP0 target-through-draft-end median is 71.7187 ms (target 65.3704, draft 6.3174),
versus the preceding 73.7678 ms: 2.0491 ms / 2.78% lower. This profiled step
measurement is separate from unprofiled TPOT. The supplied SGLang reference is
65.0983 ms, so a 6.6205 ms gap remains. Hardware memory, context, per-rank batch
and quantized providers differ; this is a stage reference, not a controlled
head-to-head result. Joint mHC/norm kernel duration sums also do not establish
isolated critical-path parity.

Serving arguments and recorded environment match the preceding accepted run.
Performance retains fake zero KV, five literal unconditional 0.6 acceptance
rates, DSpark5, 4096 input / 128 output, 64 requests/concurrency and eight
warmups. Accuracy uses real prefill without synthetic acceptance. Evidence is
in workspace `nmz18/mhc_boundary_native_rms_service_20261120/acceptance.json`
and `MHC_BOUNDARY_RESULTS.json`. Documentation follow-ups do not alter the
tested runtime or tests.

Reproduce the operator gates with `tests/tools/probe_mhc_boundary_expanded.py
--model <checkpoint> --output <new-result.json> --post-mult 2` and
`tests/tools/probe_mhc_boundary_registered.py --model <checkpoint>
--output <new-result.json>`
in the pinned v0.33 provider environment. Preserve the expanded probe's
real-weight paths and the test environment recorded in the experiment manifest.
