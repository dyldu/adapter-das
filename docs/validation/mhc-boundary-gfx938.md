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
FP32 projection/mixes, HC=4, H=5120, M=1..64, Post multiplier 2, all three
epsilons 1e-6, 20 Sinkhorn iterations, and the existing automatic split policy.
Direct-x calls, other shapes/scalars and explicit split overrides retain their
existing entry. No launch option or environment variable is added.

## Operator validation

On nmz18 in `duyl-033-w4a8-1119`, the expanded probe covers all 80 target and 6
draft attention/FFN weights, M=1/4/8/16/32/40/48/64, all four Post/previous-mix
nullable combinations, six magnitudes, and changing inputs and weights during
graph replay. All 16,512 eager and 8,256 graph checks passed. Updated residual
and collapsed input are byte exact BF16; FP32 mixes meet the existing
rtol=2e-5, atol=2e-6 contract. The actual model Post multiplier is 2.

The registered owner passed 24 eager and 24 changing-input graph checks,
including exact comparison with the same public IR RMSNorm as the existing
runtime. Its five-output fake schema also passed a fullgraph `aot_eager`
compile. CPU contract/bridge regressions: 82 passed. L0: GREEN, 30 existing
warnings. Initial RMS probe used a different private custom-op oracle; the
corrected probe uses the production public IR operation without changing the
runtime or tolerance.

Complete operator medians, native versus fused, in microseconds:

| M | Native | Fused |
|---|---:|---:|
| 8 | 33.9760 | 20.2104 |
| 48 | 45.9752 | 30.0360 |
| 64 | 51.2988 | 35.7092 |

These are operator results. Full HumanEval-164 and fixed fake-prefill,
five-position 0.6 synthetic acceptance performance are pending for this
increment. They do not change the last accepted model result of 159/164 or its
73.7678 ms profiled step. If the new full-model gates fail, revert this
increment.

Reproduce the operator gates with `tests/tools/probe_mhc_boundary_expanded.py
--model <checkpoint> --output <new-result.json> --post-mult 2` and
`tests/tools/probe_mhc_boundary_registered.py --model <checkpoint>
--output <new-result.json>`
in the pinned v0.33 provider environment. Preserve the expanded probe's
real-weight paths and the test environment recorded in the experiment manifest.
