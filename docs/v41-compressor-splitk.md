# Small-row BF16 compressor projection

The native BF16 -> FP32 compressor GEMM takes about 82 us for small decode
batches on gfx938/64 CU. The candidate uses independent 128-element FP32
partials and an FP64 reduction. Dot products with strong partial cancellation
are recomputed directly in FP64; reductions of K <= 128 use FP64 directly.
It changes no stored weights or checkpoint format and adds no environment flag.

The public kernel supports contiguous BF16 A[M,K], W[N,K], 1 <= M <= 64,
1 <= N <= 4096, K a multiple of 32 up to 16384, on gfx938/64 CU only.
Callers check `supported`; unsupported inputs raise instead of silently changing
precision. The Plugin admits M <= 48 for the owned compressor consumer and
retains native GEMM for other shapes/providers. This is a variable batch route.

Numerical acceptance uses exact BF16 operands evaluated in CPU FP64 with
rtol=2e-5, atol=2e-6; it does not require equality with the native FP32 summation
order. Real layer-14 compressor and indexer weights plus generic masked shapes
passed 45 shape cases, each with three amplitudes and changing-input graphs.
The indexer weights consumer returns BF16 and is deliberately not bound to this
FP32 provider. Boundary tests additionally cover zeros, exact/near cancellation,
exponent spread, K/N tile edges and changing weights. Earlier failing partial
sum implementations are preserved in the experiment directory.

Evidence: run/dsv4.1/033/fp8_optimization_20261123/
`AUX_SPLITK_GATE.json`, `microbench/aux_splitk_refined_shortk.json`,
`aux_splitk_boundary_tests.log`, `microbench/compressor_consumer.json`.
Those operator results do not establish an end-to-end speedup. Promotion needs
the frozen full HumanEval-164 threshold (>=156 passes, zero errors/truncations),
all-rank trace proof and unprofiled paired serving measurements.
