# W4A8 integer indexer top-k postprocessing

The V4.1 indexer selects with the existing `torch.topk`. Its subsequent index translation, invalid-score masking, short-row index sort and output combination previously used separate torch launches. The admitted HIP gfx938/64CU, FP32/K512 path combines that integer postprocessing into one Triton kernel. Top-k selection, tie handling and value order remain the native torch implementation. Other shapes and devices retain the existing torch branch; no launch flag is required.

On 2026-11-20, the pinned candidate Adapter `09ca5d1` passed80 eager exact-index cases and80 changing full-graph cases, plus5 fallback cases and86 existing Plugin CPU regressions. Coverage includes rows1/8/48/64,width512/1024/4096/32768,offset/padded input rows,short/long/empty lengths,candidate-like holes,ties,NaN/inf and changing device metadata.

The first test compared independent native selections for a NaN-heavy row and failed. An unchanged-baseline versus unchanged-baseline diagnosis reproduces6 differing indices on identical inputs: native multiblock top-k tie membership is nondeterministic. The corrected test compares postprocessing of the same actual native selection. Captured full graphs retain references to their native top-k outputs for the exact postprocessing oracle; no extra GPU selection or data copy is added. Runtime code was unchanged by this test correction. Original failed evidence is retained.

Full-wrapper timing uses8 transactions per graph,100replays and3shuffled repeats. Rows48,width4096 measured263.85→155.28us;rows8,width4096 measured262.51→155.05us. These are operator timings. Full real-prefill HumanEval164 and fixed fakezeroKV/5literal0.6 service performance are pending at this early push, as requested by the user. Revert this increment if the unchanged regression gates fail.

Workspace evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`: `TOPK_POST_UNIT_RESULTS.json`, `TOPK_POST_V2_EXPERIMENT.json`, `TOPK_POST_SERVICE_EXPERIMENT.json` and `nmz18/topk_nonfinite_diagnosis_20261120.json`.
