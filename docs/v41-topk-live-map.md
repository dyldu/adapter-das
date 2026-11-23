# Live-prefix TopK score-map launch scheduling

The existing FP32 score bit mapping, native LightOp TopK, and final output
translation are unchanged. The live-prefix mapping now uses 512-element tiles
and up to eight waves of CTAs per compute unit, instead of 4096-element tiles
and one wave. The grid depends on storage width, row count and device CU count;
device lengths still bound every loop after CUDA Graph replay. There is no
fixed batch/context specialization, new flag or extra launch.

The frozen study TOPK_MAP_GATE.json checks live mapped score bytes and offsets
against the former launch, including NaN payloads, infinities, signed zeros,
subnormals, empty/ragged rows and poisoned unused tails. It replays the actual
select_topk function with only the mapper launch replaced and retains the
unchanged native selection/output consumers. All 104 study cases passed.

For M42 and storage width1048576, complete selection measured 110.7 ->72.4 us
in the diagnostic single-device study; other devices on that node may run jobs.
This is not an end-to-end performance claim. Full HumanEval164 >=156, zero
errors/truncations, the frozen all8rank trace and unprofiled TPOT measurements
are required before promotion. Numerical GPU tests use an independent NumPy
IEEE-rank oracle and changing graph inputs for K512/1024/2048, rows1..257,
widths513..1048576. Core and Plugin call sites remain unchanged.
