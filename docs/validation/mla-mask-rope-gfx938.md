# Exact native mask/writeback + inverse RoPE on gfx938

The native inverse GPT-J RoPE expression is fused with BF16 empty-row masking
and writeback. Zeroed rows are still rotated to preserve native signed-zero
behavior. Admission: gfx938,1..64rows, full H64/H128,D512, contiguous BF16
result/output, contiguous int32 current main/extra counts, int32/int64
positions and contiguous FP32[P,64]cos/sin. Other cases keep the accepted I/O
kernel and caller rotation. The native source file SHA256 and live inverse
function identities are pinned; unknown/replaced code falls back. Core globals
are unchanged. Successful selection returns the rotated row count to the
existing native caller, which rotates only the remaining rows.

160mask/rotation cases cover all BF16 bit patterns, empty/live/mixed rows,
changed positions and both sink modes.1,152complete native decode consumer
cases pass byte-exact in eager and changed-input Graph replay, with all input
bytes unchanged. Coverage includes all eight capture buckets,H4/H8/H64/H128,
contiguous/strided/misaligned layouts, both SWA modes and magnitudes
0/.001/.25/1/8/100. Fused versus fallback row-count selection is asserted.
The complete bridge includes its caller's rotation when comparing outputs.
M48/H64 graph timing:130.557->124.265us(SWA),190.910->184.360us(combined).
153focused combined plugin/adapter CPU checks pass, including unknown native
source/function rejection. Combined serving admission with plugin2820632 now passes. HE32 is32/32,
identical baseline tasks/prompts, zero errors/truncations. Both formal repeats
complete64requests with0failures: median21.906416365/21.540173914ms,
aggregate21.723295139ms. Frozen matched22.667478730/22.311856336ms and
aggregate22.489667533ms ceilings and all previous gates pass. Separate
profile/workflow exit0,8traces show400target and30draft fused mask/RoPE calls
per10steps, with zero old mask/copy calls in those spans. Target phase
82.093..82.255ms, draft/sampling6.335..6.493ms, median idle0.864..0.907% /
0.779..0.848%. The plugin also admits exact draft Q/KV and final Post:
30Qsplit/30KVonly/60Post per10actualdraftforwards, context-Qnorm30 retained.

This is a combined comparison,1.476% lower aggregate TPOT than accepted I/O,
41.316% lower than original. Two repeats do not establish statistical confidence;
full SGLang stage parity remains incomplete. The earlier standalone draft
schedule failed its own TPOT gate and remains recorded separately.

GPU reproduction: tests/tools/probe_mla_decode_rope.py --baseline
<accepted03ef92e-or-02bbb6b>/hcu_adapter/kernels/mla_sparse_decode_dtk.py
--output <json>. Raw results: mla_rope_gpu_v1.json/log and
 dspark_mla_rope_cpu_v1.log under
/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/.
