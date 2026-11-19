> W4A8 reuse candidate: CPU contracts pass; W4A8 operator/consumer and serving gates are pending. Historical figures below belong to the earlier FP8 experiment.

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
source/function rejection. Serving HE32 and formal/profile admission pending.

GPU reproduction: tests/tools/probe_mla_decode_rope.py --baseline
<accepted03ef92e-or-02bbb6b>/hcu_adapter/kernels/mla_sparse_decode_dtk.py
--output <json>. Raw results: mla_rope_gpu_v1.json/log and
 dspark_mla_rope_cpu_v1.log under
/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/.
