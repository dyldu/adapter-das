# Delayed mHC stage2 qualification

The delayed runtime discards the native stage2's current-Pre residual collapse,
then constructs its real layer input using the previous Pre mix. The admitted
path retains the native stage1, FP32 Post/Sinkhorn arithmetic, next-Pre mix,
previous-mix collapse and decoder RMSNorm. It removes only the unused collapse
and reads the existing 32-column stage1 pitch directly instead of copying its
24 live columns to contiguous storage. Ordinary Pre still produces its layer.

Admission requires H5120/HC4, BF16 contiguous residual, FP32 contiguous mixes,
1–64 rows, split1/32, gfx938/64CU, x=None, no CUDA autocast or enabled tensor
gradients, and the qualified native TileLang wrapper/factory/source. Unknown
contracts retain the existing implementation before execution. There are no
new launch flags. Runtime execution failures are not retried on another path.

The native source is AITER `pre_big_fuse_kernel.py`, SHA256
`2ac06a653f02491b6c319d1e46cebcb9fd64081b8499fb2763c02c2ff9bf50a9`.
Its TileLang arithmetic is carried in the explicitly named `_fork` module,
with MIT attribution. The separate facade contains source/layout admission
and dispatch, with no tensor numerical implementation.

The helper probe passes 576 byte-exact eager cases and 480 changed graph
replays using actual checkpoint layers0/20/39, both attention and FFN, eight
row buckets, both splits and six magnitudes. Native stage2 versus the candidate
is9.2755→6.2165us at48 rows and9.4138→6.2505us at64 rows, medians of three
shuffled graph timing repeats. Including stage1,30.0591→26.8167us and
33.4839→28.8281us. These are operator measurements.

The actual `MhcAiterRuntime.delayed` probe additionally passes576 eager cases
and480 changing graph replays using native AITER stage1, native Core Pre mix,
native Core collapse and native Core RMSNorm. It checks all four returned
tensors, both presence/absence of the previous mix, and input immutability.
CPU admission/routing coverage passes63 cases, including an unhashable
TileLang factory identity regression. An earlier admission-cache failure
occurred before any numeric case and is retained in the experiment artifacts.

Full164 HumanEval, two fixed unprofiled performance repeats and all8 actual
traces are in progress on the immutable runtime commit `f7c8754`. This
follow-up only renames the kernel file to `_fork`, updates its two imports and
documents the facade naming rule. Its CPU tests pass63 and L0 is GREEN.
No full-service acceptance or end-to-end speedup is claimed for this candidate
until the frozen service gate and the final publication source checks finish.
