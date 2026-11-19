# W4A8 MLA scheduling reuse

The native FlashMLA computation stays intact. This change removes redundant
query/sink padding for admitted full heads, combines exact BF16 output masking
with inverse RoPE, and builds fresh native plans from current device counts.
Unsupported architectures, layouts, dtypes and unrecognized native providers
retain the existing path. No native query, cache or selector input is mutated.

Plan metadata is rebuilt on every invocation and graph replay. Device token
counts are read at execution; Python state never supplies stale routing or
sequence lengths. Native source, binary and function identities are pinned.
The existing native consumer receives the same metadata contract and values.

With W4A8 block64, I/O and inverse-RoPE probes each passed 160 changed-input
mask cases and 1,152 complete native consumer cases. The plan probe passed
832 dynamic plan records and 1,152 native consumer cases. Coverage includes
empty rows, changing counts, frozen capture buckets and fallback layouts.
Admitted BF16 outputs are byte-exact and all inputs remain unchanged.

Tested Adapter45a32f3 with Plugin9bc97af, which also includes target Q/RoPE
and optional draft final Post, passed full HumanEval 159/164 (96.9512%), first
32/32, and zero errors/missing/truncations. Paired unprofiled TPOT medians
37.145193/37.170023 ms became 33.456961/33.405761 ms. The mean reduction is
10.0282% for the complete combination; no individual MLA speedup is inferred.
Both frozen median/p99 and minimum 2% improvement gates pass. These earlier
paired measurements use synthetic rates `[0.6,0.36,0.216,0.1296,0.07776]`.
The user-requested five literal `.6` rates have been revalidated separately;
absolute TPOT across those workloads is not an operator speedup. Eight-rank
traces confirm actual fresh-plan and inverse-RoPE/mask execution in target
and draft while preserving a single FULL target graph launch.

All frozen model/capacity parameters remain: INT4 W4A8, TP1/DP8/EP8,
memory0.85, block64, max-seqs32, max-len32768, max-batched-tokens256,
capture1/4/8/16/32/40/48/64 and DSpark5. Performance uses fake KV with
synthetic acceptance and is separate from real-KV full-model accuracy.
This does not establish complete MLA parity with SGLang.

On the corrected literal-rate workload, the same AG/RS source composition
has paired median TPOT20.329348/19.632577ms versus the published activation
stack22.509010/22.116227ms. Mean medians22.312618 ->19.980962ms reduce by
10.4499% for the complete combination. Both repeats pass the frozen
median/p99 and minimum2% improvement gates. All64 requests complete in each
repeat. Sixteen independent traces of these two versions confirm FULL and
actual operator dispatch. The unchanged source commits reuse their complete
real-prefill accuracy gates; this synthetic performance rerun is not a new
accuracy evaluation. Quantized provider, batch and hardware context differ
from the supplied SGLang trace.

CPU CI now includes the I/O, inverse-RoPE and fresh-plan contract files.
Publication validation: 508 passed / 28 skipped; structural gate GREEN.
Operator and complete-model GPU results provide separate device evidence.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`:

- `nmz18/attention_reuse_probes_20261119/mla_decode_io.json`
- `nmz18/attention_reuse_probes_20261119/mla_decode_rope.json`
- `nmz18/attention_reuse_probes_20261119/mla_native_plan.json`
- `nmz18/reuse_stack_v2.acceptance.json`
- `nmz18/reuse_stack_v2_accuracy_equivalence.json`
- `nmz18/mla_publication_cpu_v2.log`
- `nmz18/mla_publication_l0.log`
- `nmz18/uniform06.acceptance.json`
- `nmz18/uniform06_20261119/`
