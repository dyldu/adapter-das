# Exact sparse decode selector compaction on gfx938

`mla_sparse_decode_dtk._selectors` uses one Triton kernel for supported integer
inputs, retaining its Torch implementation for other layouts/devices. The
operation keeps valid slots, sorts ascending with duplicates, pads with -1 and
returns an int32 count. CSR offsets and lengths remain device inputs; no host
read or scheduling-metadata cache is introduced. Native FlashMLA and KV bytes
are unchanged.

Admission: gfx938, 1..64 rows, width 0..2048, contiguous int32 dense/CSR input
and metadata, same device, bounded slot capacity. Other cases retain Torch.

This W4A8 experiment reapplies the integer kernel from `161e27b` onto upstream
main `ef63bbb`. Validation from the earlier FP8 series is historical evidence
only; no W4A8 operator or serving gain is admitted by this document yet.

The native probe checks 480 integer cases and
480 changing-metadata graph replays, covering all eight capture buckets,
widths 0/1/63/64/127/128/511/512/513/2048, empty/full/padded rows, duplicates,
negative/out-of-range slots and both dense/CSR inputs. Inputs remain unchanged.
The complete native decode consumer checks another 40 BF16 byte comparisons
with changing Q, cache, positions and main/extra selectors.

Reproduction: use `HCU_ADAPTER_HOST=none python -m pytest -q
tests/adapter/test_mla_selectors_contract.py tests/adapter/test_vllm_flashmla_decode.py`
for CPU checks. On gfx938 run `python tests/tools/probe_mla_selectors.py --output
<result.json>` with this adapter and the installed native FlashMLA/vLLM binaries.

Current W4A8 CPU checks, native precision/timing, identical-prompt HE32,
same-node unprofiled serving/profile checks and final full164 are pending.
The MoE patch remains the prior accepted intermediate; retain its latency SLA
and accuracy gates. Raw W4A8 evidence belongs under
`/public2/home/duyunlong/run/dsv4.1/033/w4a8_reverify_20261119/`.
