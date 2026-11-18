# Exact sparse decode selector compaction on gfx938

`mla_sparse_decode_dtk._selectors` uses one Triton kernel for supported integer
inputs, retaining its Torch implementation for other layouts/devices. The
operation keeps valid slots, sorts ascending with duplicates, pads with -1 and
returns an int32 count. CSR offsets and lengths remain device inputs; no host
read or scheduling-metadata cache is introduced. Native FlashMLA and KV bytes
are unchanged.

Admission: gfx938, 1..64 rows, width 0..2048, contiguous int32 dense/CSR input
and metadata, same device, bounded slot capacity. Other cases retain Torch.

32 CPU bridge/fallback checks pass. Actual helper passes 480 integer cases and
480 changing-metadata graph replays, covering all eight capture buckets,
widths 0/1/63/64/127/128/511/512/513/2048, empty/full/padded rows, duplicates,
negative/out-of-range slots and both dense/CSR inputs. Inputs remain unchanged.
The complete native decode consumer passes another 40 BF16 byte comparisons
with changing Q, cache, positions and main/extra selectors.

| Isolated M48 CSR selector | Torch graph, us | Fused graph, us |
| --- | --- | --- |
| Width 128 | 128.8223 | 3.9023 |
| Width 512 | 124.0944 | 5.6577 |

Reproduction: use `HCU_ADAPTER_HOST=none python -m pytest -q
tests/adapter/test_mla_selectors_contract.py tests/adapter/test_vllm_flashmla_decode.py`
for CPU checks. On gfx938 run `python tests/tools/probe_mla_selectors.py --output
<result.json>` with this adapter and the installed native FlashMLA/vLLM binaries.

HE32 passes **32/32**, identical baseline tasks/prompts, zero errors/truncations,
with real prefill and speculative acceptance. Two unprofiled 64-request repeats
complete without failures: median TPOT **22.87067 / 22.56973 ms**, mean of repeat
medians **22.72020 ms**. This is 14.834% below the accepted target Q/KV control
and 38.623% below the original frozen baseline. All original and additional
2% matched/aggregate ceilings frozen before measurement pass. Two repeats
provide an observed comparison, without a statistical confidence claim.

The separate profile and workflow exit 0 and export eight traces. Every rank
selects 780 target selector kernels per ten target steps, mean 8.206..8.320 us.
Draft selects 27 calls on rank4 and 30 on other ranks, exactly matching each
rank's native draft Q/KV invocation count, mean 6.495..6.692 us. Target phase
medians are 85.534..85.690 ms; draft/sampling medians 6.868..7.029 ms. Median
GPU idle fractions are 1.662..1.711% and 1.366..1.478%, respectively. The
provided SGLang trace differs in hardware/context metadata, and complete
SGLang step parity remains unproven.

Validated production source: `161e27b`, companion plugin `06e7912`. Raw evidence
is under `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/mla_selectors_*`.
