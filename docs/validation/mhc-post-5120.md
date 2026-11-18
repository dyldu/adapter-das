# gfx938 H5120 Post candidate

The existing H5120 AITER route deliberately uses the official Torch Post.
Its FP32 `einsum` and separate pointwise/cast operations are visible in the
V4.1 decode profile. Directly enabling the native AITER Post fails the frozen
precision gate: magnitude1 max_abs0.0625, magnitude8 max_abs0.5. The native
width guard remains in place.

The new Triton path uses the observed gfx938 Torch FP32 BMM accumulation order
and retains the separate FP32 post product and final BF16 storage boundary.
Other reduction trees changed BF16 halfway decisions and were rejected.
Admission requires contiguous device tensors, H5120, four residual heads,
1..1024 tokens, BF16 x/residual, FP32 mixes, highest FP32 matmul precision and
gfx938. Other inputs preserve the existing Torch/native fallback policy.
The adapter compile fingerprint now includes both Python runtime and new
kernel source content so an existing compiled graph cannot mask a code change.

Source89cc71d, nmz18 / duyl-033-1112, Torch2.13 DTK2610:

- 74 focused CPU contract, stage2 routing and public bridge tests pass.
- Actual production helper and `MhcAiterRuntime.post` pass 123 eager cases
  with zero differing BF16 bits, unchanged input tensors and finite outputs.
  Shapes1/4/8/16/32/40/48/64/128/1024, magnitudes0/.001/.25/1/8/100 and both
  supported post layouts are covered, plus signed zero, cancellation and
  BF16 halfway cases.
- All eight decode graph buckets pass five replays each with changing x,
  residual, post and comb tensors: 40 graph comparisons, byte-exact with Torch.
- Isolated unrolled graph timings in microseconds:

| Tokens | Torch | Adapter |
| --- | ---: | ---: |
| 1 | 15.885 | 2.227 |
| 8 | 41.892 | 2.934 |
| 40 | 162.937 | 6.496 |
| 48 | 194.271 | 7.342 |
| 64 | 258.730 | 9.392 |
| 128 | 505.408 | 18.008 |
| 1024 | 3931.653 | 146.002 |

These timings do not establish serving performance. HE32 and paired unprofiled
serving/profile validation are pending at this snapshot. Retain the original
SLA and additional matched ceilings32.512422146964965 /32.19653568146237ms,
aggregate32.35447891421367ms, frozen before serving measurements against the
accepted model-entry-clean control.

Reproduce on an idle GPU with this worktree on `PYTHONPATH`:

```bash
HCU_ADAPTER_HOST=none python -m pytest -q \
  tests/adapter/test_mhc_post_5120_contract.py \
  tests/adapter/test_mhc_aiter_contract.py \
  tests/adapter/test_mhc_stage2_routing.py \
  tests/adapter/test_vllm_mhc_bridge.py
HCU_ADAPTER_HOST=none python tests/tools/probe_mhc_post_5120.py --output post.json
```

Full artifacts: `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/`
`mhc_post_{cpu_v1.log,production_v1.json,production_v1.log,preflight_sla.json}`.
