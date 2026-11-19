# W4A8 FP4 decode page staging

The V4.1 FP4 reader gathers variable physical pages into64-record canonical pages. The prior path materializes slot IDs, native gather payload/scales and two repacking copies. The new `fp4_staging_triton` kernel copies the same bytes directly into caller-owned pages and fills the canonical table.

The tested domain is HIP gfx93864CU, physical32/128-record pages, page-local64byte payload and4byte raw UE8M0 scale planes. Invalid positions reproduce native slot0 bytes. Device lengths and block IDs are read on each execution, including graph replay. Unsupported shapes/hardware and injected gather collaborators retain the existing path. No scoring,quantization or top-k arithmetic changes. Workspace reservations and the conservative startup scratch budget remain unchanged.

The exact runtime in `1e11880` passed72 eager cases,72 changing metadata graph replays and4 native LightOp decode consumer cases. Tests include padded/offset sourcepages, lengths0 and page boundaries,rows1/8/48/64,context65/4096/32768. Source inputs stayed byte-exact. Existing Plugin CPU regressions86passed and Adapter L0 GREEN30warn.

At context4096,shuffled3repeat medians for full staging wrappers with8transactions pergraph were99.36→10.04us at rows8 and317.39→50.89us at rows48(R32). R128 was comparable. These measure the staging wrapper.

Full service validation completed on nmz18 on 2026-11-20. Pinned Plugin `1c935df` plus Adapter `1e11880` match the published runtime/test files byte-for-byte outside documentation/CI. Real-prefill HumanEval passed159/164(96.95%),first32=32,zero errors/missing/truncations. Unprofiled median TPOT was19.065356/18.688054ms,p9922.367298/23.229673ms,64completed/0failed per repeat. Against the frozen accepted WO_A composition(19.932703/19.343608ms),the repeat gains were4.35%/3.39% and the mean of repeat medians improved3.88%. Unchanged regression limits(median2%,p995%,mean2%) and>=2%-per-repeat gain threshold passed. FakezeroKV and five literal unconditional0.6 rates apply only to performance;accuracy retains real prefill/rejection.

All8 actual traces show3 fused staging calls,0 native FP4 gathers and8 native FP4 scorer calls per target span. FULL target/draft execution,DeepEP40/3 layers,activation40/3 calls,mHC Post>=83/6 and40 WO_A BMMs without following layout copies remain. Delayed mHC stays absent. DP0 target-through-draft step was77.608710ms versus79.812687ms in the prior accepted trace. The supplied SGLang trace is65.098274ms with different hardware,batch,context and providers;whole-step parity is unfinished.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`: `FP4_STAGING_UNIT_RESULTS.json`, `FP4_STAGING_SERVICE_EXPERIMENT.json`, `nmz18/fp4_staging_service_20261120_v2/acceptance.json` and `nmz18/fp4_staging_step_reference_20261120.json`. No new launch option or environment flag is required.
