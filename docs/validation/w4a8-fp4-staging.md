# W4A8 FP4 decode page staging

The V4.1 FP4 reader gathers variable physical pages into64-record canonical pages. The prior path materializes slot IDs, native gather payload/scales and two repacking copies. The new `fp4_staging_triton` kernel copies the same bytes directly into caller-owned pages and fills the canonical table.

The tested domain is HIP gfx93864CU, physical32/128-record pages, page-local64byte payload and4byte raw UE8M0 scale planes. Invalid positions reproduce native slot0 bytes. Device lengths and block IDs are read on each execution, including graph replay. Unsupported shapes/hardware and injected gather collaborators retain the existing path. No scoring,quantization or top-k arithmetic changes. Workspace reservations and the conservative startup scratch budget remain unchanged.

The exact runtime in `1e11880` passed72 eager cases,72 changing metadata graph replays and4 native LightOp decode consumer cases. Tests include padded/offset sourcepages, lengths0 and page boundaries,rows1/8/48/64,context65/4096/32768. Source inputs stayed byte-exact. Existing Plugin CPU regressions86passed and Adapter L0 GREEN30warn.

At context4096,shuffled3repeat medians for full staging wrappers with8transactions pergraph were99.36→10.04us at rows8 and317.39→50.89us at rows48(R32). R128 was comparable. These are operator timings; full164 accuracy and fixed fakeKV5literal0.6 service performance are pending. Published before service gate at user request; withdraw the increment if frozen gates fail.

Evidence: `run/dsv4.1/033/w4a8_reverify_20261119/FP4_STAGING_UNIT_RESULTS.json` and `FP4_STAGING_SERVICE_EXPERIMENT.json` in the experiment workspace. No new launch option or environment flag is required.
