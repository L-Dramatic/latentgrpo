# P1 Design-Review Snapshot Manifest

**Snapshot date:** 2026-07-16  
**Design verdict:** `PASS-DESIGN`  
**Execution verdict:** `BLOCK / NO-GO`  
**Freeze meaning:** reviewed document snapshot only; not a checkpoint,
calibration, held-out, or execution freeze

## 1. Independent review results

| Review | Design verdict | Remaining boundary |
|---|---|---|
| source/estimand | `PASS-DESIGN` | source-equivalent adapter, identity replay, stop-set/tokenizer verification, and JVP tests remain unimplemented |
| mathematics/statistics | `PASS-DESIGN` | fake-model implementation, measured power inputs, final code/config hashes, and post-calibration freeze review remain missing |
| collision/AAAI positioning | `PASS-DESIGN` | residual claim is narrow; P2, mechanism-distinct replication, and real optimizer/checkpoint-delta validation are required for a strong method paper |

All three reviews distinguish design validity from executable evidence. No
review claims that the phenomenon is empirically true.

## 2. Reviewed P1 file hashes

SHA-256:

```text
P1_STOCHASTIC_NATURAL_UPDATE_PREREGISTRATION_DRAFT.md b82396f81b0a74007d8544b2f272c2c504d0687941733f442fb2a814214deecb
P1_ADAPTED_METRIC_BASELINE_CONTRACT.md                  6df30846ca05a7a55842d54a49e4268e2ca0ca906d72fcc4668b07e2e947fa2a
P1_SOURCE_FEASIBILITY_AUDIT.md                          53846839692ce9359a889008efd184704706f2277bd8a2d14d4ab33e100fde2f
P1_REDTEAM_RESOLUTION.md                                91df2d65d70d9c784d66f244123ff2a47075204aef32a245df7050800c10e482
P1_COLLISION_ADDENDUM.md                                f50f709d24de3d002d217592bdfc8d047cdf7276189cfbfc600a15672ef80754
```

Any content change to a listed file invalidates this design-review snapshot and
requires a new hash plus review proportional to the changed claim, estimator,
source semantics, or statistical rule.

## 3. Bound source state

- project branch: `aaai-pivot-training-validity`;
- project base commit: `f0a20e5a91de98978a8ca56d5aa608f99e43a469`;
- official Latent-GRPO source commit:
  `c0994fb781a2d180662bb522d8ff3e8638dcf56d`;
- checkpoint revision:
  `0db191bfd8240199db894de4de800788c845cb18`;
- current P1 primary platform: official Latent-GRPO Llama 1B, conditional on all
  preflight and compute gates.

These identifiers bind the design target only. The adapter, environment,
tokenizer/config dump, and executable artifacts do not yet exist or are not yet
frozen.

## 4. P0 integrity recheck

The five immutable P0 content hashes still match `P0_FREEZE_MANIFEST.md`:

```text
P0_CLAIM_CONTRACT.md          78c95be7d81be5322058e7b64a763986d50205b5ae3c4f7fcd8892a86539e60c
P0_MATHEMATICAL_SPEC.md       9665d75e5082b9e428d6814200bec1045e124fdfb287bd613abdb4a9acf15452
P0_COLLISION_MATRIX.md        f87321987982b91e5321307d869c5ad975f628ab7567c3e42d614fe39ba5558e
P0_KILL_CRITERIA.md           5a1a093fef3482d30355bbec3522dc9cc8f71f2a527bfd18be7e8bb49e469b7c
P0_SWITCH_C2_COMPATIBILITY.md b0031f6578886b20434b0d4b12b8d3d241cdb1a87f274b7ac3a32f68d8ba639a
```

No P0 file was modified during P1 redrafting.

## 5. Current authorization boundary

Authorized by this snapshot: preserve the reviewed design, prepare an explicit
next-stage authorization decision, and continue source-only/fake-model planning.

Not authorized: real-checkpoint measurement, GPU microbenchmark, calibration,
held-out access, certificate/method implementation, GPU rental, or training.
The staged gates in the P1 preregistration remain binding.
