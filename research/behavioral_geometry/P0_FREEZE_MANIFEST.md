# P0 Freeze Manifest: Horizon Sufficiency

**Frozen:** 2026-07-16  
**Repository base:** `f0a20e5a91de98978a8ca56d5aa608f99e43a469`  
**Scientific decision:** `HOLD`  
**GPU decision:** `NO-GO`  
**Next authorized action:** CPU/source-only feasibility audit for a stochastic
reference-history P1 phenomenon gate

This manifest records the exact P0 contracts accepted after three independent
red-teams: claim discipline, mathematical validity, and collision/novelty.
Their `PASS` verdicts authorize freezing P0 only. They do not establish the
phenomenon, validate a certificate, or authorize a checkpoint run, method
implementation, or training.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `P0_CLAIM_CONTRACT.md` | `78c95be7d81be5322058e7b64a763986d50205b5ae3c4f7fcd8892a86539e60c` |
| `P0_MATHEMATICAL_SPEC.md` | `9665d75e5082b9e428d6814200bec1045e124fdfb287bd613abdb4a9acf15452` |
| `P0_COLLISION_MATRIX.md` | `f87321987982b91e5321307d869c5ad975f628ab7567c3e42d614fe39ba5558e` |
| `P0_KILL_CRITERIA.md` | `5a1a093fef3482d30355bbec3522dc9cc8f71f2a527bfd18be7e8bb49e469b7c` |
| `P0_SWITCH_C2_COMPATIBILITY.md` | `b0031f6578886b20434b0d4b12b8d3d241cdb1a87f274b7ac3a32f68d8ba639a` |

Any substantive change to a frozen artifact invalidates the corresponding
hash and requires an explicit amendment plus renewed review of the affected
contract. Empirical results may update project decisions, but must not silently
rewrite these preregistered definitions or thresholds.

## Binding boundary

1. The original coordinate-first FCTR/CETR headline remains permanently
   `KILL`.
2. Frozen SWITCH C2 remains unchanged and `Not run`; it is only a factual
   greedy-prefix screen for this revised question.
3. The revised project can advance only through the gates in
   `P0_KILL_CRITERIA.md`.
4. The next decision is only whether the existing source can support a clean,
   separately preregistered P1 phenomenon experiment within a defensible
   budget.
