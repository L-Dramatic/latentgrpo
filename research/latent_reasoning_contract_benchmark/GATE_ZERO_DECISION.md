# Gate 0 Decision

Date: 2026-07-18

Decision: `PASS_GATE_ZERO / NO GPU / NO TRAINING / NO EFFECT CLAIM`

## Scope

Gate 0 tests whether the new benchmark can execute and independently
reconstruct each pinned release's core latent recurrence or sampler on a small
controlled state. It does not use task accuracy, reward, or intervention
effects.

## Results

| Method | Unmodified source path executed | Independent oracle | Deterministic replay | Maximum error |
|---|---|---|---|---:|
| Coconut | Official `Coconut.forward` | Full recurrent sequence oracle | Pass | `5.9604645e-8` output, `0` reconstruction |
| CODI | Official `CODI.forward` | Cached projected recurrence oracle | Pass | `0` |
| Latent-GRPO | Official `Sampler.forward` | Independent source-law implementation | Pass | `0` |
| SofT-GRPO | Official `Sampler.forward` | Independent source-order implementation | Pass | `0` |

Every frozen control passed: four methods present, source equivalence,
fixed-seed replay, reconstruction tolerance, and normalization tolerance.

## Immutable records

- Config file SHA-256:
  `755cb564f217b37f7e281c74ee7e18ee944821b68a688dc6c69abd392e0a8b2c`
- Canonical config SHA-256:
  `25578cc56059eaad19071ed0ebc3c8b8d1b01e780b103eb2a460fe6a888ae15a`
- Deterministic fixture implementation SHA-256:
  `8f3bb63ddd7b0e7351e1aa5ee7aa0ad7a63b7cacbc016a61b3e6a4e66384be7d`
- Unified runner implementation SHA-256:
  `137e2e4c28250adc4d6188701c454c9435ef48b4e54e86c2b35902cc8e5513a1`
- Result SHA-256:
  `69b838e883c6d724fe07ef328a66ef7b0ad27fdf439ab6c9f17cbb8cb7aca04a`

## Engineering corrections before the result

1. The fixed environment lacks PEFT. CODI imports PEFT at module load although
   the fixture bypasses construction and never enters a LoRA path. Gate 0 now
   temporarily stubs only the three unused import symbols, then executes the
   unmodified `CODI.forward` body.
2. The first CODI fixture supplied answer positions with shape `[B, 1]`; the
   source expects `[B]` before adding gather dimensions. The fixture input was
   corrected to the source interface.

Neither attempt produced a Gate 0 result, no threshold changed, and no task
effect was available to inspect.

## Decision boundary

This pass validates the audit harness. It does not show a contract violation,
causal dependence, benchmark novelty, or AAAI-level contribution. The next
authorized work is to freeze tiny real-checkpoint state manifests and run only
no-effect/source-reconstruction controls before intervention effects.

GPU checkpoint effects remain unauthorized until those manifests are frozen.
Training remains unauthorized.
