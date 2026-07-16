# Mathematical Red Team

## Minimal moving-atom counterexample

Top-K selection is not needed to expose the core problem. Let the vocabulary
contain two tokens and select both. Let independent standard Gumbels be clipped
as

`C_i = min(max(G_i, a), b)`, with `a < b`.

For policy log-probability gap `delta`, noise scale `s`, and mixture temperature
`tau`, the executed first-token mixture weight is

`Q_delta = sigmoid((delta + s(C_1 - C_2)) / tau)`.

The clipped variables have masses `p_a = F_G(a)` and
`p_b = 1 - F_G(b)`. Therefore `Q_delta` has positive point masses at

- `sigmoid((delta + s(a-b))/tau)`, mass `p_a p_b`;
- `sigmoid(delta/tau)`, mass `p_a^2 + p_b^2`;
- `sigmoid((delta + s(b-a))/tau)`, mass `p_b p_a`.

For a generic policy update `delta -> delta'`, all three locations move and do
not coincide with the old three locations. The new policy then assigns positive
mass to singleton sets that the old policy assigns zero mass, and conversely.
Thus neither executed-action law generically dominates the other. The ordinary
finite-step Radon-Nikodym likelihood ratio used to rewrite a new-policy
expectation under old-policy samples does not exist.

`counterexample.py` computes these masses exactly. At the official bounds, the
two-noise event that creates point masses has positive probability
`(F_G(-1.5) + 1 - F_G(3))^2`; post-selection bias can make boundary events much
more frequent in the actual sampler.

## Why this is not CAPG

CAPG clips an executed action to fixed environment endpoints. Across policy
updates, its Dirac locations remain the same, so a common reference measure
consisting of Lebesgue measure plus endpoint Diracs exists. Here the noise is
clipped before adding policy-dependent scores and normalizing into mixture
weights. The resulting action atom locations move with the policy.

## Adversarial checks

### Does this prove that policy gradients do not exist?

No. It disproves the ordinary finite-step importance ratio on the executed
action law. On-policy weak derivatives, boundary corrections, differentiable
simulation, or zeroth-order finite differences may still estimate a gradient.

### Can stored base noise restore PPO?

No. Unclipped base Gumbel noise has a parameter-independent law, so its policy
score is zero. Replaying the same noise under new logits changes the executed
mixture and downstream state; it is not evaluation of the old executed action
under a new policy.

### Can clipping labels restore a common measure?

Labels make lower/interior/upper events explicit but do not fix the executed
mixture location. Conditional on a boundary label, the mixture still moves with
the policy. Adding the old mixture to the action restores action identity but
reintroduces the moving singular support.

### Does the result hold for K=1?

Not through mixture weights. With one selected token the executed mixture is
one-hot, so clipping does not move its weight. Dynamic candidate support can
still create zero-probability discrete actions, but the moving-weight theorem
requires at least two active mixture components.

### Can HPO provide the missing gradient?

Only if downstream transition and cost are differentiable and replayable for a
fixed exogenous scenario. Outcome-checked LLM reasoning does not expose that
simulator contract. Treating the terminal correctness reward as differentiable
would change the problem.

### Can a black-box correction be estimated?

In principle, yes: counterfactual continuation rollouts or parameter/action
finite differences can estimate directional effects. In this autoregressive
setting that adds branching at latent steps, has no near-GRPO unbiased estimator
identified by the audit, and collides with generic zeroth-order and
measure-valued-gradient prior art. This fails the practical-method gate rather
than proving mathematical impossibility.

## Red-team conclusion

The no-ratio statement is correct only with the stated scope. The strongest
honest result is an action-contract theorem and diagnostic. It is not, by
itself, a trainable policy-optimization method.
