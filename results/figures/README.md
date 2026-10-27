# ChronoGRN synthetic-benchmark figures (honest claims)

n = 16 independent synthetic networks. Stage-specific held-out TF4/TF5 KDs
(early/mid/late). Continuous delays. Pathway-prior-restricted edge AUPRC.

## Safe claims supported by current results

- ChronoGRN reconstructs delayed multiome trajectories.
- Accessibility→TF delays are recovered well (r ≈ 0.97).
- Directed topology recovery within the motif/enhancer prior is strong
  (AUPRC ≈ 0.94; prior baseline ≈ 0.56).
- Dynamic edge-shape recovery is strong on the controlled benchmark
  (ΔW r ≈ 0.96).
- Held-out intervention conditions are predicted accurately
  (endpoint r ≈ 0.94).
- The full model improves endpoint, trajectory, and rollout metrics over the
  no-delay, no-gate, and fixed-edge ablations in the paired benchmark.
- Edge-specific delay recovery remains the weakest mechanistic component
  (r ≈ 0.71).

## Claim boundaries

- The enhancer path is conditioned on supplied motif and peak-to-gene priors;
  it is not a de novo physical-binding discovery result.
- Edge-specific delays should not be described as precisely recovered.
- These results do not establish causal biology in real tissue.

## Figure messages

1. Delayed multiome model + delay recovery + paired ΔR² ablations
2. Held-out TF knockdown *conditions* (stage-specific)
3. Dynamic enhancer-resolved GRN (reweighting, not hairball rewiring)

This is a methodological positive-control study, not a biological discovery claim.
