# ChronoGRN — model overview

ChronoGRN is a structured model for gene regulatory networks (GRNs) from
time-ordered paired RNA + chromatin-accessibility (multiome) measurements. It
answers:

> Which TF regulates which target, through which accessible enhancer, at what
> stage of a trajectory, and after what delay?

Unlike a static GRN, ChronoGRN represents regulation as a continuous-time
system with (1) separate accessibility→TF and TF→target delays, (2) enhancer
accessibility as a multiplicative regulatory gate, (3) signed TF–target edge
strengths that change smoothly over time, and (4) counterfactual rollouts for
held-out TF interventions.

## 1. State and information flow

ChronoGRN tracks four coupled blocks of state:

```
a(t) — TF-promoter accessibility
r(t) — TF RNA / activity proxy
e(t) — target-enhancer accessibility
x(t) — target-gene RNA
```

Intended regulatory chain:

```
TF-promoter ATAC --τ_access→TF--> TF RNA/activity
                                        |
                                        | delayed W_target,TF(t)
                                        v
enhancer ATAC ---- multiplicative gate ----> target-gene RNA
```

The enhancer signal does not independently create target RNA — it gates
transmission of delayed TF activity through prior-supported TF→enhancer→gene
paths.

## 2. Implementation

| Component | File |
|---|---|
| Vector field / model class | [`src/causal_grn/enhancer_model.py`](../src/causal_grn/enhancer_model.py) |
| Structured (ridge/profile) initializer | [`src/causal_grn/enhancer_initialize.py`](../src/causal_grn/enhancer_initialize.py) |
| Recommended fit entry point | [`src/causal_grn/optimized_fit.py`](../src/causal_grn/optimized_fit.py) |
| Optional gradient fine-tuning | [`src/causal_grn/enhancer_train.py`](../src/causal_grn/enhancer_train.py) |
| Checkpoint save/load | [`src/causal_grn/checkpoint.py`](../src/causal_grn/checkpoint.py) |
| Synthetic ground truth + data | [`bench/delayed_multiome.py`](../bench/delayed_multiome.py) |
| Scoring against ground truth | [`scripts/experiments/scoring.py`](../scripts/experiments/scoring.py) (`score_model`) |

- **Continuous-time vector field.** `da/dt, dr/dt, de/dt, dx/dt` are defined
  and integrated with Euler or Heun (2nd-order predictor–corrector) stepping.
- **Two delay clocks.** One continuous bounded delay per TF
  (accessibility→TF) and one continuous bounded delay per candidate edge
  (TF→target), evaluated by differentiable interpolation between neighboring
  history states. Legacy softmax lag-mixture checkpoints remain loadable.
- **Time-varying edges.** `W(target, TF, t)` is a low-frequency Fourier basis
  over time — the validated short-trajectory default uses one frequency to
  limit overfitting.
- **Enhancer gate.** Enhancer accessibility multiplicatively scales
  TF→target transmission, restricted to motif- and peak-to-gene-prior
  supported paths.
- **Interventions.** A TF knockdown zeroes the TF's promoter accessibility
  and RNA state from a given start time; the system is then re-integrated to
  predict downstream target responses.
- **Reference-conditioned counterfactuals.** Held-out intervention
  evaluation can condition baseline drift on the observed control
  trajectory, so the reported perturbation prediction measures
  intervention-driven deviation from control rather than requiring the model
  to also reproduce long-horizon free-running drift from scratch.

## 3. Fitting procedure (the shipped default)

1. Smooth each observed trajectory with a Savitzky–Golay filter and estimate
   derivatives.
2. Fit promoter-accessibility, TF-RNA, and enhancer subsystems by ridge
   regression.
3. Profile continuous accessibility→TF delays with a deliberately small
   nuisance basis so background forcing can't absorb the delayed signal.
4. Profile edge-specific TF→target delays by coordinate regression.
5. Fit static and dynamic edge coefficients jointly in the selected temporal
   basis.
6. (Optional, disabled by default) robust gradient fine-tuning: Huber
   derivative loss, multistep rollout loss, response-delta loss.

Fine-tuning is disabled by default (`--finetune-epochs 0`) because on the
bundled short synthetic trajectories the pure structured estimator gave the
strongest overall balance of topology, delay, dynamic-edge, and perturbation
recovery. No held-out TF4/TF5 condition enters fitting, hyperparameter
estimation, or checkpoint selection.

## 4. Verified capabilities (this shipped run: 16 seeds, 0–15)

| Metric | Mean ± sd |
|---|---:|
| Directed-edge AUPRC within the candidate prior (baseline 0.562) | 0.937 ± 0.055 |
| Dynamic edge-component correlation | 0.958 ± 0.018 |
| Accessibility → TF delay correlation | 0.971 ± 0.016 |
| TF → target delay correlation | 0.710 ± 0.108 |
| Control target free-roll R² | 0.935 ± 0.017 |
| Held-out endpoint-response correlation | 0.940 ± 0.022 |
| Held-out full-response trajectory correlation | 0.702 ± 0.086 |
| Held-out target rollout R² | 0.855 ± 0.039 |
| Held-out sign accuracy (effectful targets) | 0.968 ± 0.043 |
| Stage-early / stage-mid / stage-late response r (weighted) | 0.916 / 0.901 / 0.718 |

TF→target delay estimation and late-stage full-trajectory response remain
the weakest components and should not be described as solved. Full per-seed
numbers: [`results/benchmark/per_seed_metrics.csv`](../results/benchmark/per_seed_metrics.csv).

## 5. Paired ablation interpretation (same 16 seeds)

| Variant | Held-out endpoint r | Held-out rollout R² | Edge AUPRC |
|---|---:|---:|---:|
| ChronoGRN (full) | 0.940 | 0.855 | 0.937 |
| No delay | 0.879 | 0.780 | 0.901 |
| No enhancer gate | 0.912 | 0.839 | 0.954 |
| Fixed edges (no dynamics) | 0.402 | 0.247 | 0.701 |

The full model beats the fixed-edge model on every seed on the major
predictive metrics — removing time-varying edges is catastrophic for
held-out perturbation prediction (0.940 → 0.402), because the benchmark is
explicitly designed so a fixed-W model cannot distinguish an early-stage
knockdown from a late-stage one. Delays and enhancer gating give smaller but
consistent gains, mostly on held-out prediction rather than control
reconstruction (see [`results/ablation/results.json`](../results/ablation/results.json)
for the full per-metric table).

## 6. Inputs and outputs

**Inputs:** time-ordered paired RNA and ATAC profiles (or pseudotime-binned
profiles); TF-promoter accessibility; TF RNA/activity proxy; enhancer
accessibility; motif and enhancer-to-gene candidate-path priors; optional
labeled perturbation trajectories for training.

**Outputs:** TF-specific accessibility→TF delays; edge-specific TF→target
delays; signed dynamic edge tensor `W(target, TF, t)`; enhancer-gate
trajectories; free or reference-conditioned state rollouts; predicted
responses to held-out or hypothetical TF interventions.

## 7. Scope and claim boundary

- The numbers above are controlled synthetic positive-control
  identifiability results, not validation in human or mouse tissue.
- Pseudotemporal ordering alone does not prove biological causality.
- Motif and peak-to-gene priors constrain candidates but do not prove
  physical TF binding or uniquely identify a causal enhancer — and in this
  synthetic benchmark, the "prior" is the generator's own known structure,
  not independently measured biology.
- Edge AUPRC is measured within the supplied candidate prior and must always
  be reported with its prior-only baseline (0.562 here).
- Reference-conditioned perturbation prediction does not require the model
  to generate baseline developmental drift from scratch.
- Real-data validation, external TF-binding support, and cross-dataset
  replication are a separate, harder claim than this benchmark makes — see
  the companion real GSE205117 mouse-gastrulation fit (sealed Brachyury
  CRISPR knockout test) for that evidence, which is weaker and reported
  separately with its own honest caveats.

## One-sentence summary

ChronoGRN is a delay-aware, enhancer-gated continuous-time GRN model that
recovers dynamic regulatory structure and predicts held-out TF perturbation
responses in a controlled multiome positive-control system.
