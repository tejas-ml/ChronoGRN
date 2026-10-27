"""Delayed multiome positive-control generator.

The generator is intentionally designed so that:
  - delays are continuously distributed (not two discrete levels);
  - edge dynamics have diverse temporal classes (early / late / transient / ramp);
  - enhancer accessibility swings strongly closed→open for some paths;
  - the same TF knocked down early vs late produces different responses
    (so fixed-W models fail while time-varying W succeeds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from causal_grn.enhancer_model import EnhancerDelayGRN


@dataclass
class DelayedMultiomeTruth:
    model: EnhancerDelayGRN
    tf_enhancer_prior: np.ndarray
    enhancer_target_prior: np.ndarray
    active_edges: np.ndarray
    access_to_tf_delays: np.ndarray
    tf_to_target_delays: np.ndarray
    times: np.ndarray
    initial_state: np.ndarray
    edge_class: np.ndarray  # string labels per edge (object array)
    gated_paths: List[tuple[int, int, int]]  # (tf, enhancer, target) strong gates


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.exp(min(value, 20.0)) - 1.0 + 1e-8))


def _set_delay_logits(
    logits: torch.Tensor,
    delays: np.ndarray,
    delay_grid: np.ndarray,
    peak: float = 8.0,
) -> None:
    """Place a peaked mixture at continuous delays via linear bin interpolation."""

    logits.fill_(-peak)
    grid = delay_grid.astype(np.float64)
    flat = delays.reshape(-1)
    flat_logits = logits.view(-1, logits.shape[-1])
    for i, tau in enumerate(flat):
        tau = float(np.clip(tau, grid[0], grid[-1]))
        # locate interval
        right = int(np.searchsorted(grid, tau, side="left"))
        right = min(max(right, 1), len(grid) - 1)
        left = right - 1
        span = max(grid[right] - grid[left], 1e-8)
        w = (tau - grid[left]) / span
        flat_logits[i, left] = peak * (1.0 - w)
        flat_logits[i, right] = peak * w


def build_truth(
    seed: int = 0,
    n_tfs: int = 6,
    n_enhancers: int = 18,
    n_targets: int = 24,
    n_delay_bins: int = 12,
    max_tau: float = 0.30,
) -> DelayedMultiomeTruth:
    rng = np.random.default_rng(seed)

    # Motif-supported TF→enhancer and enhancer→target structure.
    tf_enhancer = np.zeros((n_enhancers, n_tfs), dtype=np.float32)
    for enhancer in range(n_enhancers):
        primary = enhancer % n_tfs
        tf_enhancer[enhancer, primary] = 1.0
        if enhancer % 3 == 0:
            tf_enhancer[enhancer, (primary + 2) % n_tfs] = 0.6

    enhancer_target = np.zeros((n_targets, n_enhancers), dtype=np.float32)
    active = np.zeros((n_targets, n_tfs), dtype=bool)
    gated_paths: List[tuple[int, int, int]] = []
    for target in range(n_targets):
        first = target % n_enhancers
        second = (target * 5 + 3) % n_enhancers
        enhancer_target[target, first] = 1.0
        enhancer_target[target, second] = 0.7
        tf_a = first % n_tfs
        tf_b = second % n_tfs
        active[target, tf_a] = True
        gated_paths.append((tf_a, first, target))
        if target % 2 == 0:
            active[target, tf_b] = True
            gated_paths.append((tf_b, second, target))

    model = EnhancerDelayGRN(
        n_tfs=n_tfs,
        n_enhancers=n_enhancers,
        n_targets=n_targets,
        tf_enhancer_prior=torch.tensor(tf_enhancer),
        enhancer_target_prior=torch.tensor(enhancer_target),
        max_tau=max_tau,
        n_delay_bins=n_delay_bins,
        n_time_freqs=4,
    )
    delay_grid = model.delay_grid.detach().numpy()

    # Continuous delays — not two discrete stripes. Active-edge delays are
    # spaced across the lag range so identification is not confounded by
    # nearly-identical ground-truth values.
    access_delays = rng.uniform(0.04, 0.27, size=n_tfs).astype(np.float32)
    target_delays = np.full((n_targets, n_tfs), 0.05, dtype=np.float32)
    active_pairs = list(zip(*np.where(active)))
    # Jittered linspace across lag range for active edges.
    base_grid = np.linspace(0.03, 0.27, max(2, len(active_pairs)))
    rng.shuffle(base_grid)
    for k, (target, tf) in enumerate(active_pairs):
        jitter = float(rng.uniform(-0.012, 0.012))
        target_delays[target, tf] = float(
            np.clip(base_grid[k % len(base_grid)] + jitter, 0.02, 0.28)
        )

    with torch.no_grad():
        _set_delay_logits(model.access_delay_logits, access_delays, delay_grid)
        _set_delay_logits(model.target_delay_logits, target_delays, delay_grid)

        signs = rng.choice([-1.0, 1.0], size=(n_targets, n_tfs)).astype(np.float32)
        strength = rng.uniform(1.0, 1.8, size=(n_targets, n_tfs)).astype(np.float32)
        base = signs * strength * active.astype(np.float32)

        # Diverse dynamic edge classes that Fourier basis can approximate.
        # Classes: early, late, transient, ramp_up, ramp_down, static_active
        classes = np.full((n_targets, n_tfs), "inactive", dtype=object)
        sin_coef = np.zeros((4, n_targets, n_tfs), dtype=np.float32)
        cos_coef = np.zeros_like(sin_coef)
        edge_base = base.copy()

        active_pairs = list(zip(*np.where(active)))
        for k, (target, tf) in enumerate(active_pairs):
            cls = ["early", "late", "transient", "ramp_up", "ramp_down", "static"][
                k % 6
            ]
            classes[target, tf] = cls
            s = float(base[target, tf])
            if cls == "early":
                # high early, falls later
                edge_base[target, tf] = 0.55 * s
                cos_coef[0, target, tf] = 0.70 * s
                sin_coef[0, target, tf] = 0.15 * s
            elif cls == "late":
                edge_base[target, tf] = 0.55 * s
                cos_coef[0, target, tf] = -0.70 * s
                sin_coef[0, target, tf] = -0.10 * s
            elif cls == "transient":
                edge_base[target, tf] = 0.35 * s
                sin_coef[0, target, tf] = 0.95 * s
                cos_coef[1, target, tf] = -0.15 * s
            elif cls == "ramp_up":
                edge_base[target, tf] = 0.45 * s
                sin_coef[0, target, tf] = 0.55 * s
                cos_coef[0, target, tf] = -0.35 * s
            elif cls == "ramp_down":
                edge_base[target, tf] = 0.45 * s
                sin_coef[0, target, tf] = -0.55 * s
                cos_coef[0, target, tf] = 0.35 * s
            else:  # static
                edge_base[target, tf] = s
                # tiny residual dynamics so recovery is non-trivial but small
                sin_coef[0, target, tf] = 0.05 * s

        model.edge_base.copy_(torch.tensor(edge_base))
        model.edge_time_sin.copy_(torch.tensor(sin_coef))
        model.edge_time_cos.copy_(torch.tensor(cos_coef))

        for parameter, rate in [
            (model.raw_decay_access, 0.85),
            (model.raw_decay_tf, 1.25),
            (model.raw_decay_enhancer, 0.95),
            (model.raw_decay_target, 0.80),
        ]:
            parameter.fill_(_inverse_softplus(rate))
        model.access_gain.fill_(1.4)
        model.enhancer_tf_gain.fill_(0.35)

        # Accessibility programs with rapid features → identifiable lags.
        model.access_forcing[:, 0] = 0.35
        for tf in range(n_tfs):
            phase = 0.9 * tf
            model.access_forcing[tf, 1] = 1.05 + 0.06 * tf
            model.access_forcing[tf, 2] = -0.55 + 0.08 * tf
            model.access_forcing[tf, 3] = 0.35 * np.sin(phase)
            model.access_forcing[tf, 4] = 0.25 * np.cos(phase)
            model.tf_forcing[tf, 2] = 0.06 * (tf - 2.5)

        # Strong enhancer waves: some closed early, open late (and reverse).
        # This makes gating indispensable for stage-correct predictions.
        model.enhancer_forcing[:, 0] = -0.10
        for enhancer in range(n_enhancers):
            mode = enhancer % 4
            if mode == 0:
                # closed → open
                model.enhancer_forcing[enhancer, 0] = -0.4
                model.enhancer_forcing[enhancer, 1] = 1.4
                model.enhancer_forcing[enhancer, 2] = -0.2
            elif mode == 1:
                # open → closed
                model.enhancer_forcing[enhancer, 0] = 0.5
                model.enhancer_forcing[enhancer, 1] = -1.3
                model.enhancer_forcing[enhancer, 2] = 0.15
            elif mode == 2:
                # mid-open pulse
                model.enhancer_forcing[enhancer, 0] = -0.2
                model.enhancer_forcing[enhancer, 1] = 0.2
                model.enhancer_forcing[enhancer, 3] = 1.2
            else:
                model.enhancer_forcing[enhancer, 0] = 0.15
                model.enhancer_forcing[enhancer, 2] = 0.7 * np.sin(0.5 * enhancer)

    # Initial state: many enhancers start closed (negative logits).
    initial = np.concatenate(
        [
            rng.normal(0.12, 0.07, n_tfs),
            rng.normal(0.08, 0.05, n_tfs),
            rng.normal(-1.2, 0.35, n_enhancers),  # mostly closed at t=0
            rng.normal(0.0, 0.04, n_targets),
        ]
    ).astype(np.float32)
    # Force a few enhancers strongly closed / open at start for clarity.
    for enhancer in range(n_enhancers):
        if enhancer % 4 == 0:
            initial[2 * n_tfs + enhancer] = -2.0
        elif enhancer % 4 == 1:
            initial[2 * n_tfs + enhancer] = 1.5

    times = np.linspace(0.0, 1.0, 61, dtype=np.float32)
    # Refresh continuous delays from peaked logits (may differ slightly).
    delays = model.get_delays()
    return DelayedMultiomeTruth(
        model=model,
        tf_enhancer_prior=tf_enhancer,
        enhancer_target_prior=enhancer_target,
        active_edges=active,
        access_to_tf_delays=delays["access_to_tf"].detach().cpu().numpy(),
        tf_to_target_delays=delays["tf_to_target"].detach().cpu().numpy(),
        times=times,
        initial_state=initial,
        edge_class=classes,
        gated_paths=gated_paths,
    )


@torch.no_grad()
def simulate_dataset(
    truth: DelayedMultiomeTruth,
    noise_sd: float = 0.005,
    seed: int = 0,
    train_tfs: Optional[Sequence[int]] = None,
    heldout_tfs: Optional[Sequence[int]] = None,
    stage_starts: Sequence[float] = (0.12, 0.45, 0.72),
) -> List[Dict[str, object]]:
    """Simulate control + stage-specific TF knockdowns.

    Training typically uses early-stage KDs of train_tfs.
    Held-out uses early/mid/late KDs of heldout_tfs so fixed-W fails.
    """

    rng = np.random.default_rng(seed)
    n_tfs = truth.model.n_tfs
    if train_tfs is None:
        train_tfs = list(range(0, max(1, n_tfs - 2)))
    if heldout_tfs is None:
        heldout_tfs = list(range(max(1, n_tfs - 2), n_tfs))

    device = next(truth.model.parameters()).device
    times = torch.tensor(truth.times, device=device)
    initial = torch.tensor(truth.initial_state, device=device)
    conditions: List[Dict[str, object]] = []

    def add(name, intervention, init=None, role="train"):
        init_t = initial if init is None else torch.tensor(init, device=device)
        init_np = (
            truth.initial_state.copy()
            if init is None
            else np.asarray(init, dtype=np.float32)
        )
        result = truth.model.integrate(
            times, init_t, max_step=0.005, intervention=intervention, method="heun"
        )
        obs = result.obs.detach().cpu().numpy()
        if noise_sd > 0:
            obs = obs + rng.normal(0.0, noise_sd, size=obs.shape)
        conditions.append(
            {
                "name": name,
                "time": truth.times.copy(),
                "obs": obs.astype(np.float32),
                "intervention": intervention,
                "initial_state": init_np,
                "role": role,
            }
        )

    add("control", None, role="train")

    # Training knockdowns: only early stage for train TFs (fixed-W can fit these
    # average effects). Mid-stage train KDs give dynamic edges a training signal.
    for tf in train_tfs:
        add(
            f"TF{tf}_KD_early",
            {"tf_index": int(tf), "start_time": float(stage_starts[0]), "scale": 0.0},
            role="train",
        )
        add(
            f"TF{tf}_KD_mid",
            {"tf_index": int(tf), "start_time": float(stage_starts[1]), "scale": 0.0},
            role="train",
        )

    # Held-out: early / mid / late for each held-out TF.
    for tf in heldout_tfs:
        for stage_name, start in zip(("early", "mid", "late"), stage_starts):
            add(
                f"TF{tf}_KD_{stage_name}",
                {
                    "tf_index": int(tf),
                    "start_time": float(start),
                    "scale": 0.0,
                },
                role="heldout",
            )

    # Control replicates from shifted initial conditions.
    for replicate in range(1, 4):
        shifted = truth.initial_state + rng.normal(
            0.0, 0.18, size=truth.initial_state.shape
        ).astype(np.float32)
        add(f"control_rep{replicate + 1}", None, init=shifted, role="train")

    return conditions


def split_train_heldout(data: List[Dict[str, object]]):
    train = [d for d in data if d.get("role", "train") != "heldout"]
    heldout = [i for i, d in enumerate(data) if d.get("role") == "heldout"]
    return train, heldout
