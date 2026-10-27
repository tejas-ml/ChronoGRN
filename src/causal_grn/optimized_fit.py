"""Leakage-safe high-performance fitting recipe for ChronoGRN.

A delayed dynamic GRN is difficult to identify when delays, edge trajectories,
decay terms, and smooth background forcing are optimized jointly from a random
initialization.  This module therefore uses a structured two-stage estimator:

1. Savitzky-Golay smoothing plus ridge/profile regression estimates the four
   biological subsystems and continuous delays from *training trajectories
   only*.
2. An optional conservative neural fine-tune can refine rollout and response
   losses while keeping the profiled delays fixed.

On the bundled short-trajectory positive-control benchmark, the structured
initializer by itself gives the strongest multi-objective result.  Consequently
fine-tuning is disabled by default rather than being applied unconditionally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import torch

from .enhancer_initialize import InitializationReport, initialize_enhancer_model
from .enhancer_model import EnhancerDelayGRN
from .enhancer_train import EnhancerFit, fit_enhancer_model


@dataclass(frozen=True)
class OptimizedFitConfig:
    """Configuration for :func:`fit_optimized_chronogrn`.

    The defaults were selected using training-only positive-control experiments
    across independent synthetic networks.  They are conservative starting
    points for real data, not universal hyperparameters.
    """

    # Structured subsystem and delay initialization.
    smooth_window: int = 11
    smooth_polynomial: int = 3
    target_delay_candidates: int = 31
    coordinate_passes: int = 2
    ridge: float = 2.0e-3
    delay_response_weight: float = 0.0
    delay_control_difference_weight: float = 0.0
    coefficient_response_weight: float = 0.0
    initialize_access_delays: bool = True
    access_smooth_window: int = 13
    access_profile_candidates: int = 59
    access_profile_frequencies: int = 1
    access_forcing_penalty: float = 10.0

    # Optional conservative end-to-end fine-tune.  Disabled by default because
    # it can trade mechanistic identifiability for a small rollout improvement.
    finetune_epochs: int = 0
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    edge_group_l1: float = 1.0e-3
    dynamic_l2: float = 5.0e-5
    target_forcing_l2: float = 1.0e-2
    response_derivative_weight: float = 0.20
    rollout_weight: float = 0.25
    response_weight: float = 0.40
    rollout_start_epoch: int = 2
    response_start_epoch: int = 2
    rollout_every: int = 2
    auxiliary_trajectory_batch: int = 5
    auxiliary_response_batch: int = 4
    patience: int = 25
    seed: int = 31

    # Profile estimates are substantially more stable than free delay updates
    # for short and noisy trajectories.
    freeze_access_delays: bool = True
    freeze_target_delays: bool = True


@dataclass
class OptimizedFitResult:
    """Fitted model together with initialization and optimizer diagnostics."""

    model: EnhancerDelayGRN
    initialization: InitializationReport
    history: list[dict[str, float]]
    config: OptimizedFitConfig

    def checkpoint_payload(self) -> dict:
        """Return a self-contained payload suitable for :func:`torch.save`."""

        model = self.model
        return {
            "format_version": 2,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "tf_enhancer_prior": model.tf_enhancer_prior.detach().cpu(),
            "enhancer_target_prior": model.enhancer_target_prior.detach().cpu(),
            "model_kwargs": {
                "n_tfs": model.n_tfs,
                "n_enhancers": model.n_enhancers,
                "n_targets": model.n_targets,
                "max_tau": model.max_tau,
                "n_delay_bins": model.n_delay_bins,
                "n_time_freqs": model.n_time_freqs,
                "forcing_basis": model.forcing_basis,
                "dynamic_edges": model.dynamic_edges,
                "enhancer_gating": model.enhancer_gating,
                "delay_parameterization": model.delay_parameterization,
            },
            "optimized_fit_config": asdict(self.config),
            "initialization": asdict(self.initialization),
            "history": self.history,
        }


def _set_delay_trainability(
    model: EnhancerDelayGRN,
    *,
    freeze_access: bool,
    freeze_target: bool,
) -> None:
    """Set gradient flags for either supported delay parameterization."""

    if model.delay_parameterization == "softmax":
        model.access_delay_logits.requires_grad_(not freeze_access)
        model.target_delay_logits.requires_grad_(not freeze_target)
    else:
        model.raw_access_delay.requires_grad_(not freeze_access)
        model.raw_target_delay.requires_grad_(not freeze_target)


def fit_optimized_chronogrn(
    model: EnhancerDelayGRN,
    trajectories: Iterable[dict],
    *,
    config: Optional[OptimizedFitConfig] = None,
    device: Optional[torch.device] = None,
) -> OptimizedFitResult:
    """Fit ChronoGRN without inspecting held-out conditions or simulation truth.

    Parameters
    ----------
    model:
        Configured enhancer-resolved ChronoGRN.  For short trajectories the
        recommended architecture uses one Fourier frequency and continuous
        delays.
    trajectories:
        Training trajectories only.  Do not include held-out interventions.
    config:
        Structured initialization and optional fine-tuning settings.
    device:
        Device for the optional neural fine-tune.  Structured initialization is
        a small CPU regression problem.
    """

    chosen = config or OptimizedFitConfig()
    trajectory_list = list(trajectories)
    if not trajectory_list:
        raise ValueError("At least one training trajectory is required")

    initialization = initialize_enhancer_model(
        model,
        trajectory_list,
        smooth_window=chosen.smooth_window,
        polynomial=chosen.smooth_polynomial,
        target_delay_candidates=chosen.target_delay_candidates,
        coordinate_passes=chosen.coordinate_passes,
        ridge=chosen.ridge,
        delay_response_weight=chosen.delay_response_weight,
        delay_control_difference_weight=chosen.delay_control_difference_weight,
        coefficient_response_weight=chosen.coefficient_response_weight,
        initialize_access_delays=chosen.initialize_access_delays,
        access_smooth_window=chosen.access_smooth_window,
        access_profile_candidates=chosen.access_profile_candidates,
        access_profile_frequencies=chosen.access_profile_frequencies,
        access_forcing_penalty=chosen.access_forcing_penalty,
    )

    _set_delay_trainability(
        model,
        freeze_access=chosen.freeze_access_delays,
        freeze_target=chosen.freeze_target_delays,
    )

    if chosen.finetune_epochs <= 0:
        model.clear_reference()
        return OptimizedFitResult(
            model=model,
            initialization=initialization,
            history=[],
            config=chosen,
        )

    fitted: EnhancerFit = fit_enhancer_model(
        model,
        trajectory_list,
        epochs=chosen.finetune_epochs,
        lr=chosen.learning_rate,
        weight_decay=chosen.weight_decay,
        edge_group_l1=chosen.edge_group_l1,
        dynamic_l2=chosen.dynamic_l2,
        target_forcing_l2=chosen.target_forcing_l2,
        delay_entropy=0.0,
        delay_lr_scale=0.0,
        response_derivative_weight=chosen.response_derivative_weight,
        rollout_weight=chosen.rollout_weight,
        response_weight=chosen.response_weight,
        rollout_start_epoch=chosen.rollout_start_epoch,
        response_start_epoch=chosen.response_start_epoch,
        rollout_every=chosen.rollout_every,
        aux_trajectory_batch=chosen.auxiliary_trajectory_batch,
        aux_response_batch=chosen.auxiliary_response_batch,
        patience=chosen.patience,
        seed=chosen.seed,
        device=device,
    )
    return OptimizedFitResult(
        model=fitted.model,
        initialization=initialization,
        history=fitted.history,
        config=chosen,
    )
