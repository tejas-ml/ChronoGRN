"""Checkpoint utilities for :class:`~causal_grn.enhancer_model.EnhancerDelayGRN`.

New checkpoints are self-contained: candidate-path priors and model metadata are
stored together with the state dictionary.  The loader remains compatible with
legacy ChronoGRN checkpoints, which require priors to be supplied explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .enhancer_model import EnhancerDelayGRN


_MODEL_KWARGS = {
    "n_tfs",
    "n_enhancers",
    "n_targets",
    "max_tau",
    "n_delay_bins",
    "n_time_freqs",
    "forcing_basis",
    "dynamic_edges",
    "enhancer_gating",
    "delay_parameterization",
    "initial_delay",
}


def model_configuration(model: EnhancerDelayGRN) -> dict[str, Any]:
    """Return constructor arguments needed to recreate ``model``."""

    return {
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
    }


def save_checkpoint(
    path: str | Path,
    model: EnhancerDelayGRN,
    *,
    history: Optional[list[dict[str, float]]] = None,
    calibration: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save a self-contained ChronoGRN checkpoint.

    The priors are stored on CPU so the checkpoint can be loaded without
    rebuilding a synthetic benchmark or preprocessing pipeline.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 2,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_kwargs": model_configuration(model),
        "tf_enhancer_prior": model.tf_enhancer_prior.detach().cpu(),
        "enhancer_target_prior": model.enhancer_target_prior.detach().cpu(),
    }
    if history is not None:
        payload["history"] = history
    if calibration is not None:
        payload["calibration"] = dict(calibration)
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    tf_enhancer_prior: Optional[torch.Tensor] = None,
    enhancer_target_prior: Optional[torch.Tensor] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> tuple[EnhancerDelayGRN, dict[str, Any]]:
    """Load a current or legacy ChronoGRN checkpoint.

    Parameters
    ----------
    path:
        ``.pt`` checkpoint path.
    tf_enhancer_prior, enhancer_target_prior:
        Required only for legacy checkpoints that did not store their priors.
    strict:
        Passed to ``load_state_dict``.  The default is ``False`` so legacy
        softmax-delay checkpoints can be upgraded after continuous-delay
        parameters were added.
    """

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path!s} is not a ChronoGRN checkpoint")

    if tf_enhancer_prior is None:
        tf_enhancer_prior = payload.get("tf_enhancer_prior")
    if enhancer_target_prior is None:
        enhancer_target_prior = payload.get("enhancer_target_prior")
    if tf_enhancer_prior is None or enhancer_target_prior is None:
        raise ValueError(
            "legacy checkpoint does not contain priors; pass "
            "tf_enhancer_prior and enhancer_target_prior"
        )

    raw_kwargs = dict(payload.get("model_kwargs", {}))
    kwargs = {key: value for key, value in raw_kwargs.items() if key in _MODEL_KWARGS}
    kwargs.setdefault("delay_parameterization", "softmax")
    model = EnhancerDelayGRN(
        tf_enhancer_prior=torch.as_tensor(tf_enhancer_prior, dtype=torch.float32),
        enhancer_target_prior=torch.as_tensor(
            enhancer_target_prior, dtype=torch.float32
        ),
        **kwargs,
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=strict)
    payload["missing_keys"] = list(incompatible.missing_keys)
    payload["unexpected_keys"] = list(incompatible.unexpected_keys)
    return model, payload
