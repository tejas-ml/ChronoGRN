"""Data-driven initialization for :class:`EnhancerDelayGRN`.

The delayed GRN is conditionally linear once regulatory delays are fixed.  This
module exploits that structure before gradient optimization:

* smooth noisy trajectories and estimate derivatives;
* fit accessibility, TF-RNA and enhancer dynamics by ridge regression;
* select edge-specific TF-to-target delays by coordinate search;
* jointly initialize static/dynamic edge coefficients from absolute dynamics;
* optionally use intervention-minus-control equations for delay profiling.

This is an initializer, not a use of simulation ground truth.  It consumes only
training trajectories, known intervention labels and the supplied pathway prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.signal import savgol_filter

from .enhancer_model import EnhancerDelayGRN


@dataclass
class InitializationReport:
    """Diagnostics returned by :func:`initialize_enhancer_model`."""

    access_to_tf_delays: np.ndarray
    tf_to_target_delays: np.ndarray
    target_delay_improvement: np.ndarray
    target_fit_mse: np.ndarray
    n_absolute_rows: int
    n_response_rows: int


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-5)
    return float(np.log(np.expm1(min(value, 20.0)) + 1e-8))


def _ridge(
    design: np.ndarray,
    response: np.ndarray,
    ridge: float,
    penalty: Optional[np.ndarray] = None,
) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    if penalty is None:
        penalty = np.ones(design.shape[1], dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    gram = design.T @ design
    rhs = design.T @ response
    gram.flat[:: gram.shape[0] + 1] += float(ridge) * penalty
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _scaled_ridge(
    design: np.ndarray,
    response: np.ndarray,
    penalty: np.ndarray,
) -> np.ndarray:
    """Solve generalized ridge after RMS column standardization.

    Delay profiling compares designs whose delayed-input columns can have very
    different scales.  Standardizing before applying the penalty prevents the
    profile score from preferring a lag simply because its interpolated signal
    has a smaller numerical range.
    """

    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale[scale < 1e-8] = 1.0
    standardized = design / scale
    system = standardized.T @ standardized + np.diag(penalty)
    rhs = standardized.T @ response
    try:
        coefficient = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=1e-8)[0]
    return coefficient / scale


def _fourier_basis(times: np.ndarray, frequencies: int) -> np.ndarray:
    """Small nuisance basis used only while profiling access-to-TF delays."""

    values = [np.ones_like(times, dtype=np.float64)]
    for frequency in range(1, max(0, int(frequencies)) + 1):
        values.extend(
            [
                np.sin(2.0 * np.pi * frequency * times),
                np.cos(2.0 * np.pi * frequency * times),
            ]
        )
    return np.stack(values, axis=1)


def _basis(times: np.ndarray, model: EnhancerDelayGRN) -> np.ndarray:
    t = torch.as_tensor(times, dtype=torch.float32)
    with torch.no_grad():
        return model.time_basis_batch(t).cpu().numpy().astype(np.float64)


def _valid_window(length: int, requested: int, polynomial: int) -> int:
    if length <= polynomial + 2:
        return 0
    window = min(int(requested), length if length % 2 else length - 1)
    if window % 2 == 0:
        window -= 1
    return window if window > polynomial else 0


def _smooth(
    times: np.ndarray,
    observations: np.ndarray,
    window: int,
    polynomial: int,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    chosen = _valid_window(len(times), window, polynomial)
    if chosen == 0:
        smooth = observations.copy()
        derivative = np.gradient(smooth, times, axis=0, edge_order=1)
        return smooth, derivative
    dt = float(np.median(np.diff(times)))
    smooth = savgol_filter(
        observations,
        window_length=chosen,
        polyorder=polynomial,
        axis=0,
        mode="interp",
    )
    derivative = savgol_filter(
        observations,
        window_length=chosen,
        polyorder=polynomial,
        deriv=1,
        delta=dt,
        axis=0,
        mode="interp",
    )
    return smooth, derivative


def _lagged_matrix(
    times: np.ndarray,
    values: np.ndarray,
    delays: np.ndarray,
) -> np.ndarray:
    """Return observations × delays × variables with linear interpolation."""

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(
        (len(times), len(delays), values.shape[1]), dtype=np.float64
    )
    for delay_index, delay in enumerate(delays):
        query = np.clip(times - float(delay), times[0], times[-1])
        for variable in range(values.shape[1]):
            output[:, delay_index, variable] = np.interp(
                query, times, values[:, variable]
            )
    return output


def _set_softmax_delays(
    logits: torch.Tensor,
    values: np.ndarray,
    delay_grid: np.ndarray,
    peak: float = 10.0,
) -> None:
    """Set a peaked two-bin mixture whose expectation is ``values``."""

    with torch.no_grad():
        logits.fill_(-peak)
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
        for row, value in enumerate(flat_values):
            value = float(np.clip(value, delay_grid[0], delay_grid[-1]))
            if len(delay_grid) == 1:
                flat_logits[row, 0] = peak
                continue
            right = int(np.searchsorted(delay_grid, value, side="left"))
            right = min(max(right, 1), len(delay_grid) - 1)
            left = right - 1
            span = max(float(delay_grid[right] - delay_grid[left]), 1e-8)
            fraction = (value - float(delay_grid[left])) / span
            flat_logits[row, left] = peak * (1.0 - fraction)
            flat_logits[row, right] = peak * fraction


def _set_delays(
    model: EnhancerDelayGRN,
    access_delays: np.ndarray,
    target_delays: np.ndarray,
) -> None:
    if model.delay_parameterization == "continuous":
        model.set_delays(
            torch.as_tensor(access_delays, dtype=torch.float32),
            torch.as_tensor(target_delays, dtype=torch.float32),
        )
        return
    grid = model.delay_grid.detach().cpu().numpy()
    _set_softmax_delays(model.access_delay_logits, access_delays, grid)
    _set_softmax_delays(model.target_delay_logits, target_delays, grid)


def _prepare(
    model: EnhancerDelayGRN,
    trajectories: Iterable[dict],
    delay_candidates: np.ndarray,
    smooth_window: int,
    polynomial: int,
) -> list[dict]:
    prepared: list[dict] = []
    for index, item in enumerate(trajectories):
        times = np.asarray(item["time"], dtype=np.float64)
        observations = np.asarray(item["obs"], dtype=np.float64)
        smooth, derivative = _smooth(
            times, observations, smooth_window, polynomial
        )
        tf_values = smooth[:, model.tf_slice]
        access_values = smooth[:, model.access_slice]
        prepared.append(
            {
                "name": item.get("name", f"trajectory_{index}"),
                "time": times,
                "obs": smooth,
                "derivative": derivative,
                "basis": _basis(times, model),
                "lagged_tf": _lagged_matrix(
                    times, tf_values, delay_candidates
                ),
                "lagged_access": _lagged_matrix(
                    times, access_values, delay_candidates
                ),
                "intervention": item.get("intervention"),
            }
        )
    return prepared


def _interior_mask(record: dict, margin: int, around_intervention: bool) -> np.ndarray:
    times = record["time"]
    mask = np.ones(len(times), dtype=bool)
    mask[:margin] = False
    mask[-margin:] = False
    intervention = record["intervention"]
    if around_intervention and intervention is not None:
        start = float(intervention.get("start_time", 0.0))
        local_dt = float(np.median(np.diff(times)))
        mask &= np.abs(times - start) > max(2.0 * local_dt, 0.025)
    return mask


def initialize_enhancer_model(
    model: EnhancerDelayGRN,
    trajectories: Iterable[dict],
    *,
    smooth_window: int = 11,
    polynomial: int = 3,
    target_delay_candidates: int = 31,
    coordinate_passes: int = 2,
    ridge: float = 2.0e-3,
    delay_response_weight: float = 0.0,
    delay_control_difference_weight: float = 0.0,
    coefficient_response_weight: float = 0.0,
    initialize_access_delays: bool = True,
    access_smooth_window: int = 13,
    access_profile_candidates: int = 59,
    access_profile_frequencies: int = 1,
    access_forcing_penalty: float = 10.0,
) -> InitializationReport:
    """Initialize model parameters using only the supplied training data.

    Parameters
    ----------
    initialize_access_delays:
        Apply the profiled access→TF delay estimates to the model.  The profile
        uses control trajectories only and a deliberately small one-harmonic
        nuisance basis so flexible time forcing cannot absorb delayed promoter
        signal.  Set this to ``False`` only for a sensitivity analysis.
    delay_response_weight:
        Weight for intervention-minus-control derivative equations during
        TF→target delay selection.  These rows are not used for the final edge
        coefficient refit unless ``coefficient_response_weight`` is positive.
    delay_control_difference_weight:
        Optional weight for differences between control replicates.
    coefficient_response_weight:
        Optional response-row weight during the final coefficient refit.  The
        validated default is zero because absolute equations gave better
        amplitude calibration on the bundled noisy benchmark.
    access_smooth_window:
        Savitzky–Golay window used for access→TF profiling.  This may be wider
        than ``smooth_window`` because the profile is sensitive to derivative
        noise.
    """

    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    if model.max_tau <= 0.0 or model.n_delay_bins <= 1:
        candidate_delays = np.array([0.0], dtype=np.float64)
    else:
        candidate_delays = np.linspace(
            0.0,
            model.max_tau,
            max(model.n_delay_bins, int(target_delay_candidates)),
            dtype=np.float64,
        )
    prepared = _prepare(
        model,
        trajectories,
        candidate_delays,
        smooth_window,
        polynomial,
    )
    controls = [p for p in prepared if p["intervention"] is None]
    if not controls:
        controls = [prepared[0]]
    primary_control = controls[0]
    basis_dim = primary_control["basis"].shape[1]
    margin = max(2, smooth_window // 3)

    # A separate, slightly smoother representation is used to profile the
    # access→TF clock.  The target initializer keeps the narrower window above
    # to retain sharp perturbation responses.
    if model.max_tau <= 0.0 or model.n_delay_bins <= 1:
        access_candidates = np.array([0.0], dtype=np.float64)
    else:
        edge = min(0.005, 0.25 * model.max_tau)
        access_candidates = np.linspace(
            edge,
            max(edge, model.max_tau - edge),
            max(model.n_delay_bins, int(access_profile_candidates)),
            dtype=np.float64,
        )
    access_prepared = _prepare(
        model,
        trajectories,
        access_candidates,
        access_smooth_window,
        polynomial,
    )
    access_controls = [
        record for record in access_prepared if record["intervention"] is None
    ]
    if not access_controls:
        access_controls = [access_prepared[0]]
    access_margin = max(2, access_smooth_window // 2)

    # ------------------------------------------------------------------
    # Accessibility and TF-RNA subsystems.
    # ------------------------------------------------------------------
    access_estimates = np.full(model.n_tfs, 0.0, dtype=np.float64)
    with torch.no_grad():
        for tf in range(model.n_tfs):
            access_design = []
            access_response = []
            for record in controls:
                mask = _interior_mask(record, margin, False)
                access_design.append(
                    np.column_stack(
                        [
                            record["basis"][mask],
                            -record["obs"][mask, model.access_slice][:, tf],
                        ]
                    )
                )
                access_response.append(
                    record["derivative"][mask, model.access_slice][:, tf]
                )
            design = np.concatenate(access_design)
            response = np.concatenate(access_response)
            coefficient = _ridge(design, response, ridge)
            model.access_forcing[tf].copy_(
                torch.as_tensor(
                    coefficient[:-1],
                    dtype=model.access_forcing.dtype,
                    device=model.access_forcing.device,
                )
            )
            model.raw_decay_access[tf] = _inverse_softplus(
                max(0.03, float(coefficient[-1]))
            )

            # Profile the TF equation over candidate lags.  A one-harmonic
            # nuisance basis is intentionally smaller than the complete model
            # basis: otherwise arbitrary TF forcing is nearly collinear with a
            # phase-shifted accessibility trajectory and the delay is not
            # identifiable.
            best = (float("inf"), 0, None)
            for delay_index in range(len(access_candidates)):
                designs = []
                responses = []
                for record in access_controls:
                    mask = _interior_mask(record, access_margin, False)
                    nuisance = _fourier_basis(
                        record["time"], access_profile_frequencies
                    )[mask]
                    designs.append(
                        np.column_stack(
                            [
                                record["lagged_access"][
                                    mask, delay_index, tf
                                ],
                                -record["obs"][mask, model.tf_slice][:, tf],
                                nuisance,
                            ]
                        )
                    )
                    responses.append(
                        record["derivative"][mask, model.tf_slice][:, tf]
                    )
                design = np.concatenate(designs)
                response = np.concatenate(responses)
                nuisance_dim = 1 + 2 * max(
                    0, int(access_profile_frequencies)
                )
                penalty = np.concatenate(
                    [
                        np.asarray([1.0e-4, 1.0e-4]),
                        np.full(nuisance_dim, float(access_forcing_penalty)),
                    ]
                )
                coefficient = _scaled_ridge(design, response, penalty)
                mse = float(np.mean((response - design @ coefficient) ** 2))
                if coefficient[0] < 0.0:
                    mse += 100.0 * float(coefficient[0] ** 2)
                if coefficient[1] < 0.0:
                    mse += 100.0 * float(coefficient[1] ** 2)
                if mse < best[0]:
                    best = (mse, delay_index, coefficient)
            _, best_index, coefficient = best
            access_estimates[tf] = access_candidates[best_index]
            model.access_gain[tf] = max(0.01, float(coefficient[0]))
            model.raw_decay_tf[tf] = _inverse_softplus(
                max(0.03, float(coefficient[1]))
            )
            model.tf_forcing[tf].zero_()
            nuisance_coefficient = coefficient[2:]
            usable = min(
                len(nuisance_coefficient), model.tf_forcing.shape[1]
            )
            model.tf_forcing[tf, :usable].copy_(
                torch.as_tensor(
                    nuisance_coefficient[:usable],
                    dtype=model.tf_forcing.dtype,
                    device=model.tf_forcing.device,
                )
            )

        # Enhancer dynamics are linear conditional on observed TF RNA.
        motif_prior = model.tf_enhancer_prior.detach().cpu().numpy()
        for enhancer in range(model.n_enhancers):
            designs = []
            responses = []
            norm = max(1.0, float(motif_prior[enhancer].sum()))
            for record in prepared:
                mask = _interior_mask(record, margin, True)
                tf_rna = record["obs"][mask, model.tf_slice]
                motif_drive = tf_rna @ motif_prior[enhancer] / norm
                designs.append(
                    np.column_stack(
                        [
                            record["basis"][mask],
                            motif_drive,
                            -record["obs"][mask, model.enhancer_slice][
                                :, enhancer
                            ],
                        ]
                    )
                )
                responses.append(
                    record["derivative"][mask, model.enhancer_slice][
                        :, enhancer
                    ]
                )
            design = np.concatenate(designs)
            response = np.concatenate(responses)
            coefficient = _ridge(design, response, ridge)
            model.enhancer_forcing[enhancer].copy_(
                torch.as_tensor(
                    coefficient[:-2],
                    dtype=model.enhancer_forcing.dtype,
                    device=model.enhancer_forcing.device,
                )
            )
            model.enhancer_tf_gain[enhancer] = max(
                0.0, float(coefficient[-2])
            )
            model.raw_decay_enhancer[enhancer] = _inverse_softplus(
                max(0.03, float(coefficient[-1]))
            )

    # Precompute enhancer gates and lagged regulatory signals.
    tf_enhancer = model.tf_enhancer_prior.detach().cpu().numpy()
    enhancer_target = model.enhancer_target_prior.detach().cpu().numpy()
    denominator = (enhancer_target @ tf_enhancer).clip(min=1e-6)
    for record in prepared:
        if not model.enhancer_gating:
            record["gates"] = np.broadcast_to(
                model.pathway_prior.detach().cpu().numpy()[None, :, :],
                (len(record["time"]), model.n_targets, model.n_tfs),
            ).copy()
        else:
            enhancer = record["obs"][:, model.enhancer_slice]
            open_enhancer = 1.0 / (1.0 + np.exp(-enhancer))
            numerator = np.einsum(
                "gm,mf,bm->bgf",
                enhancer_target,
                tf_enhancer,
                open_enhancer,
            )
            record["gates"] = numerator / denominator[None, :, :]
        record["signal_bank"] = (
            record["gates"][:, None, :, :]
            * record["lagged_tf"][:, :, None, :]
        )

    def interpolate_array(
        source_time: np.ndarray,
        source: np.ndarray,
        query_time: np.ndarray,
    ) -> np.ndarray:
        if np.array_equal(source_time, query_time):
            return source.copy()
        flat = source.reshape(len(source_time), -1)
        result = np.column_stack(
            [
                np.interp(query_time, source_time, flat[:, column])
                for column in range(flat.shape[1])
            ]
        )
        return result.reshape((len(query_time),) + source.shape[1:])

    # Paired control subtraction removes target bias and shared time forcing.
    # These rows are used for delay selection only; final edge coefficients are
    # refit primarily on absolute trajectories to preserve calibrated amplitude.
    for record in prepared:
        if record is primary_control:
            record["delta_obs"] = np.zeros_like(record["obs"])
            record["delta_derivative"] = np.zeros_like(record["derivative"])
            record["delta_signal_bank"] = np.zeros_like(record["signal_bank"])
            continue
        control_obs = interpolate_array(
            primary_control["time"], primary_control["obs"], record["time"]
        )
        control_derivative = interpolate_array(
            primary_control["time"],
            primary_control["derivative"],
            record["time"],
        )
        control_signal = interpolate_array(
            primary_control["time"],
            primary_control["signal_bank"],
            record["time"],
        )
        record["delta_obs"] = record["obs"] - control_obs
        record["delta_derivative"] = record["derivative"] - control_derivative
        record["delta_signal_bank"] = record["signal_bank"] - control_signal

    possible = model.pathway_prior.detach().cpu().numpy() > 0
    selected = np.zeros((model.n_targets, model.n_tfs), dtype=np.int64)
    delay_improvement = np.zeros_like(selected, dtype=np.float64)
    target_fit_mse = np.zeros(model.n_targets, dtype=np.float64)
    edge_basis_dim = (
        1 + 2 * model.n_time_freqs if model.dynamic_edges else 1
    )

    def target_design(
        target: int,
        candidates: list[int],
        selection: dict[int, int],
        *,
        response_weight: float,
        control_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        blocks: list[np.ndarray] = []
        outcomes: list[np.ndarray] = []

        # Absolute derivative equations.
        for record in prepared:
            mask = _interior_mask(record, margin, True)
            edge_basis = (
                record["basis"][mask]
                if model.dynamic_edges
                else np.ones((int(mask.sum()), 1), dtype=np.float64)
            )
            columns: list[np.ndarray] = [
                np.ones((int(mask.sum()), 1), dtype=np.float64),
                -record["obs"][mask, model.target_slice][:, [target]],
            ]
            for tf in candidates:
                delay_index = int(selection[tf])
                signal = record["signal_bank"][
                    mask, delay_index, target, tf
                ]
                columns.append(signal[:, None] * edge_basis)
            blocks.append(np.concatenate(columns, axis=1))
            outcomes.append(
                record["derivative"][mask, model.target_slice][:, target]
            )

        # Intervention/control-difference equations.
        for record in prepared:
            if record is primary_control:
                continue
            row_weight = (
                float(response_weight)
                if record["intervention"] is not None
                else float(control_weight)
            )
            if row_weight <= 0.0:
                continue
            mask = _interior_mask(record, margin, True)
            edge_basis = (
                record["basis"][mask]
                if model.dynamic_edges
                else np.ones((int(mask.sum()), 1), dtype=np.float64)
            )
            columns = [
                np.zeros((int(mask.sum()), 1), dtype=np.float64),
                -record["delta_obs"][mask, model.target_slice][:, [target]],
            ]
            for tf in candidates:
                delay_index = int(selection[tf])
                signal_delta = record["delta_signal_bank"][
                    mask, delay_index, target, tf
                ]
                columns.append(signal_delta[:, None] * edge_basis)
            scale = float(np.sqrt(row_weight))
            blocks.append(scale * np.concatenate(columns, axis=1))
            outcomes.append(
                scale
                * record["delta_derivative"][
                    mask, model.target_slice
                ][:, target]
            )

        return np.concatenate(blocks), np.concatenate(outcomes)

    with torch.no_grad():
        for target in range(model.n_targets):
            candidates = np.flatnonzero(possible[target]).tolist()
            if not candidates:
                continue

            # Univariate profile start.
            for tf in candidates:
                best_mse = float("inf")
                best_index = 0
                for delay_index in range(len(candidate_delays)):
                    design, response = target_design(
                        target,
                        [tf],
                        {tf: delay_index},
                        response_weight=delay_response_weight,
                        control_weight=delay_control_difference_weight,
                    )
                    penalty = np.concatenate(
                        [[0.0, 0.2], np.ones(edge_basis_dim)]
                    )
                    coefficient = _ridge(design, response, ridge, penalty)
                    mse = float(np.mean((response - design @ coefficient) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_index = delay_index
                selected[target, tf] = best_index

            selection = {
                tf: int(selected[target, tf]) for tf in candidates
            }
            penalty = np.concatenate(
                [
                    [0.0, 0.2],
                    np.ones(len(candidates) * edge_basis_dim),
                ]
            )
            baseline_design, baseline_response = target_design(
                target,
                candidates,
                selection,
                response_weight=delay_response_weight,
                control_weight=delay_control_difference_weight,
            )
            baseline_coefficient = _ridge(
                baseline_design, baseline_response, ridge, penalty
            )
            baseline_mse = float(
                np.mean(
                    (baseline_response - baseline_design @ baseline_coefficient)
                    ** 2
                )
            )

            # Coordinate profile refinement for multiple candidate regulators.
            for _ in range(max(1, int(coordinate_passes))):
                for tf in candidates:
                    best_mse = float("inf")
                    best_index = int(selection[tf])
                    for delay_index in range(len(candidate_delays)):
                        selection[tf] = delay_index
                        design, response = target_design(
                            target,
                            candidates,
                            selection,
                            response_weight=delay_response_weight,
                            control_weight=delay_control_difference_weight,
                        )
                        coefficient = _ridge(
                            design, response, ridge, penalty
                        )
                        mse = float(
                            np.mean((response - design @ coefficient) ** 2)
                        )
                        if mse < best_mse:
                            best_mse = mse
                            best_index = delay_index
                    selection[tf] = best_index

            # Final coefficient calibration.  By default response rows are used
            # only to choose delays; setting coefficient_response_weight > 0 is
            # available for unusually high-SNR perturbation experiments.
            design, response = target_design(
                target,
                candidates,
                selection,
                response_weight=coefficient_response_weight,
                control_weight=0.0,
            )
            coefficient = _ridge(design, response, ridge, penalty)
            final_mse = float(np.mean((response - design @ coefficient) ** 2))
            target_fit_mse[target] = final_mse
            model.target_bias[target] = float(coefficient[0])
            model.target_forcing[target].zero_()
            model.raw_decay_target[target] = _inverse_softplus(
                max(0.03, float(coefficient[1]))
            )
            position = 2
            for tf in candidates:
                group = coefficient[position : position + edge_basis_dim]
                position += edge_basis_dim
                model.edge_base[target, tf] = float(group[0])
                if model.dynamic_edges:
                    for frequency in range(model.n_time_freqs):
                        model.edge_time_sin[frequency, target, tf] = float(
                            group[1 + 2 * frequency]
                        )
                        model.edge_time_cos[frequency, target, tf] = float(
                            group[2 + 2 * frequency]
                        )
                selected[target, tf] = int(selection[tf])
                delay_improvement[target, tf] = max(
                    0.0, baseline_mse - final_mse
                )

    target_delays = candidate_delays[selected]
    existing = model.get_delays()
    access_to_set = (
        access_estimates
        if initialize_access_delays
        else existing["access_to_tf"].detach().cpu().numpy()
    )
    _set_delays(model, access_to_set, target_delays)

    n_absolute_rows = sum(
        int(_interior_mask(record, margin, True).sum()) for record in prepared
    )
    n_response_rows = sum(
        int(_interior_mask(record, margin, True).sum())
        for record in prepared
        if record["intervention"] is not None
    )
    return InitializationReport(
        access_to_tf_delays=access_estimates.astype(np.float32),
        tf_to_target_delays=target_delays.astype(np.float32),
        target_delay_improvement=delay_improvement.astype(np.float32),
        target_fit_mse=target_fit_mse.astype(np.float32),
        n_absolute_rows=n_absolute_rows,
        n_response_rows=n_response_rows,
    )
