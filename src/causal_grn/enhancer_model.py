from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EnhancerIntegrationResult:
    """Output of an enhancer-resolved delayed rollout."""

    obs: torch.Tensor
    edges: torch.Tensor
    enhancer_gates: torch.Tensor
    access_to_tf_delays: torch.Tensor
    tf_to_target_delays: torch.Tensor


class EnhancerDelayGRN(nn.Module):
    """Continuous-time GRN with enhancer gates and separately learned delays.

    State order is ``[TF promoter accessibility, TF RNA, enhancer
    accessibility, target RNA]``.  Regulatory target drive is

    ``W_target,TF(t) * TF_RNA(t - tau_target,TF) * enhancer_gate(t)``.

    Delays can be represented either as soft mixtures over a fixed lag grid
    (legacy-compatible mode) or as bounded continuous scalars with adjacent-bin
    interpolation.  Both modes provide TF-specific accessibility→TF delays and
    edge-specific TF→target delays without one DDE interpolation per edge.
    """

    def __init__(
        self,
        n_tfs: int,
        n_enhancers: int,
        n_targets: int,
        tf_enhancer_prior: torch.Tensor,
        enhancer_target_prior: torch.Tensor,
        max_tau: float = 0.35,
        n_delay_bins: int = 8,
        n_time_freqs: int = 3,
        forcing_basis: str = "fourier",
        dynamic_edges: bool = True,
        enhancer_gating: bool = True,
        delay_parameterization: str = "softmax",
        initial_delay: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.n_tfs = int(n_tfs)
        self.n_enhancers = int(n_enhancers)
        self.n_targets = int(n_targets)
        self.max_tau = float(max_tau)
        self.n_delay_bins = int(n_delay_bins)
        self.n_time_freqs = int(n_time_freqs)
        if min(self.n_tfs, self.n_enhancers, self.n_targets) <= 0:
            raise ValueError("state dimensions must all be positive")
        if self.max_tau < 0.0:
            raise ValueError("max_tau must be non-negative")
        if self.n_delay_bins < 1:
            raise ValueError("n_delay_bins must be at least one")
        if self.n_time_freqs < 0:
            raise ValueError("n_time_freqs must be non-negative")
        if forcing_basis not in {"fourier", "rbf"}:
            raise ValueError("forcing_basis must be 'fourier' or 'rbf'")
        self.forcing_basis = str(forcing_basis)
        self.dynamic_edges = bool(dynamic_edges)
        self.enhancer_gating = bool(enhancer_gating)
        if delay_parameterization not in {"softmax", "continuous"}:
            raise ValueError(
                "delay_parameterization must be 'softmax' or 'continuous'"
            )
        self.delay_parameterization = str(delay_parameterization)

        if tf_enhancer_prior.shape != (self.n_enhancers, self.n_tfs):
            raise ValueError(
                "tf_enhancer_prior must have shape "
                f"({self.n_enhancers}, {self.n_tfs})"
            )
        if enhancer_target_prior.shape != (self.n_targets, self.n_enhancers):
            raise ValueError(
                "enhancer_target_prior must have shape "
                f"({self.n_targets}, {self.n_enhancers})"
            )

        tf_enhancer = torch.clamp(tf_enhancer_prior.float(), min=0.0)
        enhancer_target = torch.clamp(enhancer_target_prior.float(), min=0.0)
        pathway = (enhancer_target @ tf_enhancer > 0).float()
        self.register_buffer("tf_enhancer_prior", tf_enhancer)
        self.register_buffer("enhancer_target_prior", enhancer_target)
        self.register_buffer("pathway_prior", pathway)
        self.register_buffer(
            "delay_grid", torch.linspace(0.0, self.max_tau, self.n_delay_bins)
        )
        self.register_buffer(
            "forcing_centers",
            torch.linspace(0.0, 1.0, max(1, 2 * self.n_time_freqs)),
        )
        self.register_buffer("reference_times", torch.empty(0))
        self.register_buffer("reference_obs", torch.empty(0, self.obs_dim))
        self.register_buffer(
            "reference_derivatives", torch.empty(0, self.obs_dim)
        )

        # Two delay parameterizations are supported.  The original softmax
        # mixture is retained for checkpoint compatibility.  The continuous
        # mode uses one bounded scalar per TF/edge and linearly interpolates
        # between adjacent lag-grid states.  This removes multimodal lag
        # mixtures and substantially improves delay calibration.
        if initial_delay is None or self.max_tau <= 0.0:
            # A uniform lag prior is neutral and avoids the systematic
            # short-delay bias produced by the former exponentially decaying
            # initialization.
            initial_delay_logits = torch.zeros(self.n_delay_bins)
        else:
            width = max(
                self.max_tau / max(1, self.n_delay_bins - 1),
                0.08 * self.max_tau,
            )
            initial_delay_logits = -0.5 * (
                (self.delay_grid - float(initial_delay)) / max(width, 1e-6)
            ) ** 2
        self.access_delay_logits = nn.Parameter(
            initial_delay_logits.repeat(self.n_tfs, 1),
            requires_grad=self.delay_parameterization == "softmax",
        )
        self.target_delay_logits = nn.Parameter(
            initial_delay_logits.repeat(self.n_targets, self.n_tfs, 1),
            requires_grad=self.delay_parameterization == "softmax",
        )
        if self.max_tau <= 0.0 or self.n_delay_bins <= 1:
            initial_fraction = 0.0
        else:
            default_delay = min(0.10, 0.45 * self.max_tau)
            chosen_delay = (
                default_delay if initial_delay is None else float(initial_delay)
            )
            initial_fraction = min(
                1.0 - 1e-4, max(1e-4, chosen_delay / self.max_tau)
            )
        initial_raw = (
            math.log(initial_fraction / (1.0 - initial_fraction))
            if 0.0 < initial_fraction < 1.0
            else -12.0
        )
        self.raw_access_delay = nn.Parameter(
            torch.full((self.n_tfs,), float(initial_raw)),
            requires_grad=self.delay_parameterization == "continuous"
            and self.max_tau > 0.0,
        )
        self.raw_target_delay = nn.Parameter(
            torch.full((self.n_targets, self.n_tfs), float(initial_raw)),
            requires_grad=self.delay_parameterization == "continuous"
            and self.max_tau > 0.0,
        )

        self.edge_base = nn.Parameter(
            0.03 * torch.randn(self.n_targets, self.n_tfs)
        )
        self.edge_time_sin = nn.Parameter(
            torch.zeros(self.n_time_freqs, self.n_targets, self.n_tfs)
        )
        self.edge_time_cos = nn.Parameter(
            torch.zeros(self.n_time_freqs, self.n_targets, self.n_tfs)
        )

        self.raw_decay_access = nn.Parameter(torch.zeros(self.n_tfs))
        self.raw_decay_tf = nn.Parameter(torch.zeros(self.n_tfs))
        self.raw_decay_enhancer = nn.Parameter(torch.zeros(self.n_enhancers))
        self.raw_decay_target = nn.Parameter(torch.zeros(self.n_targets))
        self.access_gain = nn.Parameter(torch.ones(self.n_tfs))
        self.enhancer_tf_gain = nn.Parameter(
            0.05 * torch.ones(self.n_enhancers)
        )
        self.target_bias = nn.Parameter(torch.zeros(self.n_targets))

        # Smooth exogenous forcing is required to model accessibility waves that
        # are not themselves explained by the selected TF set.
        basis_dim = 1 + 2 * self.n_time_freqs
        self.access_forcing = nn.Parameter(
            torch.zeros(self.n_tfs, basis_dim)
        )
        self.tf_forcing = nn.Parameter(torch.zeros(self.n_tfs, basis_dim))
        self.enhancer_forcing = nn.Parameter(
            torch.zeros(self.n_enhancers, basis_dim)
        )
        # A shared developmental trajectory contains target programs driven by
        # regulators outside the selected TF panel.  This nuisance forcing
        # captures that common background; it is identical in control and
        # counterfactual rollouts, so TF-dependent perturbation differences do
        # not inherit it.
        self.target_forcing = nn.Parameter(
            torch.zeros(self.n_targets, basis_dim)
        )

    @property
    def obs_dim(self) -> int:
        return 2 * self.n_tfs + self.n_enhancers + self.n_targets

    @property
    def access_slice(self) -> slice:
        return slice(0, self.n_tfs)

    @property
    def tf_slice(self) -> slice:
        return slice(self.n_tfs, 2 * self.n_tfs)

    @property
    def enhancer_slice(self) -> slice:
        return slice(2 * self.n_tfs, 2 * self.n_tfs + self.n_enhancers)

    @property
    def target_slice(self) -> slice:
        return slice(2 * self.n_tfs + self.n_enhancers, self.obs_dim)

    def split_state(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            state[..., self.access_slice],
            state[..., self.tf_slice],
            state[..., self.enhancer_slice],
            state[..., self.target_slice],
        )

    def set_reference_trajectory(
        self, times: torch.Tensor, observations: torch.Tensor
    ) -> None:
        """Condition the vector field on an observed control trajectory.

        The corrected field is ``f(y,t) - f(y_ref,t) + dy_ref/dt``.  Thus the
        reference is a solution by construction, while counterfactual
        deviations are still generated by the learned delayed GRN.
        """

        device = self.delay_grid.device
        times = torch.as_tensor(times, dtype=torch.float32, device=device)
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=device
        )
        if (
            times.ndim != 1
            or observations.shape != (len(times), self.obs_dim)
            or len(times) < 2
        ):
            raise ValueError("invalid reference trajectory shape")
        if not torch.all(times[1:] > times[:-1]):
            raise ValueError("reference times must be strictly increasing")
        slopes = (observations[1:] - observations[:-1]) / (
            times[1:] - times[:-1]
        )[:, None]
        derivatives = torch.empty_like(observations)
        derivatives[0] = slopes[0]
        derivatives[-1] = slopes[-1]
        if len(times) > 2:
            derivatives[1:-1] = 0.5 * (slopes[:-1] + slopes[1:])
        self.reference_times = times
        self.reference_obs = observations
        self.reference_derivatives = derivatives

    def clear_reference(self) -> None:
        """Remove any conditioned control trajectory."""

        device = self.delay_grid.device
        self.reference_times = torch.empty(0, device=device)
        self.reference_obs = torch.empty(0, self.obs_dim, device=device)
        self.reference_derivatives = torch.empty(
            0, self.obs_dim, device=device
        )

    @property
    def has_reference(self) -> bool:
        return len(self.reference_times) >= 2

    def _reference_interpolate(
        self, query: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        flat = query.reshape(-1).to(
            dtype=self.reference_times.dtype, device=self.reference_times.device
        )
        right = torch.searchsorted(self.reference_times, flat, right=False)
        right = right.clamp(1, len(self.reference_times) - 1)
        left = right - 1
        t0 = self.reference_times[left]
        t1 = self.reference_times[right]
        weight = ((flat - t0) / (t1 - t0).clamp_min(1e-8)).clamp(0.0, 1.0)
        interpolated = (
            values[left] * (1.0 - weight[:, None])
            + values[right] * weight[:, None]
        )
        return interpolated.reshape(*query.shape, self.obs_dim)

    def reference_state(self, query: torch.Tensor) -> torch.Tensor:
        if not self.has_reference:
            raise RuntimeError("No reference trajectory has been set")
        return self._reference_interpolate(query, self.reference_obs)

    def reference_derivative(self, query: torch.Tensor) -> torch.Tensor:
        if not self.has_reference:
            raise RuntimeError("No reference trajectory has been set")
        return self._reference_interpolate(
            query, self.reference_derivatives
        )

    def time_basis(self, t: torch.Tensor) -> torch.Tensor:
        values = [torch.ones_like(t)]
        if self.forcing_basis == "fourier":
            for k in range(1, self.n_time_freqs + 1):
                values.extend(
                    [
                        torch.sin(2.0 * math.pi * k * t),
                        torch.cos(2.0 * math.pi * k * t),
                    ]
                )
        else:
            spacing = 1.0 / max(1, len(self.forcing_centers) - 1)
            width = 1.35 * spacing
            values.extend(
                [
                    torch.exp(-0.5 * ((t - center) / width) ** 2)
                    for center in self.forcing_centers
                ]
            )
        return torch.stack(values)

    def time_basis_batch(self, t: torch.Tensor) -> torch.Tensor:
        """Return time features with shape observations × features."""

        values = [torch.ones_like(t)]
        if self.forcing_basis == "fourier":
            for k in range(1, self.n_time_freqs + 1):
                values.extend(
                    [
                        torch.sin(2.0 * math.pi * k * t),
                        torch.cos(2.0 * math.pi * k * t),
                    ]
                )
        else:
            spacing = 1.0 / max(1, len(self.forcing_centers) - 1)
            width = 1.35 * spacing
            values.extend(
                [
                    torch.exp(-0.5 * ((t - center) / width) ** 2)
                    for center in self.forcing_centers
                ]
            )
        return torch.stack(values, dim=-1)

    def _continuous_delays(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_tau <= 0.0 or self.n_delay_bins <= 1:
            return (
                torch.zeros_like(self.raw_access_delay),
                torch.zeros_like(self.raw_target_delay),
            )
        return (
            self.max_tau * torch.sigmoid(self.raw_access_delay),
            self.max_tau * torch.sigmoid(self.raw_target_delay),
        )

    def _linear_lag_weights(self, delays: torch.Tensor) -> torch.Tensor:
        """Convert continuous delays to adjacent-bin interpolation weights.

        The triangular kernel is exactly linear between neighboring lag bins,
        sums to one, and remains differentiable with respect to the delay away
        from grid knots.
        """

        if self.n_delay_bins <= 1 or self.max_tau <= 0.0:
            return torch.ones(
                *delays.shape, 1, dtype=delays.dtype, device=delays.device
            )
        step = self.max_tau / float(self.n_delay_bins - 1)
        distance = torch.abs(delays[..., None] - self.delay_grid)
        weights = torch.clamp(1.0 - distance / max(step, 1e-8), min=0.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def delay_probabilities(
        self, temperature: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.delay_parameterization == "continuous":
            access_delay, target_delay = self._continuous_delays()
            return (
                self._linear_lag_weights(access_delay),
                self._linear_lag_weights(target_delay),
            )
        temperature = max(float(temperature), 1e-3)
        access = F.softmax(self.access_delay_logits / temperature, dim=-1)
        target = F.softmax(self.target_delay_logits / temperature, dim=-1)
        return access, target

    def get_delays(self) -> Dict[str, torch.Tensor]:
        if self.delay_parameterization == "continuous":
            access_delay, target_delay = self._continuous_delays()
            return {
                "access_to_tf": access_delay,
                "tf_to_target": target_delay,
            }
        access_prob, target_prob = self.delay_probabilities()
        return {
            "access_to_tf": access_prob @ self.delay_grid,
            "tf_to_target": torch.sum(
                target_prob * self.delay_grid.view(1, 1, -1), dim=-1
            ),
        }

    def set_delays(
        self,
        access_to_tf: Optional[torch.Tensor] = None,
        tf_to_target: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize delays from estimates in trajectory-time units."""

        if self.delay_parameterization != "continuous":
            raise RuntimeError("set_delays is available in continuous mode only")
        if self.max_tau <= 0.0:
            return

        def inverse(delay: torch.Tensor) -> torch.Tensor:
            fraction = (delay / self.max_tau).clamp(1e-4, 1.0 - 1e-4)
            return torch.log(fraction) - torch.log1p(-fraction)

        with torch.no_grad():
            if access_to_tf is not None:
                value = torch.as_tensor(
                    access_to_tf,
                    dtype=self.raw_access_delay.dtype,
                    device=self.raw_access_delay.device,
                )
                if value.shape != self.raw_access_delay.shape:
                    raise ValueError("invalid accessibility-to-TF delay shape")
                self.raw_access_delay.copy_(inverse(value))
            if tf_to_target is not None:
                value = torch.as_tensor(
                    tf_to_target,
                    dtype=self.raw_target_delay.dtype,
                    device=self.raw_target_delay.device,
                )
                if value.shape != self.raw_target_delay.shape:
                    raise ValueError("invalid TF-to-target delay shape")
                self.raw_target_delay.copy_(inverse(value))

    def edge_weights(self, t: torch.Tensor) -> torch.Tensor:
        weights = self.edge_base
        if self.dynamic_edges:
            for k in range(self.n_time_freqs):
                freq = 2.0 * math.pi * (k + 1)
                weights = (
                    weights
                    + self.edge_time_sin[k] * torch.sin(freq * t)
                    + self.edge_time_cos[k] * torch.cos(freq * t)
                )
        return weights * self.pathway_prior

    def edge_weights_batch(self, t: torch.Tensor) -> torch.Tensor:
        weights = self.edge_base.unsqueeze(0).expand(
            len(t), self.n_targets, self.n_tfs
        )
        if self.dynamic_edges:
            for k in range(self.n_time_freqs):
                freq = 2.0 * math.pi * (k + 1)
                weights = (
                    weights
                    + self.edge_time_sin[k].unsqueeze(0)
                    * torch.sin(freq * t)[:, None, None]
                    + self.edge_time_cos[k].unsqueeze(0)
                    * torch.cos(freq * t)[:, None, None]
                )
        return weights * self.pathway_prior.unsqueeze(0)

    def enhancer_gate(self, enhancer_access: torch.Tensor) -> torch.Tensor:
        """Return a target-by-TF gate supported by explicit enhancer paths."""

        if not self.enhancer_gating:
            return self.pathway_prior
        open_enhancer = torch.sigmoid(enhancer_access)
        weighted_motifs = self.tf_enhancer_prior * open_enhancer[:, None]
        numerator = self.enhancer_target_prior @ weighted_motifs
        denominator = (
            self.enhancer_target_prior @ self.tf_enhancer_prior
        ).clamp_min(1e-6)
        return numerator / denominator

    def enhancer_gate_batch(
        self, enhancer_access: torch.Tensor
    ) -> torch.Tensor:
        if not self.enhancer_gating:
            return self.pathway_prior.unsqueeze(0).expand(
                len(enhancer_access), self.n_targets, self.n_tfs
            )
        open_enhancer = torch.sigmoid(enhancer_access)
        numerator = torch.einsum(
            "gm,mf,bm->bgf",
            self.enhancer_target_prior,
            self.tf_enhancer_prior,
            open_enhancer,
        )
        denominator = (
            self.enhancer_target_prior @ self.tf_enhancer_prior
        ).clamp_min(1e-6)
        return numerator / denominator.unsqueeze(0)

    def enhancer_attribution(
        self, enhancer_access: torch.Tensor
    ) -> torch.Tensor:
        """Element contributions with shape target × enhancer × TF."""

        open_enhancer = torch.sigmoid(enhancer_access)
        score = (
            self.enhancer_target_prior[:, :, None]
            * self.tf_enhancer_prior[None, :, :]
            * open_enhancer[None, :, None]
        )
        return score / score.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _raw_derivative(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        lag_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the vector field.

        ``lag_states[k]`` is the interpolated state at
        ``t - delay_grid[k]``.
        """

        if lag_states.shape != (self.n_delay_bins, self.obs_dim):
            raise ValueError(
                "lag_states must have shape "
                f"({self.n_delay_bins}, {self.obs_dim})"
            )
        access, tf_rna, enhancer, target = self.split_state(state)
        lag_access, lag_tf, _, _ = self.split_state(lag_states)
        access_prob, target_prob = self.delay_probabilities()

        delayed_access = torch.sum(
            access_prob * lag_access.transpose(0, 1), dim=-1
        )
        delayed_tf = torch.einsum(
            "gfk,kf->gf", target_prob, lag_tf
        )

        basis = self.time_basis(t)
        decay_access = F.softplus(self.raw_decay_access)
        decay_tf = F.softplus(self.raw_decay_tf)
        decay_enhancer = F.softplus(self.raw_decay_enhancer)
        decay_target = F.softplus(self.raw_decay_target)

        da = self.access_forcing @ basis - decay_access * access
        dr = (
            self.access_gain * delayed_access
            + self.tf_forcing @ basis
            - decay_tf * tf_rna
        )
        motif_tf_drive = self.tf_enhancer_prior @ tf_rna
        motif_norm = self.tf_enhancer_prior.sum(dim=1).clamp_min(1.0)
        de = (
            self.enhancer_forcing @ basis
            + self.enhancer_tf_gain * motif_tf_drive / motif_norm
            - decay_enhancer * enhancer
        )

        edges = self.edge_weights(t)
        gates = self.enhancer_gate(enhancer)
        regulatory_drive = torch.sum(edges * gates * delayed_tf, dim=1)
        # ``target_bias`` is the intercept.  Excluding the constant forcing
        # column prevents an exact duplicate intercept and reduces a nuisance
        # shortcut that can otherwise absorb regulatory signal.
        target_background = (
            self.target_forcing[:, 1:] @ basis[1:]
            if basis.numel() > 1
            else torch.zeros_like(self.target_bias)
        )
        dx = (
            self.target_bias
            + target_background
            + regulatory_drive
            - decay_target * target
        )

        return {
            "dy": torch.cat([da, dr, de, dx]),
            "edges": edges,
            "enhancer_gates": gates,
        }

    def derivative(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        lag_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        result = self._raw_derivative(t, state, lag_states)
        if not self.has_reference:
            return result
        reference = self.reference_state(t)
        reference_lags = self.reference_state(t - self.delay_grid)
        raw_reference = self._raw_derivative(
            t, reference, reference_lags
        )
        return {
            **result,
            "dy": (
                result["dy"]
                - raw_reference["dy"]
                + self.reference_derivative(t)
            ),
        }

    def _raw_derivative_batch(
        self,
        times: torch.Tensor,
        states: torch.Tensor,
        lag_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized teacher-forced vector-field evaluation."""

        if states.ndim != 2 or lag_states.shape != (
            len(states),
            self.n_delay_bins,
            self.obs_dim,
        ):
            raise ValueError("invalid batched state or lag-bank shape")
        access, tf_rna, enhancer, target = self.split_state(states)
        lag_access, lag_tf, _, _ = self.split_state(lag_states)
        access_prob, target_prob = self.delay_probabilities()
        delayed_access = torch.einsum("fk,bkf->bf", access_prob, lag_access)
        delayed_tf = torch.einsum("gfk,bkf->bgf", target_prob, lag_tf)
        basis = self.time_basis_batch(times)

        decay_access = F.softplus(self.raw_decay_access)
        decay_tf = F.softplus(self.raw_decay_tf)
        decay_enhancer = F.softplus(self.raw_decay_enhancer)
        decay_target = F.softplus(self.raw_decay_target)
        da = basis @ self.access_forcing.T - access * decay_access
        dr = (
            delayed_access * self.access_gain
            + basis @ self.tf_forcing.T
            - tf_rna * decay_tf
        )
        motif_tf_drive = tf_rna @ self.tf_enhancer_prior.T
        motif_norm = self.tf_enhancer_prior.sum(dim=1).clamp_min(1.0)
        de = (
            basis @ self.enhancer_forcing.T
            + motif_tf_drive
            * (self.enhancer_tf_gain / motif_norm).unsqueeze(0)
            - enhancer * decay_enhancer
        )
        edges = self.edge_weights_batch(times)
        gates = self.enhancer_gate_batch(enhancer)
        regulatory_drive = torch.sum(edges * gates * delayed_tf, dim=-1)
        target_background = (
            basis[:, 1:] @ self.target_forcing[:, 1:].T
            if basis.shape[1] > 1
            else torch.zeros(
                len(states),
                self.n_targets,
                dtype=states.dtype,
                device=states.device,
            )
        )
        dx = (
            self.target_bias.unsqueeze(0)
            + target_background
            + regulatory_drive
            - target * decay_target
        )
        return {
            "dy": torch.cat([da, dr, de, dx], dim=-1),
            "edges": edges,
            "enhancer_gates": gates,
        }

    def derivative_batch(
        self,
        times: torch.Tensor,
        states: torch.Tensor,
        lag_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        result = self._raw_derivative_batch(times, states, lag_states)
        if not self.has_reference:
            return result
        reference = self.reference_state(times)
        lag_queries = times[:, None] - self.delay_grid[None, :]
        reference_lags = self.reference_state(lag_queries)
        raw_reference = self._raw_derivative_batch(
            times, reference, reference_lags
        )
        return {
            **result,
            "dy": (
                result["dy"]
                - raw_reference["dy"]
                + self.reference_derivative(times)
            ),
        }

    @staticmethod
    def _interpolate(
        query_t: float,
        history_times: list[float],
        history_states: list[torch.Tensor],
    ) -> torch.Tensor:
        if query_t <= history_times[0]:
            return history_states[0]
        if query_t >= history_times[-1]:
            return history_states[-1]
        # Binary search keeps delayed integration O(log N) per query rather
        # than scanning the entire trajectory history at every solver step.
        right = bisect_right(history_times, query_t)
        right = min(max(right, 1), len(history_times) - 1)
        left = right - 1
        t0, t1 = history_times[left], history_times[right]
        weight = (query_t - t0) / max(t1 - t0, 1e-8)
        return (
            (1.0 - weight) * history_states[left]
            + weight * history_states[right]
        )

    def _lag_bank(
        self,
        t: float,
        history_times: list[float],
        history_states: list[torch.Tensor],
    ) -> torch.Tensor:
        return torch.stack(
            [
                self._interpolate(
                    t - float(delay), history_times, history_states
                )
                for delay in self.delay_grid
            ]
        )

    def _apply_intervention(
        self,
        state: torch.Tensor,
        t: float,
        intervention: Optional[Dict[str, float | int]],
    ) -> torch.Tensor:
        if intervention is None or t < float(intervention.get("start_time", 0.0)):
            return state
        index = int(intervention["tf_index"])
        scale = float(intervention.get("scale", 0.0))
        out = state.clone()
        access_value = intervention.get("access_value")
        rna_value = intervention.get("rna_value")
        if access_value is None:
            out[index] = out[index] * scale
        else:
            out[index] = torch.as_tensor(
                access_value, dtype=out.dtype, device=out.device
            )
        if rna_value is None:
            out[self.n_tfs + index] = out[self.n_tfs + index] * scale
        else:
            out[self.n_tfs + index] = torch.as_tensor(
                rna_value, dtype=out.dtype, device=out.device
            )
        return out

    def integrate(
        self,
        times: torch.Tensor,
        obs0: torch.Tensor,
        max_step: float = 0.01,
        intervention: Optional[Dict[str, float | int]] = None,
        method: str = "heun",
    ) -> EnhancerIntegrationResult:
        """Integrate the delayed state-space model.

        ``method='heun'`` uses a second-order predictor-corrector step which is
        more accurate for delayed counterfactual rollouts than forward Euler.
        """

        if times.ndim != 1 or obs0.shape != (self.obs_dim,):
            raise ValueError("Invalid times or initial observation shape")
        if not torch.all(times[1:] >= times[:-1]):
            raise ValueError("times must be non-decreasing")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")

        start = float(times[0])
        initial = self._apply_intervention(obs0, start, intervention)
        history_times = [start]
        history_states = [initial]
        observations = [initial]
        edges = [self.edge_weights(times[0])]
        gates = [self.enhancer_gate(initial[self.enhancer_slice])]

        for j in range(len(times) - 1):
            current = history_states[-1]
            current_t = history_times[-1]
            end_t = float(times[j + 1])
            n_steps = max(1, int(math.ceil((end_t - current_t) / max_step)))
            step = (end_t - current_t) / n_steps
            aux = None
            for _ in range(n_steps):
                lag_bank = self._lag_bank(
                    current_t, history_times, history_states
                )
                t_tensor = torch.as_tensor(
                    current_t, dtype=obs0.dtype, device=obs0.device
                )
                aux = self.derivative(t_tensor, current, lag_bank)
                if method == "euler":
                    proposed = current + step * aux["dy"]
                    next_t = current_t + step
                else:
                    # Heun / improved Euler: average slope at t and t+h.
                    predictor = current + step * aux["dy"]
                    next_t = current_t + step
                    predictor = self._apply_intervention(
                        predictor, next_t, intervention
                    )
                    # Temporary history tip for lag interpolation at t+h.
                    history_times.append(next_t)
                    history_states.append(predictor)
                    lag_bank_next = self._lag_bank(
                        next_t, history_times, history_states
                    )
                    history_times.pop()
                    history_states.pop()
                    t_next = torch.as_tensor(
                        next_t, dtype=obs0.dtype, device=obs0.device
                    )
                    aux_next = self.derivative(
                        t_next, predictor, lag_bank_next
                    )
                    proposed = current + 0.5 * step * (
                        aux["dy"] + aux_next["dy"]
                    )
                    aux = {
                        "dy": aux_next["dy"],
                        "edges": aux_next["edges"],
                        "enhancer_gates": aux_next["enhancer_gates"],
                    }
                current = self._apply_intervention(
                    proposed, next_t, intervention
                )
                current_t = next_t
                history_times.append(current_t)
                history_states.append(current)
            if aux is None:
                raise RuntimeError("integration did not take a step")
            observations.append(current)
            edges.append(aux["edges"])
            gates.append(aux["enhancer_gates"])

        delays = self.get_delays()
        return EnhancerIntegrationResult(
            obs=torch.stack(observations),
            edges=torch.stack(edges),
            enhancer_gates=torch.stack(gates),
            access_to_tf_delays=delays["access_to_tf"],
            tf_to_target_delays=delays["tf_to_target"],
        )
