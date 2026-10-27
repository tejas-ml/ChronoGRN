#!/usr/bin/env python3
"""Score a fitted ChronoGRN against synthetic delayed-multiome ground truth.

Used by run_optimized_benchmark.py and run_optimized_ablation_benchmark.py
to compute the metrics in results/benchmark/ and results/ablation/.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def correlation(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pearsonr(a, b).statistic)


def spearman_corr(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a, b).statistic)


def sign_accuracy(true, pred) -> float:
    true = np.asarray(true).ravel()
    pred = np.asarray(pred).ravel()
    mask = np.abs(true) > 1e-4
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(true[mask]) == np.sign(pred[mask])))


def r2_score(true, pred) -> float:
    true = np.asarray(true)
    pred = np.asarray(pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0)) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _transfer_non_delay_weights(source, target) -> None:
    with torch.no_grad():
        for name in [
            "edge_base",
            "edge_time_sin",
            "edge_time_cos",
            "raw_decay_access",
            "raw_decay_tf",
            "raw_decay_enhancer",
            "raw_decay_target",
            "access_gain",
            "enhancer_tf_gain",
            "target_bias",
            "access_forcing",
            "tf_forcing",
            "enhancer_forcing",
            "target_forcing",
        ]:
            if hasattr(source, name) and hasattr(target, name):
                src = getattr(source, name)
                dst = getattr(target, name)
                if src.shape == dst.shape:
                    dst.copy_(src.detach())


def score_model(
    model,
    truth,
    data,
    heldout_indices,
    use_reference: bool = True,
    integration_step: float = 0.008,
):
    """Score a fitted ChronoGRN against synthetic truth (v0.2 metrics)."""

    device = next(model.parameters()).device
    edge_times = np.linspace(0, 1, 41)
    pred_edge_series = np.stack(
        [
            model.edge_weights(torch.tensor(t, device=device)).detach().cpu().numpy()
            for t in edge_times
        ]
    )
    true_edge_series = np.stack(
        [
            truth.model.edge_weights(torch.tensor(t)).detach().cpu().numpy()
            for t in edge_times
        ]
    )
    score = np.mean(np.abs(pred_edge_series), axis=0)
    active = truth.active_edges.astype(int)
    possible = truth.model.pathway_prior.detach().cpu().numpy() > 0
    y_true = active[possible]
    y_score = score[possible]
    delays = model.get_delays()
    access_delay = delays["access_to_tf"].detach().cpu().numpy()
    target_delay = delays["tf_to_target"].detach().cpu().numpy()
    active_path = truth.active_edges & possible

    access_true = truth.access_to_tf_delays
    target_true = truth.tf_to_target_delays[active_path]
    target_pred = target_delay[active_path]
    access_err = access_delay - access_true
    target_err = target_pred - target_true
    grid_step = float(model.max_tau / max(1, model.n_delay_bins - 1))

    times = torch.tensor(truth.times, device=device)
    control_true = data[0]["obs"]
    control_obs = torch.tensor(control_true, device=device)
    initial = torch.tensor(data[0]["initial_state"], device=device)
    target_slice = model.target_slice

    model.clear_reference()
    free_control = (
        model.integrate(
            times, initial, max_step=integration_step, method="heun"
        )
        .obs.detach()
        .cpu()
        .numpy()
    )
    control_r2 = r2_score(
        control_true[:, target_slice], free_control[:, target_slice]
    )
    mid = len(truth.times) // 2
    late_r2 = r2_score(
        control_true[mid:, target_slice], free_control[mid:, target_slice]
    )

    if use_reference:
        model.set_reference_trajectory(times, control_obs)
    else:
        model.clear_reference()

    control_pred = (
        model.integrate(
            times, initial, max_step=integration_step, method="heun"
        )
        .obs.detach()
        .cpu()
        .numpy()
    )

    true_responses = []
    pred_responses = []
    true_trajs = []
    pred_trajs = []
    rollout_true = []
    rollout_pred = []
    tests = []
    stage_rs = {"early": [], "mid": [], "late": []}
    stage_w = {"early": [], "mid": [], "late": []}
    all_true_delta = []
    all_pred_delta = []
    effectful_true = []
    effectful_pred = []
    effectful_traj_t = []
    effectful_traj_p = []
    effectful_sign = []
    pre_noise_levels = []

    for index in heldout_indices:
        record = data[index]
        intervention = record["intervention"] or {}
        start_time = float(intervention.get("start_time", 0.0))
        initial_test = torch.tensor(record["initial_state"], device=device)
        pred = (
            model.integrate(
                times,
                initial_test,
                max_step=integration_step,
                intervention=intervention if intervention else None,
                method="heun",
            )
            .obs.detach()
            .cpu()
            .numpy()
        )
        true = record["obs"]
        true_delta = (
            true[-20:, target_slice].mean(axis=0)
            - control_true[-20:, target_slice].mean(axis=0)
        )
        pred_delta = (
            pred[-20:, target_slice].mean(axis=0)
            - control_pred[-20:, target_slice].mean(axis=0)
        )
        t_traj = true[:, target_slice] - control_true[:, target_slice]
        p_traj = pred[:, target_slice] - control_pred[:, target_slice]
        # Pre-intervention residual as noise estimate.
        pre_mask = truth.times < start_time
        if pre_mask.sum() < 3:
            pre_mask = np.arange(len(truth.times)) < max(3, len(truth.times) // 5)
        pre_noise = float(np.sqrt(np.mean(t_traj[pre_mask] ** 2)))
        pre_noise = max(pre_noise, 1e-6)
        true_rms = float(np.sqrt(np.mean(true_delta ** 2)))
        pred_rms = float(np.sqrt(np.mean(pred_delta ** 2)))
        effect_snr = true_rms / pre_noise
        rmse = float(np.sqrt(np.mean((true_delta - pred_delta) ** 2)))
        nrmse = rmse / max(true_rms, 1e-6)
        name = str(record["name"])
        stage = "early"
        if name.endswith("_mid"):
            stage = "mid"
        elif name.endswith("_late"):
            stage = "late"
        resp_r = correlation(true_delta, pred_delta)
        traj_r = correlation(t_traj, p_traj)
        # Intervention-level effectful if response exceeds pre-noise.
        effectful_iv = bool(effect_snr >= 1.0)
        entry = {
            "condition": name,
            "stage": stage,
            "response_r": resp_r,
            "trajectory_r": traj_r,
            "sign_accuracy": sign_accuracy(true_delta, pred_delta),
            "rmse": rmse,
            "normalized_rmse": nrmse,
            "true_response_rms": true_rms,
            "pred_response_rms": pred_rms,
            "pre_intervention_noise_rms": pre_noise,
            "effect_snr": effect_snr,
            "effectful": effectful_iv,
            "true_delta": true_delta.tolist(),
            "pred_delta": pred_delta.tolist(),
        }
        tests.append(entry)
        true_responses.append(true_delta)
        pred_responses.append(pred_delta)
        true_trajs.append(t_traj)
        pred_trajs.append(p_traj)
        rollout_true.append(true[:, target_slice])
        rollout_pred.append(pred[:, target_slice])
        stage_rs[stage].append(resp_r)
        stage_w[stage].append(true_rms)
        all_true_delta.append(true_delta)
        all_pred_delta.append(pred_delta)
        pre_noise_levels.append(pre_noise)
        # Target-level effectful: |delta| > noise floor
        thr_g = pre_noise
        eff_mask = np.abs(true_delta) > thr_g
        if eff_mask.any():
            effectful_true.append(true_delta[eff_mask])
            effectful_pred.append(pred_delta[eff_mask])
            effectful_traj_t.append(t_traj[:, eff_mask])
            effectful_traj_p.append(p_traj[:, eff_mask])
            effectful_sign.append(
                sign_accuracy(true_delta[eff_mask], pred_delta[eff_mask])
            )

    model.clear_reference()

    true_cat = np.concatenate(true_responses)
    pred_cat = np.concatenate(pred_responses)
    traj_true = np.concatenate(true_trajs)
    traj_pred = np.concatenate(pred_trajs)
    roll_true = np.concatenate(rollout_true)
    roll_pred = np.concatenate(rollout_pred)
    noise_floor = float(np.median(pre_noise_levels)) if pre_noise_levels else 1e-3

    def _weighted_stage(stage: str) -> float:
        rs = np.asarray(stage_rs[stage], dtype=float)
        ws = np.asarray(stage_w[stage], dtype=float)
        if len(rs) == 0:
            return 0.0
        if ws.sum() <= 0:
            return float(rs.mean())
        return float(np.average(rs, weights=ws))

    if effectful_true:
        eff_t = np.concatenate(effectful_true)
        eff_p = np.concatenate(effectful_pred)
        eff_traj_t = np.concatenate(effectful_traj_t, axis=1)
        eff_traj_p = np.concatenate(effectful_traj_p, axis=1)
        # macro: average per-intervention effectful target correlations
        macro_r = []
        macro_tr = []
        for test, t_delta, p_delta, t_tr, p_tr in zip(
            tests, true_responses, pred_responses, true_trajs, pred_trajs
        ):
            mask = np.abs(t_delta) > float(test["pre_intervention_noise_rms"])
            if mask.sum() < 2:
                continue
            macro_r.append(correlation(t_delta[mask], p_delta[mask]))
            macro_tr.append(correlation(t_tr[:, mask], p_tr[:, mask]))
        heldout_macro_r = float(np.mean(macro_r)) if macro_r else 0.0
        heldout_macro_tr = float(np.mean(macro_tr)) if macro_tr else 0.0
        n_eff_targets = int(eff_t.size)
        sign_eff = float(np.mean(effectful_sign)) if effectful_sign else 0.0
    else:
        heldout_macro_r = 0.0
        heldout_macro_tr = 0.0
        n_eff_targets = 0
        sign_eff = 0.0

    # Strong-sign accuracy: larger effects only
    strong_mask = np.abs(true_cat) > max(noise_floor * 2.0, 1e-4)
    sign_strong = (
        sign_accuracy(true_cat[strong_mask], pred_cat[strong_mask])
        if strong_mask.any()
        else 0.0
    )

    path_hits = []
    with torch.no_grad():
        for target, tf in zip(*np.where(truth.active_edges)):
            candidates = np.where(
                (truth.enhancer_target_prior[target] > 0)
                & (truth.tf_enhancer_prior[:, tf] > 0)
            )[0]
            if len(candidates) == 0:
                continue
            true_enh = int(
                candidates[
                    np.argmax(
                        truth.enhancer_target_prior[target, candidates]
                        * truth.tf_enhancer_prior[candidates, tf]
                    )
                ]
            )
            enh_state = free_control[:, model.enhancer_slice][:, candidates]
            open_mean = 1.0 / (1.0 + np.exp(-enh_state.mean(axis=0)))
            prior = (
                truth.enhancer_target_prior[target, candidates]
                * truth.tf_enhancer_prior[candidates, tf]
            )
            ranking = open_mean * prior
            top = int(candidates[int(np.argmax(ranking))])
            path_hits.append(1.0 if top == true_enh else 0.0)
    path_top1 = float(np.mean(path_hits)) if path_hits else 0.0
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    endpoint_rmse = float(np.sqrt(np.mean((true_cat - pred_cat) ** 2)))
    true_cat_rms = float(np.sqrt(np.mean(true_cat ** 2)))

    return {
        "edge_auroc": float(roc_auc_score(y_true, y_score)) if y_true.sum() else 0.0,
        "edge_auprc": float(average_precision_score(y_true, y_score))
        if y_true.sum()
        else 0.0,
        "edge_auprc_prior_baseline": prevalence,
        "evaluation_universe": "pathway_prior_supported_pairs",
        "dynamic_edge_r": correlation(
            true_edge_series[:, active_path], pred_edge_series[:, active_path]
        ),
        "dynamic_component_r": correlation(
            (true_edge_series - true_edge_series.mean(0))[:, active_path],
            (pred_edge_series - pred_edge_series.mean(0))[:, active_path],
        ),
        "access_delay_mae": float(np.mean(np.abs(access_err))),
        "access_delay_median_ae": float(np.median(np.abs(access_err))),
        "access_delay_r": correlation(access_delay, access_true),
        "access_delay_spearman": spearman_corr(access_delay, access_true),
        "access_delay_bias": float(np.mean(access_err)),
        "access_delay_within_bin": float(np.mean(np.abs(access_err) <= grid_step)),
        "target_delay_mae": float(np.mean(np.abs(target_err))),
        "target_delay_median_ae": float(np.median(np.abs(target_err))),
        "target_delay_r": correlation(target_pred, target_true),
        "target_delay_spearman": spearman_corr(target_pred, target_true),
        "target_delay_bias": float(np.mean(target_err)),
        "target_delay_within_bin": float(np.mean(np.abs(target_err) <= grid_step)),
        "target_delay_frac_trajectory": float(
            np.mean(np.abs(target_err)) / max(float(truth.times[-1]), 1e-6)
        ),
        "control_target_r2": control_r2,
        "late_horizon_control_r2": late_r2,
        "late_horizon_reconstruction_r2": late_r2,
        "heldout_perturbation_r": correlation(true_cat, pred_cat),
        "heldout_response_trajectory_r": correlation(traj_true, traj_pred),
        "heldout_target_rollout_r2": r2_score(roll_true, roll_pred),
        "heldout_sign_accuracy": sign_accuracy(true_cat, pred_cat),
        "heldout_sign_accuracy_strong": sign_strong,
        "heldout_sign_accuracy_effectful_targets": sign_eff,
        "heldout_endpoint_rmse": endpoint_rmse,
        "heldout_endpoint_nrmse": float(endpoint_rmse / max(true_cat_rms, 1e-6)),
        "heldout_effectful_target_threshold": noise_floor,
        "n_effectful_target_responses": n_eff_targets,
        "n_effectful_interventions": int(sum(1 for t in tests if t["effectful"])),
        "heldout_macro_response_r_effectful": heldout_macro_r,
        "heldout_macro_trajectory_r_effectful": heldout_macro_tr,
        "stage_early_response_r": float(np.mean(stage_rs["early"]) if stage_rs["early"] else 0.0),
        "stage_mid_response_r": float(np.mean(stage_rs["mid"]) if stage_rs["mid"] else 0.0),
        "stage_late_response_r": float(np.mean(stage_rs["late"]) if stage_rs["late"] else 0.0),
        "stage_early_response_r_weighted": _weighted_stage("early"),
        "stage_mid_response_r_weighted": _weighted_stage("mid"),
        "stage_late_response_r_weighted": _weighted_stage("late"),
        "enhancer_path_top1": path_top1,
        "prior_conditioned_path_top1": path_top1,
        "tests": tests,
        "per_intervention_response_r": [t["response_r"] for t in tests],
        "per_intervention_trajectory_r": [t["trajectory_r"] for t in tests],
        "learned_access_delays": access_delay.tolist(),
        "true_access_delays": access_true.tolist(),
        "learned_target_delays": target_delay.tolist(),
        "true_target_delays": truth.tf_to_target_delays.tolist(),
        "edge_score": score.tolist(),
        "true_active_edges": active.tolist(),
        "edge_times": edge_times.tolist(),
        "pred_edge_series": pred_edge_series.tolist(),
        "true_edge_series": true_edge_series.tolist(),
        "edge_class": np.asarray(truth.edge_class, dtype=str).tolist(),
    }


