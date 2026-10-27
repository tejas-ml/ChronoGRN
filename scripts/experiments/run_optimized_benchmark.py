#!/usr/bin/env python3
"""Run the optimized, leakage-safe ChronoGRN positive-control benchmark.

The validated default is a training-only structured profile/ridge estimator with
one Fourier frequency and continuous delays.  Optional neural fine-tuning is
available, but is disabled by default because it reduced mechanistic
identifiability on the bundled short-trajectory benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bench.delayed_multiome import build_truth, simulate_dataset, split_train_heldout
from causal_grn.enhancer_model import EnhancerDelayGRN
from causal_grn.optimized_fit import OptimizedFitConfig, fit_optimized_chronogrn
from scripts.experiments.scoring import score_model

SUMMARY_METRICS = [
    "edge_auprc",
    "edge_auprc_prior_baseline",
    "dynamic_component_r",
    "access_delay_r",
    "access_delay_mae",
    "target_delay_r",
    "target_delay_mae",
    "control_target_r2",
    "late_horizon_reconstruction_r2",
    "heldout_perturbation_r",
    "heldout_response_trajectory_r",
    "heldout_target_rollout_r2",
    "heldout_endpoint_nrmse",
    "heldout_sign_accuracy_effectful_targets",
    "n_effectful_target_responses",
    "heldout_macro_response_r_effectful",
    "heldout_macro_trajectory_r_effectful",
    "stage_early_response_r_weighted",
    "stage_mid_response_r_weighted",
    "stage_late_response_r_weighted",
    "prior_conditioned_path_top1",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def run_seed(
    seed: int,
    output: Path,
    config: OptimizedFitConfig,
    *,
    n_time_freqs: int,
    delay_parameterization: str,
    device: torch.device,
    integration_step: float,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Fit and score one independent synthetic network seed."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    truth = build_truth(seed=seed)
    cache_path = ROOT / "results" / "cache" / f"seed{seed}.pt"
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        data = cached["data"]
    else:
        data = simulate_dataset(truth, seed=seed + 100)
    train, heldout = split_train_heldout(data)

    model = EnhancerDelayGRN(
        n_tfs=truth.model.n_tfs,
        n_enhancers=truth.model.n_enhancers,
        n_targets=truth.model.n_targets,
        tf_enhancer_prior=torch.tensor(truth.tf_enhancer_prior),
        enhancer_target_prior=torch.tensor(truth.enhancer_target_prior),
        max_tau=0.30,
        n_delay_bins=12,
        n_time_freqs=n_time_freqs,
        dynamic_edges=True,
        enhancer_gating=True,
        delay_parameterization=delay_parameterization,
    )
    per_seed_config = replace(config, seed=seed)
    fit = fit_optimized_chronogrn(
        model,
        train,
        config=per_seed_config,
        device=device,
    )
    metrics = score_model(
        fit.model,
        truth,
        data,
        heldout,
        use_reference=True,
        integration_step=integration_step,
    )
    metrics["seed"] = seed
    metrics["n_training_trajectories"] = len(train)
    metrics["n_heldout_interventions"] = len(heldout)
    metrics["finetune_epochs_trained"] = len(fit.history)

    payload = fit.checkpoint_payload()
    payload["metrics"] = _jsonable(metrics)
    payload["metadata"] = {
        "seed": seed,
        "benchmark": "delayed_multiome_positive_control",
        "training_conditions": [record["name"] for record in train],
        "heldout_conditions": [data[index]["name"] for index in heldout],
        "heldout_data_used_for_fitting": False,
        "integration_step": integration_step,
    }
    checkpoint_path = output / f"seed{seed}.pt"
    torch.save(payload, checkpoint_path)

    compact = {metric: float(metrics[metric]) for metric in SUMMARY_METRICS}
    compact["seed"] = seed
    print(
        f"seed={seed:02d} "
        f"AUPRC={compact['edge_auprc']:.3f} "
        f"dynamic_r={compact['dynamic_component_r']:.3f} "
        f"access_delay_r={compact['access_delay_r']:.3f} "
        f"target_delay_r={compact['target_delay_r']:.3f} "
        f"endpoint_r={compact['heldout_perturbation_r']:.3f} "
        f"trajectory_r={compact['heldout_response_trajectory_r']:.3f} "
        f"rollout_R2={compact['heldout_target_rollout_r2']:.3f}",
        flush=True,
    )
    return {
        "seed": seed,
        "checkpoint": (
            str(checkpoint_path.relative_to(ROOT))
            if checkpoint_path.is_relative_to(ROOT)
            else str(checkpoint_path)
        ),
        "metrics": metrics,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [run["metrics"][metric] for run in runs], dtype=np.float64
        )
        summary[metric] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def write_csv(runs: list[dict[str, Any]], output: Path) -> None:
    with (output / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *SUMMARY_METRICS])
        writer.writeheader()
        for run in runs:
            row = {"seed": run["seed"]}
            row.update(
                {metric: run["metrics"][metric] for metric in SUMMARY_METRICS}
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--out", default="results/benchmark")
    parser.add_argument(
        "--cohort",
        choices=("auto", "development", "validation", "custom"),
        default="auto",
        help="label the evaluated seed set in the saved design metadata",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--time-frequencies", type=int, default=1)
    parser.add_argument("--target-delay-candidates", type=int, default=31)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--delay-response-weight", type=float, default=0.0)
    parser.add_argument("--integration-step", type=float, default=0.008)
    parser.add_argument(
        "--delay-parameterization",
        choices=("continuous", "softmax"),
        default="continuous",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="resimulate data even when results/cache/seedN.pt exists",
    )
    parser.add_argument(
        "--unfreeze-delays",
        action="store_true",
        help="allow delay parameters to move during optional fine-tuning",
    )
    args = parser.parse_args()

    if args.time_frequencies < 0:
        parser.error("--time-frequencies must be non-negative")
    if args.integration_step <= 0:
        parser.error("--integration-step must be positive")

    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(max(1, args.threads))
    except RuntimeError:
        pass

    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    config = OptimizedFitConfig(
        finetune_epochs=args.finetune_epochs,
        smooth_window=args.smooth_window,
        target_delay_candidates=args.target_delay_candidates,
        delay_response_weight=args.delay_response_weight,
        freeze_access_delays=not args.unfreeze_delays,
        freeze_target_delays=not args.unfreeze_delays,
    )
    device = _resolve_device(args.device)
    runs = [
        run_seed(
            seed,
            output,
            config,
            n_time_freqs=args.time_frequencies,
            delay_parameterization=args.delay_parameterization,
            device=device,
            integration_step=args.integration_step,
            use_cache=not args.ignore_cache,
        )
        for seed in args.seeds
    ]
    cohort = args.cohort
    if cohort == "auto":
        if args.seeds and max(args.seeds) <= 9:
            cohort = "development"
        elif args.seeds and min(args.seeds) >= 10:
            cohort = "validation"
        else:
            cohort = "custom"
    payload = {
        "design": {
            "training": "control + early/mid KD of TF0-TF3 + control replicates",
            "heldout": "early/mid/late KD of TF4-TF5",
            "independent_unit": "simulated network seed",
            "n_seeds": len(runs),
            "cohort": cohort,
            "evaluated_seeds": list(args.seeds),
            "development_seeds": list(args.seeds) if cohort == "development" else [],
            "validation_seeds": list(args.seeds) if cohort == "validation" else [],
            "dynamic_basis_frequencies": args.time_frequencies,
            "delay_parameterization": args.delay_parameterization,
            "delay_initialization": "training-only profile regression",
            "coefficient_initialization": "training-only ridge regression",
            "heldout_data_used_for_fitting": False,
            "scope": "synthetic positive-control identifiability",
        },
        "fit_config": asdict(config),
        "summary": summarize(runs),
        "runs": _jsonable(runs),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    write_csv(runs, output)
    print(f"wrote {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
