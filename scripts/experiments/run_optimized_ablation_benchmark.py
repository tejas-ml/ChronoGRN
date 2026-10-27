#!/usr/bin/env python3
"""Paired optimized ChronoGRN ablations on independent synthetic networks."""
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

VARIANTS = {
    "ChronoGRN": {"delays": True, "dynamic_edges": True, "enhancer_gating": True},
    "No delay": {"delays": False, "dynamic_edges": True, "enhancer_gating": True},
    "No gate": {"delays": True, "dynamic_edges": True, "enhancer_gating": False},
    "Fixed W": {"delays": True, "dynamic_edges": False, "enhancer_gating": True},
}

METRICS = [
    "edge_auprc",
    "dynamic_component_r",
    "control_target_r2",
    "late_horizon_reconstruction_r2",
    "heldout_perturbation_r",
    "heldout_response_trajectory_r",
    "heldout_target_rollout_r2",
    "heldout_sign_accuracy_effectful_targets",
    "stage_early_response_r_weighted",
    "stage_mid_response_r_weighted",
    "stage_late_response_r_weighted",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        result[variant] = {}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            result[variant][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    full = {row["seed"]: row for row in rows if row["variant"] == "ChronoGRN"}
    paired: dict[str, Any] = {}
    for variant in VARIANTS:
        if variant == "ChronoGRN":
            continue
        paired[variant] = {}
        comparison = {row["seed"]: row for row in rows if row["variant"] == variant}
        shared = sorted(set(full) & set(comparison))
        for metric in METRICS:
            delta = np.asarray(
                [full[seed][metric] - comparison[seed][metric] for seed in shared],
                dtype=float,
            )
            paired[variant][metric] = {
                "mean_full_minus_ablation": float(delta.mean()),
                "sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "wins": int(np.sum(delta > 0)),
                "ties": int(np.sum(np.isclose(delta, 0.0))),
                "n": len(delta),
            }
    return {"variants": result, "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-frequencies", type=int, default=1)
    parser.add_argument("--target-delay-candidates", type=int, default=31)
    parser.add_argument("--integration-step", type=float, default=0.008)
    parser.add_argument("--ignore-cache", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(max(1, args.threads))
    except RuntimeError:
        pass

    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    base_config = OptimizedFitConfig(
        finetune_epochs=0,
        target_delay_candidates=args.target_delay_candidates,
    )
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        truth = build_truth(seed=seed)
        cache_path = ROOT / "results" / "cache" / f"seed{seed}.pt"
        if cache_path.exists() and not args.ignore_cache:
            data = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )["data"]
        else:
            data = simulate_dataset(truth, seed=seed + 100)
        train, heldout = split_train_heldout(data)

        for variant, settings in VARIANTS.items():
            delays = bool(settings["delays"])
            model = EnhancerDelayGRN(
                n_tfs=truth.model.n_tfs,
                n_enhancers=truth.model.n_enhancers,
                n_targets=truth.model.n_targets,
                tf_enhancer_prior=torch.tensor(truth.tf_enhancer_prior),
                enhancer_target_prior=torch.tensor(truth.enhancer_target_prior),
                max_tau=0.30 if delays else 0.0,
                n_delay_bins=12 if delays else 1,
                n_time_freqs=args.time_frequencies,
                dynamic_edges=bool(settings["dynamic_edges"]),
                enhancer_gating=bool(settings["enhancer_gating"]),
                delay_parameterization="continuous",
            )
            fit = fit_optimized_chronogrn(
                model,
                train,
                config=replace(base_config, seed=seed),
                device=torch.device("cpu"),
            )
            metrics = score_model(
                fit.model,
                truth,
                data,
                heldout,
                use_reference=True,
                integration_step=args.integration_step,
            )
            row = {"seed": seed, "variant": variant}
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            rows.append(row)
            print(
                f"seed={seed:02d} {variant:10s} "
                f"endpoint={row['heldout_perturbation_r']:.3f} "
                f"trajectory={row['heldout_response_trajectory_r']:.3f} "
                f"late={row['stage_late_response_r_weighted']:.3f} "
                f"dynamic={row['dynamic_component_r']:.3f}",
                flush=True,
            )

    with (output / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "variant", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "design": {
            "independent_unit": "simulated network seed",
            "n_seeds": len(args.seeds),
            "paired_variants": list(VARIANTS),
            "heldout_data_used_for_fitting": False,
            "time_frequencies": args.time_frequencies,
        },
        "fit_config": asdict(base_config),
        "summary": summarize(rows),
        "rows": _jsonable(rows),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
