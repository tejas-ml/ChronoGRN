#!/usr/bin/env python3
"""Three capability figures for the synthetic positive-control benchmark, with honest claims.

Figure 1 — delayed multiome dynamics (what the model is + delay recovery)
Figure 2 — held-out stage-specific TF knockdown prediction
Figure 3 — dynamic enhancer-resolved GRN recovery

Claims are matched to what the stage-specific benchmark actually supports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import networkx as nx
import numpy as np
import torch
from scipy.stats import pearsonr, linregress

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bench.delayed_multiome import build_truth, simulate_dataset, split_train_heldout
from causal_grn.enhancer_model import EnhancerDelayGRN
from causal_grn.checkpoint import load_checkpoint


PURPLE = "#4C3D9B"
TEAL = "#0F766E"
ORANGE = "#D45C2C"
BLUE = "#1D6A9A"
ROSE = "#B23A48"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#E8E6E0"
SOFT = "#F7F6F2"
BG = "#FFFFFF"

VARIANT_COLORS = {
    "ChronoGRN": PURPLE,
    "No delays": ORANGE,
    "No enhancer gate": TEAL,
    "Fixed edges": BLUE,
}
FILE_STEMS = {
    "ChronoGRN": "chronogrn",
    "No delays": "no_delays",
    "No enhancer gate": "no_enhancer_gate",
    "Fixed edges": "fixed_edges",
}
SHORT = {
    "ChronoGRN": "ChronoGRN",
    "No delays": "No delay",
    "No enhancer gate": "No gate",
    "Fixed edges": "Fixed W",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.linewidth": 1.0, "axes.edgecolor": INK,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0,
    })


def title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", color=INK, pad=7, fontsize=10.5)


def clean(ax):
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.2, width=0.9, color=MUTED)


def save(fig, outdir, stem):
    fig.savefig(outdir / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  wrote {stem}")


def load_model(result_dir, seed, name, truth):
    """Load either a v0.2 self-contained checkpoint or a legacy checkpoint."""

    optimized = result_dir / f"seed{seed}.pt"
    if name == "ChronoGRN" and optimized.exists():
        model, _ = load_checkpoint(optimized, map_location="cpu", strict=True)
        model.eval()
        return model

    ckpt = torch.load(
        result_dir / f"seed{seed}_{FILE_STEMS[name]}.pt",
        map_location="cpu", weights_only=False,
    )
    model = EnhancerDelayGRN(
        tf_enhancer_prior=torch.tensor(truth.tf_enhancer_prior),
        enhancer_target_prior=torch.tensor(truth.enhancer_target_prior),
        **ckpt["model_kwargs"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _optimized_to_enriched(optimized, ablation):
    """Adapt optimized benchmark JSON to the legacy figure-data interface."""

    by_seed = {}
    if ablation is not None:
        for row in ablation.get("rows", []):
            by_seed.setdefault(int(row["seed"]), {})[row["variant"]] = row

    variant_map = {
        "No delay": "No delays",
        "No gate": "No enhancer gate",
        "Fixed W": "Fixed edges",
    }
    runs = []
    for run in optimized["runs"]:
        seed = int(run["seed"])
        variants = {"ChronoGRN": run["metrics"]}
        for raw_name, row in by_seed.get(seed, {}).items():
            if raw_name == "ChronoGRN":
                continue
            variants[variant_map.get(raw_name, raw_name)] = row
        runs.append({"seed": seed, "variants": variants})
    return {
        "design": optimized.get("design", {}),
        "runs": runs,
    }


def load_enriched(result_dir, ablation_path=None):
    for name in ("results_enriched.json", "results.json"):
        path = result_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload.get("runs") and "metrics" in payload["runs"][0]:
            ablation = None
            if ablation_path is not None and Path(ablation_path).exists():
                ablation = json.loads(Path(ablation_path).read_text())
            return _optimized_to_enriched(payload, ablation)
        return payload
    raise FileNotFoundError(result_dir)


def metric(enriched, variant, key):
    return np.asarray(
        [run["variants"][variant][key] for run in enriched["runs"]
         if variant in run["variants"] and key in run["variants"][variant]],
        dtype=float,
    )


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def paired_delta_plot(ax, enriched, key, ablations, ylabel):
    clean(ax)
    base = metric(enriched, "ChronoGRN", key)
    x = np.arange(len(ablations))
    for i, abl in enumerate(ablations):
        delta = base - metric(enriched, abl, key)
        jitter = np.linspace(-0.08, 0.08, len(delta))
        ax.axhline(0, color=MUTED, lw=1.0, ls=":")
        ax.scatter(
            i + jitter, delta, s=36, color=VARIANT_COLORS[abl],
            edgecolor="white", lw=0.5, zorder=3,
        )
        ax.hlines(delta.mean(), i - 0.18, i + 0.18, color=INK, lw=2.4, zorder=4)
        # bootstrap CI
        if len(delta) >= 5:
            boots = [
                np.mean(np.random.default_rng(b).choice(delta, size=len(delta), replace=True))
                for b in range(400)
            ]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ax.vlines(i, lo, hi, color=INK, lw=1.2, zorder=2)
    ax.set_xticks(x, [SHORT[a] for a in ablations])
    ax.set_ylabel(ylabel)


def get_bundle(result_dir, seed=0):
    truth = build_truth(seed)
    cache_path = ROOT / "results" / "cache" / f"seed{seed}.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)["data"]
    else:
        data = simulate_dataset(truth, seed=seed + 100)
    model = load_model(result_dir, seed, "ChronoGRN", truth)
    times = torch.tensor(truth.times)
    control_obs = torch.tensor(data[0]["obs"])
    model.set_reference_trajectory(times, control_obs)
    roll = {}
    for rec in data:
        if rec["name"] == "control" or rec.get("role") == "heldout":
            roll[rec["name"]] = {
                "true": rec["obs"],
                "pred": model.integrate(
                    times, torch.tensor(rec["initial_state"]),
                    max_step=0.008, intervention=rec["intervention"], method="heun",
                ).obs.detach().numpy(),
                "intervention": rec["intervention"],
            }
    model.clear_reference()
    free = model.integrate(
        times, torch.tensor(data[0]["initial_state"]), max_step=0.008, method="heun",
    ).obs.detach().numpy()
    return truth, data, model, roll, free


def pick_activating_chain(truth, free, model):
    """Prefer an activating TF→target chain with strong gate swing."""
    best = None
    best_score = -np.inf
    for target, tf in np.argwhere(truth.active_edges):
        enhancers = np.where(
            (truth.enhancer_target_prior[target] > 0)
            & (truth.tf_enhancer_prior[:, tf] > 0)
        )[0]
        if len(enhancers) == 0:
            continue
        # mean signed edge over time from truth
        w_series = np.array([
            truth.model.edge_weights(torch.tensor(float(t))).detach().numpy()[target, tf]
            for t in np.linspace(0, 1, 21)
        ])
        if w_series.mean() <= 0:
            continue  # prefer activation
        enh = int(enhancers[0])
        e = free[:, model.enhancer_slice][:, enh]
        gate = 1 / (1 + np.exp(-e))
        score = float(np.std(w_series) + np.std(gate) + 0.3 * truth.access_to_tf_delays[tf]
                      + 0.3 * truth.tf_to_target_delays[target, tf])
        if score > best_score:
            best_score = score
            best = (int(tf), enh, int(target), float(truth.access_to_tf_delays[tf]),
                    float(truth.tf_to_target_delays[target, tf]), float(w_series.mean()))
    if best is None:
        # fallback any active
        target, tf = np.argwhere(truth.active_edges)[0]
        enh = int(np.where(
            (truth.enhancer_target_prior[target] > 0)
            & (truth.tf_enhancer_prior[:, tf] > 0)
        )[0][0])
        best = (int(tf), enh, int(target), float(truth.access_to_tf_delays[tf]),
                float(truth.tf_to_target_delays[target, tf]), 1.0)
    return best


# ===========================================================================
# FIGURE 1
# ===========================================================================
def figure1(outdir, enriched, result_dir):
    truth, data, model, roll, free = get_bundle(result_dir, 0)
    times = truth.times
    true_ctrl = data[0]["obs"]
    mid = len(times) // 2
    tf, enh, gene, tau_a, tau_t, wmean = pick_activating_chain(truth, free, model)

    access_true, access_pred, target_true, target_pred = [], [], [], []
    for run in enriched["runs"]:
        rec = run["variants"]["ChronoGRN"]
        access_true.extend(rec["true_access_delays"])
        access_pred.extend(rec["learned_access_delays"])
        active = np.asarray(rec["true_active_edges"], dtype=bool)
        target_true.extend(np.asarray(rec["true_target_delays"])[active])
        target_pred.extend(np.asarray(rec["learned_target_delays"])[active])
    access_true, access_pred = np.asarray(access_true), np.asarray(access_pred)
    target_true, target_pred = np.asarray(target_true), np.asarray(target_pred)

    fig = plt.figure(figsize=(11.8, 8.8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1.0],
                  hspace=0.42, wspace=0.42, left=0.06, right=0.98, top=0.90, bottom=0.07)

    # A — model with multiplication gate node
    ax = fig.add_subplot(gs[0, 0:2])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    title(ax, "a  Model overview")
    boxes = [
        (0.02, 0.62, 0.18, 0.24, "TF promoter\naccessibility", BLUE),
        (0.30, 0.62, 0.18, 0.24, "TF RNA /\nactivity proxy", PURPLE),
        (0.78, 0.62, 0.18, 0.24, "Target\ngene RNA", ORANGE),
        (0.30, 0.12, 0.18, 0.24, "Enhancer\naccessibility", TEAL),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
            facecolor=color, edgecolor=color, alpha=0.16, lw=1.7, transform=ax.transAxes,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK, transform=ax.transAxes)
    # delay 1
    ax.annotate("", xy=(0.30, 0.74), xytext=(0.20, 0.74),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=12),
                xycoords=ax.transAxes)
    ax.text(0.25, 0.82, r"$\tau_{\mathrm{acc}\to\mathrm{TF}}$", ha="center",
            fontsize=9, transform=ax.transAxes)
    # multiplication node
    ax.add_patch(Circle((0.60, 0.74), 0.045, facecolor=SOFT, edgecolor=INK,
                         lw=1.5, transform=ax.transAxes))
    ax.text(0.60, 0.74, r"$\times$", ha="center", va="center", fontsize=14,
            fontweight="bold", transform=ax.transAxes)
    ax.annotate("", xy=(0.555, 0.74), xytext=(0.48, 0.74),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=12),
                xycoords=ax.transAxes)
    ax.annotate("", xy=(0.78, 0.74), xytext=(0.645, 0.74),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=12),
                xycoords=ax.transAxes)
    ax.text(0.70, 0.82, r"$\tau_{\mathrm{TF}\to\mathrm{target}}$", ha="center",
            fontsize=9, transform=ax.transAxes)
    # gate into multiply
    ax.annotate("", xy=(0.58, 0.70), xytext=(0.45, 0.36),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6, mutation_scale=11),
                xycoords=ax.transAxes)
    ax.text(0.48, 0.48, "gate", color=TEAL, fontsize=8.5, fontweight="bold",
            transform=ax.transAxes)
    # W(t) into multiply
    ax.text(0.60, 0.92, r"$W_{ij}(t)$", ha="center", color=PURPLE, fontsize=10,
            fontweight="bold", transform=ax.transAxes)
    ax.annotate("", xy=(0.60, 0.785), xytext=(0.60, 0.90),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4, mutation_scale=10),
                xycoords=ax.transAxes)
    ax.text(0.5, 0.02,
            "Regulatory drive = $W_{ij}(t)\\,\\times\\,$gate$\\,\\times\\,$delayed TF RNA",
            ha="center", fontsize=8, color=MUTED, transform=ax.transAxes)

    # B — stacked trajectory with reconstruction note
    series = [
        (true_ctrl[:, model.access_slice][:, tf], free[:, model.access_slice][:, tf],
         f"TF{tf} promoter ATAC", BLUE),
        (true_ctrl[:, model.tf_slice][:, tf], free[:, model.tf_slice][:, tf],
         f"TF{tf} RNA", PURPLE),
        (true_ctrl[:, model.enhancer_slice][:, enh], free[:, model.enhancer_slice][:, enh],
         f"Enhancer E{enh}", TEAL),
        (true_ctrl[:, model.target_slice][:, gene], free[:, model.target_slice][:, gene],
         f"Target G{gene} (activated)", ORANGE),
    ]
    inner = gs[0, 2:4].subgridspec(4, 1, hspace=0.10)
    axes_b = []
    for i, (obs, pred, label, color) in enumerate(series):
        ax = fig.add_subplot(inner[i, 0], sharex=axes_b[0] if axes_b else None)
        axes_b.append(ax)
        clean(ax)
        ax.plot(times, obs, color=color, lw=2.1)
        ax.plot(times, pred, color=color, lw=1.5, ls=(0, (4, 2)))
        # half-horizon visual: left reconstruction, right free-roll quality
        ax.axvspan(0.5, 1.0, color="#F0EEE8", alpha=0.55, zorder=0)
        ax.axvline(0.5, color=MUTED, ls=":", lw=1.0)
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        ax.tick_params(labelbottom=(i == 3), labelsize=7.5)
        if i == 0:
            ax.set_title("b  Example delayed multiome trajectory", loc="left",
                         fontweight="bold", color=INK, pad=7, fontsize=10.5)
            ax.legend(handles=[
                Line2D([0], [0], color=INK, lw=2.1, label="Observed"),
                Line2D([0], [0], color=INK, lw=1.5, ls=(0, (4, 2)), label="ChronoGRN free roll"),
            ], loc="upper right", fontsize=7)
            ax.text(0.26, 0.08, "full free-roll reconstruction", transform=ax.transAxes,
                    fontsize=7, color=MUTED, ha="center")
            ax.text(0.75, 0.08, "late horizon", transform=ax.transAxes,
                    fontsize=7, color=MUTED, ha="center")
        if i == 3:
            ax.set_xlabel("Trajectory time")
            ax.text(0.98, 0.12,
                    f"true delays: τ_acc→TF={tau_a:.2f} · τ_TF→G={tau_t:.2f}",
                    transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    axes_b[0].set_xlim(0, 1)

    # C/D delay recovery with regression
    for col, true, pred, panel, color in [
        (0, access_true, access_pred, "c  Accessibility → TF delays", PURPLE),
        (1, target_true, target_pred, "d  TF → target delays", ORANGE),
    ]:
        ax = fig.add_subplot(gs[1, col])
        clean(ax)
        title(ax, panel)
        ax.scatter(true, pred, s=22, c=color, alpha=0.55, edgecolors="none", zorder=3)
        lim = 0.32
        ax.plot([0, lim], [0, lim], color=INK, ls=(0, (3, 2)), lw=1.1, label="identity")
        if np.std(true) > 1e-8 and np.std(pred) > 1e-8:
            slope, intercept, r, _, _ = linregress(true, pred)
            xs = np.linspace(0, lim, 50)
            ax.plot(xs, slope * xs + intercept, color=color, lw=1.4, alpha=0.8, label="fit")
            mae = np.mean(np.abs(true - pred))
            bias = np.mean(pred - true)
            ax.text(0.05, 0.95,
                    f"MAE={mae:.3f}\n$r$={r:.2f}\nslope={slope:.2f}\nbias={bias:+.3f}",
                    transform=ax.transAxes, va="top", fontsize=7.8,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("True delay"); ax.set_ylabel("Learned delay")
        ax.set_aspect("equal", adjustable="box")
        if col == 0:
            ax.legend(loc="lower right", fontsize=7)

    # E — paired deltas vs no-delay (honest title)
    ax = fig.add_subplot(gs[1, 2:4])
    title(ax, "e  Paired Δ$R^2$ vs ablations (held-out rollout)")
    paired_delta_plot(
        ax, enriched, "heldout_target_rollout_r2",
        ["No delays", "No enhancer gate", "Fixed edges"],
        r"$R^2_{\mathrm{ChronoGRN}}-R^2_{\mathrm{ablation}}$",
    )
    d = (metric(enriched, "ChronoGRN", "heldout_target_rollout_r2")
         - metric(enriched, "No delays", "heldout_target_rollout_r2"))
    ax.text(0.98, 0.08,
            f"vs no delay: mean Δ={d.mean():+.3f}  "
            f"({(d > 0).sum()}/{len(d)} seeds)",
            transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.28", fc=SOFT, ec=GRID))

    n = len(enriched["runs"])
    fig.suptitle(
        "Figure 1 | ChronoGRN models delayed multiome regulation",
        x=0.06, y=0.975, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.945,
        f"Positive-control delayed multiome · free-roll reconstruction · continuous delays · "
        f"n = {n} independent networks",
        color=MUTED, fontsize=9,
    )
    save(fig, outdir, "figure1_delayed_dynamics")


# ===========================================================================
# FIGURE 2
# ===========================================================================
def figure2(outdir, enriched, result_dir):
    truth, data, model, roll, free = get_bundle(result_dir, 0)
    times = truth.times
    ts = model.target_slice

    all_true, all_pred, labels, per_seed_r = [], [], [], []
    for run in enriched["runs"]:
        seed = run["seed"]
        seed_true, seed_pred = [], []
        for test in run["variants"]["ChronoGRN"]["tests"]:
            all_true.append(test["true_delta"])
            all_pred.append(test["pred_delta"])
            labels.append(f"S{seed}·{test['condition'].replace('TF','').replace('_KD_',' ')}")
            seed_true.append(test["true_delta"])
            seed_pred.append(test["pred_delta"])
        st, sp = np.concatenate(seed_true), np.concatenate(seed_pred)
        if np.std(st) > 1e-8 and np.std(sp) > 1e-8:
            per_seed_r.append(pearsonr(st, sp).statistic)
        else:
            per_seed_r.append(0.0)
    true = np.asarray(all_true)
    pred = np.asarray(all_pred)
    pooled_r = pearsonr(true.ravel(), pred.ravel()).statistic
    responsive = np.argsort(-np.mean(np.abs(true), axis=0))[:12]
    order = np.argsort(labels)
    labels_ord = [labels[i] for i in order]
    true_ord, pred_ord = true[order], pred[order]
    vmax = np.percentile(np.abs(true_ord[:, responsive]), 97)

    # Illustrative trajectory selected without looking at model predictions:
    # choose the held-out condition with the largest observed response energy,
    # then display its three most responsive observed targets.
    condition_names = [name for name in roll if name != "control"]
    response_energy = {}
    for name in condition_names:
        observed_delta = (
            roll[name]["true"][:, ts] - roll["control"]["true"][:, ts]
        )
        response_energy[name] = float(np.sqrt(np.mean(observed_delta**2)))
    example_name = max(condition_names, key=response_energy.get)
    true_d = roll[example_name]["true"][:, ts] - roll["control"]["true"][:, ts]
    pred_d = roll[example_name]["pred"][:, ts] - roll["control"]["pred"][:, ts]
    mag = np.max(np.abs(true_d), axis=0)
    genes = [int(g) for g in np.argsort(-mag)[:3]]

    fig = plt.figure(figsize=(12.0, 9.4))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.95, 1.1, 1.0],
                  hspace=0.48, wspace=0.40, left=0.07, right=0.98, top=0.90, bottom=0.06)

    # A design
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    title(ax, "a  Held-out design")
    ax.add_patch(FancyBboxPatch((0.05, 0.55), 0.90, 0.38,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=SOFT, edgecolor=GRID, lw=1.2, transform=ax.transAxes))
    ax.text(0.5, 0.84, "TRAINING", ha="center", fontsize=8.5, fontweight="bold",
            color=MUTED, transform=ax.transAxes)
    ax.text(0.5, 0.68, "control · TF0–3 early/mid KD\ncontrol replicates\npriors + control trajectories used",
            ha="center", va="center", fontsize=7.5, color=INK, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.90, 0.40,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#F8EDE6", edgecolor=ORANGE, lw=1.5, transform=ax.transAxes))
    ax.text(0.5, 0.38, "HELD-OUT CONDITIONS", ha="center", fontsize=8.5,
            fontweight="bold", color=ORANGE, transform=ax.transAxes)
    ax.text(0.5, 0.20, "TF4 & TF5 · early / mid / late KD\nall targets blinded · stage-specific",
            ha="center", va="center", fontsize=7.5, color=INK, transform=ax.transAxes)

    # B scatter + per-seed inset
    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    title(ax, "b  Endpoint responses")
    ax.scatter(true.ravel(), pred.ravel(), s=12, c=PURPLE, alpha=0.45, edgecolors="none")
    lim = np.percentile(np.abs(np.r_[true.ravel(), pred.ravel()]), 99) * 1.12
    ax.plot([-lim, lim], [-lim, lim], color=INK, ls=(0, (3, 2)), lw=1.1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Observed Δexpression"); ax.set_ylabel("Predicted Δexpression")
    ax.set_aspect("equal", adjustable="box")
    per_seed_r = np.asarray(per_seed_r)
    ax.text(0.05, 0.95,
            f"pooled $r$={pooled_r:.3f}\n"
            f"per-network mean $r$={per_seed_r.mean():.3f}±{per_seed_r.std(ddof=1):.3f}\n"
            f"{true.shape[0]} interventions · {len(enriched['runs'])} networks",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID))

    # C/D heatmaps + residual
    err = pred_ord[:, responsive].T - true_ord[:, responsive].T
    for col, matrix, panel, cmap, vlim in [
        (2, true_ord[:, responsive].T, "c  Observed", "RdBu_r", vmax),
        (3, pred_ord[:, responsive].T, "d  Predicted", "RdBu_r", vmax),
    ]:
        ax = fig.add_subplot(gs[0, col])
        title(ax, panel)
        im = ax.imshow(matrix, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(0, len(labels_ord), max(1, len(labels_ord) // 6)))
        ax.set_xticklabels(
            [labels_ord[i] for i in range(0, len(labels_ord), max(1, len(labels_ord) // 6))],
            rotation=60, ha="right", fontsize=6,
        )
        if col == 2:
            ax.set_yticks(range(len(responsive)))
            ax.set_yticklabels([f"G{g}" for g in responsive], fontsize=7)
            ax.set_ylabel("Most responsive targets*")
        else:
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Δexpr", fontsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.6); spine.set_color(GRID)

    # E trajectory
    ax = fig.add_subplot(gs[1, :3])
    clean(ax)
    title(ax, f"e  Illustrative high-SNR trajectory ({example_name})")
    colors = [PURPLE, ORANGE, TEAL]
    start = 0.72
    if roll[example_name]["intervention"]:
        start = float(roll[example_name]["intervention"].get("start_time", 0.72))
    for i, g in enumerate(genes):
        c = colors[i]
        tr = pearsonr(true_d[:, g], pred_d[:, g]).statistic if np.std(true_d[:, g]) > 1e-8 else 0
        ax.plot(times, true_d[:, g], color=c, lw=2.2, label=f"G{g} obs")
        ax.plot(times, pred_d[:, g], color=c, lw=1.6, ls=(0, (4, 2)),
                label=f"G{g} pred ($r$={tr:.2f})")
    ax.axvline(start, color=MUTED, ls=":", lw=1.2)
    ax.text(start + 0.01, 0.95, "KD start", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("Trajectory time"); ax.set_ylabel("Δexpression vs control")
    ax.legend(ncol=3, loc="best", fontsize=7.5)

    # residual heatmap small
    ax = fig.add_subplot(gs[1, 3])
    title(ax, "f  Residual (pred−obs)")
    emax = np.percentile(np.abs(err), 97)
    ax.imshow(err, cmap="RdBu_r", vmin=-emax, vmax=emax, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks(range(len(responsive)))
    ax.set_yticklabels([f"G{g}" for g in responsive], fontsize=6.5)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.6); spine.set_color(GRID)

    # G/H paired deltas for stage-aware metrics
    ax = fig.add_subplot(gs[2, :2])
    title(ax, "g  Paired Δ endpoint $r$ (Chrono − ablation)")
    paired_delta_plot(
        ax, enriched, "heldout_perturbation_r",
        ["No delays", "No enhancer gate", "Fixed edges"],
        r"$r_{\mathrm{Chrono}}-r_{\mathrm{ablation}}$",
    )
    d = (metric(enriched, "ChronoGRN", "heldout_perturbation_r")
         - metric(enriched, "Fixed edges", "heldout_perturbation_r"))
    ax.text(0.02, 0.95, f"vs Fixed W: mean Δ={d.mean():+.3f} ({(d>0).sum()}/{len(d)} wins)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc=SOFT, ec=GRID))

    ax = fig.add_subplot(gs[2, 2:])
    title(ax, "h  Late-stage held-out response $r$ by model")
    clean(ax)
    variants = ["ChronoGRN", "No delays", "No enhancer gate", "Fixed edges"]
    vals = [metric(enriched, v, "stage_late_response_r_weighted") for v in variants]
    x = np.arange(len(variants))
    for seed_i in range(len(vals[0])):
        ax.plot(x, [v[seed_i] for v in vals], color="#D4D1C8", lw=1.0, zorder=1)
    for i, (name, v) in enumerate(zip(variants, vals)):
        jitter = np.linspace(-0.08, 0.08, len(v))
        ax.scatter(i + jitter, v, s=40, color=VARIANT_COLORS[name],
                   edgecolor="white", lw=0.5, zorder=3)
        ax.hlines(v.mean(), i - 0.18, i + 0.18, color=INK, lw=2.4, zorder=4)
    ax.set_xticks(x, [SHORT[v] for v in variants])
    ax.set_ylabel("Late-stage response $r$")
    ax.set_ylim(-0.05, 1.02)

    n = len(enriched["runs"])
    fig.suptitle(
        "Figure 2 | ChronoGRN predicts held-out TF knockdown conditions",
        x=0.06, y=0.975, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.945,
        f"Held-out = intervention conditions (not TF identity) · early/mid/late stages · "
        f"n = {n} networks  ·  *rows = top-|Δ| targets on observed held-out (viz only)",
        color=MUTED, fontsize=8.5,
    )
    save(fig, outdir, "figure2_unseen_perturbations")


# ===========================================================================
# FIGURE 3
# ===========================================================================
def figure3(outdir, enriched, result_dir):
    truth, data, model, roll, free = get_bundle(result_dir, 0)
    record = enriched["runs"][0]["variants"]["ChronoGRN"]
    true_edges = np.asarray(record["true_edge_series"])
    pred_edges = np.asarray(record["pred_edge_series"])
    active = np.asarray(record["true_active_edges"], dtype=bool)
    times_e = np.asarray(record["edge_times"])
    classes = np.asarray(record.get("edge_class", [["inactive"] * true_edges.shape[2]] * true_edges.shape[1]))

    # top dynamic edges by true variance among active
    variation = np.std(true_edges, axis=0) * active
    selected = sorted(np.argwhere(variation > 0), key=lambda ij: variation[tuple(ij)], reverse=True)

    # small subnetwork for snapshots: top 4 TFs and 10 targets by dynamics
    top_edges = selected[:12]
    tfs_keep = sorted({int(tf) for _, tf in top_edges})[:5]
    genes_keep = sorted({int(g) for g, _ in top_edges})[:10]

    matrices = [pred_edges[0], pred_edges[len(times_e) // 2], pred_edges[-1]]
    labels_snap = ["Early", "Mid", "Late"]
    thr = 0.30 * np.percentile(np.abs(np.stack(matrices)), 95)

    tf, enh, gene, tau_a, tau_t, wmean = pick_activating_chain(truth, free, model)
    gate = 1 / (1 + np.exp(-free[:, model.enhancer_slice][:, enh]))
    enh_acc = free[:, model.enhancer_slice][:, enh]
    w_t = np.array([
        model.edge_weights(torch.tensor(float(t))).detach().numpy()[gene, tf]
        for t in truth.times
    ])
    drive = w_t * gate * free[:, model.tf_slice][:, tf]

    fig = plt.figure(figsize=(12.0, 9.2))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.15, 1.0, 0.95],
                  hspace=0.48, wspace=0.55, left=0.05, right=0.98, top=0.89, bottom=0.06)

    # A small subnetwork snapshots
    pos = {}
    for i, tfi in enumerate(tfs_keep):
        pos[f"TF{tfi}"] = np.array([0.0, i - (len(tfs_keep) - 1) / 2])
    for j, gi in enumerate(genes_keep):
        pos[f"G{gi}"] = np.array([2.0, 0.7 * (j - (len(genes_keep) - 1) / 2)])

    for i, (matrix, lab) in enumerate(zip(matrices, labels_snap)):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        ax.set_title(lab, fontsize=11, fontweight="bold", pad=5)
        if i == 0:
            ax.text(-0.05, 1.12, "a  Dynamic edge reweighting (top subnetwork)",
                    transform=ax.transAxes, fontweight="bold", fontsize=10.5, clip_on=False)
        # draw all nodes
        for tfi in tfs_keep:
            ax.scatter(*pos[f"TF{tfi}"], s=120, c=PURPLE, zorder=3, edgecolors="white")
            ax.text(pos[f"TF{tfi}"][0] - 0.15, pos[f"TF{tfi}"][1], f"TF{tfi}",
                    ha="right", va="center", fontsize=7, color=PURPLE)
        for gi in genes_keep:
            ax.scatter(*pos[f"G{gi}"], s=70, c=ORANGE, zorder=3, edgecolors="white")
            ax.text(pos[f"G{gi}"][0] + 0.12, pos[f"G{gi}"][1], f"G{gi}",
                    ha="left", va="center", fontsize=6.5, color=ORANGE)
        # edges present now
        early_mat = matrices[0]
        for gi in genes_keep:
            for tfi in tfs_keep:
                w = matrix[gi, tfi]
                w0 = early_mat[gi, tfi]
                if abs(w) < thr and abs(w0) < thr:
                    continue
                color = ROSE if w < 0 else TEAL
                width = 0.5 + 2.5 * min(abs(w) / (abs(matrix).max() + 1e-6), 1.0)
                # highlight gained / lost relative to early
                alpha = 0.85
                if abs(w0) < thr and abs(w) >= thr:
                    color = "#1A7A3A"  # gained
                    width += 0.6
                elif abs(w0) >= thr and abs(w) < thr:
                    color = "#999999"  # lost
                    alpha = 0.35
                ax.annotate(
                    "", xy=pos[f"G{gi}"], xytext=pos[f"TF{tfi}"],
                    arrowprops=dict(
                        arrowstyle="-|>" if w >= 0 else "-[",
                        color=color, lw=width, alpha=alpha,
                        mutation_scale=8, shrinkA=5, shrinkB=5,
                    ),
                )
        ax.set_xlim(-0.7, 2.6)
        ys = [p[1] for p in pos.values()]
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        ax.axis("off")
    # legend
    lax = fig.add_axes([0.72, 0.905, 0.25, 0.035])
    lax.axis("off")
    lax.plot([0, 0.08], [0.5, 0.5], color=TEAL, lw=2.5)
    lax.text(0.09, 0.5, "act.", va="center", fontsize=7)
    lax.plot([0.22, 0.30], [0.5, 0.5], color=ROSE, lw=2.5)
    lax.text(0.31, 0.5, "rep.", va="center", fontsize=7)
    lax.plot([0.48, 0.56], [0.5, 0.5], color="#1A7A3A", lw=2.5)
    lax.text(0.57, 0.5, "gained", va="center", fontsize=7)
    lax.plot([0.78, 0.86], [0.5, 0.5], color="#999999", lw=2.0)
    lax.text(0.87, 0.5, "lost", va="center", fontsize=7)
    lax.set_xlim(0, 1.15)

    # B path with repression/activation encoding
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    title(ax, "b  Prior-supported enhancer path")
    sign = np.sign(wmean) if wmean != 0 else 1.0
    nodes = [
        (0.14, 0.55, f"TF{tf}", PURPLE),
        (0.50, 0.55, f"E{enh}", TEAL),
        (0.86, 0.55, f"G{gene}", ORANGE),
    ]
    for x, y, text, color in nodes:
        ax.add_patch(FancyBboxPatch(
            (x - 0.10, y - 0.14), 0.20, 0.28,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor=color, alpha=0.18, lw=1.7, transform=ax.transAxes,
        ))
        ax.text(x, y, text, ha="center", va="center", fontsize=12, fontweight="bold",
                color=INK, transform=ax.transAxes)
    style_arrow = "-|>" if sign >= 0 else "]-"
    col_arrow = TEAL if sign >= 0 else ROSE
    ax.annotate("", xy=(0.40, 0.55), xytext=(0.24, 0.55),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=INK, mutation_scale=12),
                xycoords=ax.transAxes)
    ax.annotate("", xy=(0.76, 0.55), xytext=(0.60, 0.55),
                arrowprops=dict(arrowstyle=style_arrow, lw=2.2, color=col_arrow, mutation_scale=12),
                xycoords=ax.transAxes)
    kind = "activation" if sign >= 0 else "repression"
    gate_mean = float(gate.mean())
    ax.text(0.5, 0.18,
            f"motif-supported · {kind} · mean gate={gate_mean:.2f} (partially permissive)\n"
            f"edge delay τ={tau_t:.2f}",
            ha="center", fontsize=8, color=MUTED, transform=ax.transAxes)

    # C time-varying signed edges (diverse classes)
    ax = fig.add_subplot(gs[1, 2:4])
    clean(ax)
    title(ax, "c  Time-varying signed edges")
    colors = [PURPLE, ORANGE, TEAL, ROSE]
    shown = 0
    for target, tfi in selected:
        if shown >= 4:
            break
        # show truth signed
        color = colors[shown]
        ax.plot(times_e, true_edges[:, target, tfi], color=color, lw=2.2,
                label=f"TF{tfi}→G{target}")
        ax.plot(times_e, pred_edges[:, target, tfi], color=color, lw=1.5, ls=(0, (4, 2)))
        shown += 1
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Trajectory time")
    ax.set_ylabel(r"Signed edge strength $W(t)$")
    ax.legend(ncol=2, fontsize=7, loc="best")
    ax.text(0.98, 0.04, "solid=truth · dashed=learned", transform=ax.transAxes,
            ha="right", color=MUTED, fontsize=7.5)

    # D three gate curves
    ax = fig.add_subplot(gs[1, 4:6])
    clean(ax)
    title(ax, "d  Gate dynamics (one path)")
    # normalize for display
    acc_s = 1 / (1 + np.exp(-enh_acc))
    ax.plot(truth.times, acc_s, color=BLUE, lw=2.0, label="enhancer accessibility")
    ax.plot(truth.times, gate, color=TEAL, lw=2.2, label="learned gate")
    # scaled effective drive
    if np.std(drive) > 1e-8:
        d_sc = (drive - drive.min()) / (drive.max() - drive.min() + 1e-8)
        ax.plot(truth.times, d_sc, color=PURPLE, lw=1.6, ls=(0, (4, 2)),
                label="effective drive (scaled)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Trajectory time"); ax.set_ylabel("Scaled value")
    ax.legend(loc="best", fontsize=7)
    ax.text(0.02, 0.05, f"TF{tf}→E{enh}→G{gene}", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)

    # E dynamic component vs meaningful ablations (not fixed W alone)
    ax = fig.add_subplot(gs[2, 0:3])
    title(ax, "e  Dynamic $\\Delta W(t)$ recovery vs ablations")
    clean(ax)
    variants = ["ChronoGRN", "No delays", "No enhancer gate"]
    vals = [metric(enriched, v, "dynamic_component_r") for v in variants]
    x = np.arange(len(variants))
    for seed_i in range(len(vals[0])):
        ax.plot(x, [v[seed_i] for v in vals], color="#D4D1C8", lw=1.0)
    for i, (name, v) in enumerate(zip(variants, vals)):
        jitter = np.linspace(-0.08, 0.08, len(v))
        ax.scatter(i + jitter, v, s=42, color=VARIANT_COLORS[name],
                   edgecolor="white", lw=0.5, zorder=3)
        ax.hlines(v.mean(), i - 0.18, i + 0.18, color=INK, lw=2.4, zorder=4)
    ax.set_xticks(x, [SHORT[v] for v in variants])
    ax.set_ylabel(r"Corr. of demeaned $W(t)$")
    ax.set_ylim(-0.05, 1.02)
    ax.text(0.98, 0.08, "Fixed $W$ has $\\Delta W\\equiv 0$ by construction (not shown)",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)

    # F summary with separate metric groups
    ax = fig.add_subplot(gs[2, 3:6])
    clean(ax)
    title(ax, "f  Recovery metrics (ChronoGRN)")
    labels = [
        "Edge AUPRC\n(within prior)",
        r"Dynamic $\Delta W$ $r$",
        "Access delay $r$",
        "Edge delay $r$",
    ]
    keys = [
        "edge_auprc", "dynamic_component_r", "access_delay_r",
        "target_delay_r",
    ]
    means, sds = [], []
    for key in keys:
        try:
            m, s = mean_sd(metric(enriched, "ChronoGRN", key))
        except Exception:
            m, s = 0.0, 0.0
        means.append(m); sds.append(s)
    y = np.arange(len(labels))
    colors = [PURPLE, ORANGE, BLUE, TEAL]
    ax.barh(y, means, xerr=sds, color=colors, height=0.62, edgecolor="white",
            error_kw=dict(ecolor=INK, lw=1.0, capsize=3))
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(min(m + s + 0.02, 0.97), i, f"{m:.2f}±{s:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Score (metrics not commensurate)")
    ax.invert_yaxis()
    # prior baseline note
    try:
        prior = mean_sd(metric(enriched, "ChronoGRN", "edge_auprc_prior_baseline"))[0]
        ax.text(0.98, 0.05, f"prior-only AUPRC baseline ≈ {prior:.2f}",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    except Exception:
        pass

    n = len(enriched["runs"])
    fig.suptitle(
        "Figure 3 | ChronoGRN recovers time-varying edges gated by accessible enhancers",
        x=0.05, y=0.975, ha="left", fontsize=13.0, fontweight="bold", color=INK,
    )
    fig.text(
        0.05, 0.945,
        f"Edge AUPRC evaluated within motif/enhancer pathway prior · n = {n} networks · "
        "dynamic and topology recovery are strong on the synthetic positive control",
        color=MUTED, fontsize=8.5,
    )
    save(fig, outdir, "figure3_dynamic_grn")


def summary_table(enriched, outdir):
    import csv
    keys = [
        "access_delay_mae", "target_delay_mae", "access_delay_r", "target_delay_r",
        "target_delay_spearman", "dynamic_component_r", "edge_auprc",
        "edge_auprc_prior_baseline", "control_target_r2", "heldout_perturbation_r",
        "stage_early_response_r_weighted", "stage_mid_response_r_weighted", "stage_late_response_r_weighted",
        "heldout_response_trajectory_r", "heldout_target_rollout_r2",
        "heldout_sign_accuracy",
    ]
    path = outdir / "main_figure_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "metric", "mean", "sd", "n_network_seeds"],
        )
        writer.writeheader()
        for name in VARIANT_COLORS:
            if name not in enriched["runs"][0]["variants"]:
                continue
            for key in keys:
                try:
                    values = metric(enriched, name, key)
                except Exception:
                    continue
                if len(values) == 0:
                    continue
                m, s = mean_sd(values)
                writer.writerow({
                    "model": name, "metric": key, "mean": m, "sd": s,
                    "n_network_seeds": int(len(values)),
                })


def write_explainer(outdir, enriched):
    def msd(key):
        try:
            return mean_sd(metric(enriched, "ChronoGRN", key))
        except Exception:
            return (float("nan"), float("nan"))

    n = len(enriched["runs"])
    text = f"""# ChronoGRN synthetic-benchmark figures (honest claims)

n = {n} independent synthetic networks. Stage-specific held-out TF4/TF5 KDs
(early/mid/late). Continuous delays. Pathway-prior-restricted edge AUPRC.

## Safe claims supported by current results

- ChronoGRN reconstructs delayed multiome trajectories.
- Accessibility→TF delays are recovered well (r ≈ {msd('access_delay_r')[0]:.2f}).
- Directed topology recovery within the motif/enhancer prior is strong
  (AUPRC ≈ {msd('edge_auprc')[0]:.2f}; prior baseline ≈ {msd('edge_auprc_prior_baseline')[0]:.2f}).
- Dynamic edge-shape recovery is strong on the controlled benchmark
  (ΔW r ≈ {msd('dynamic_component_r')[0]:.2f}).
- Held-out intervention conditions are predicted accurately
  (endpoint r ≈ {msd('heldout_perturbation_r')[0]:.2f}).
- The full model improves endpoint, trajectory, and rollout metrics over the
  no-delay, no-gate, and fixed-edge ablations in the paired benchmark.
- Edge-specific delay recovery remains the weakest mechanistic component
  (r ≈ {msd('target_delay_r')[0]:.2f}).

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
"""
    (outdir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/benchmark")
    parser.add_argument(
        "--ablation-results",
        default="results/ablation/results.json",
        help="paired ablation JSON; required for ablation panels",
    )
    parser.add_argument("--out", default="results/figures")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("1", "2", "3"),
        default=("1", "2", "3"),
        help="build a subset of figures; useful on short-lived batch runners",
    )
    args = parser.parse_args()
    result_dir = ROOT / args.results
    ablation_path = ROOT / args.ablation_results if args.ablation_results else None
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    enriched = load_enriched(result_dir, ablation_path)
    style()
    print(f"n_seeds = {len(enriched['runs'])}")
    builders = {
        "1": ("figure1_delayed_dynamics", figure1),
        "2": ("figure2_unseen_perturbations", figure2),
        "3": ("figure3_dynamic_grn", figure3),
    }
    for figure_id in args.figures:
        stem, builder = builders[figure_id]
        for suffix in (".png", ".pdf"):
            path = outdir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
        print(f"Figure {figure_id}…")
        builder(outdir, enriched, result_dir)
    summary_table(enriched, outdir)
    write_explainer(outdir, enriched)
    print(f"Done → {outdir}")


if __name__ == "__main__":
    main()
