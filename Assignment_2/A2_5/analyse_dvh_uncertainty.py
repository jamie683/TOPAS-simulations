"""
A2_5 — plot ensemble DVHs with ±1σ shaded bands for the proton single beam.

Reads ``A2_5/output/uncertainty/aggregated.npz`` (produced by
``run_uncertainty.py``) and produces an uncertainty-aware DVH plot. Each
structure appears as a single mean curve with a ±1σ band of the same colour.

Usage:
    python3 A2_5/analyse_dvh_uncertainty.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "A2_4"))
import dvh_utils  # noqa: E402


AGG_PATH    = os.path.join(SCRIPT_DIR, "output", "uncertainty", "aggregated.npz")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output", "dvh_proton_1beam_uncertainty.png")

STRUCTURE_COLORS = {
    "tumour": "red",
    "ptv":    "magenta",
    "lung_r": "orange",
    "lung_l": "olive",
    "heart":  "blue",
    "cord":   "green",
    "body":   "grey",
}
STRUCTURE_DISPLAY = {
    "tumour": "GTVp",
    "ptv":    "PTV",
    "lung_r": "Right Lung",
    "lung_l": "Left Lung",
    "heart":  "Heart",
    "cord":   "Spinal Cord",
    "body":   "Body",
}
LEGEND_ORDER = ("tumour", "ptv", "lung_r", "lung_l", "heart", "cord", "body")


def main():
    if not os.path.isfile(AGG_PATH):
        print(f"Aggregated file not found: {AGG_PATH}")
        print("Run A2_5/run_uncertainty.py first.")
        return

    centres, agg = dvh_utils.load_aggregated_npz(AGG_PATH)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    n_replicates = None
    for key in LEGEND_ORDER:
        if key not in agg:
            continue
        stats = agg[key]
        if n_replicates is None:
            n_replicates = int(stats.get("n", 0))
        dvh_utils.plot_dvh_with_band(
            ax, centres, stats["mean"], stats["std"],
            color=STRUCTURE_COLORS.get(key, "black"),
            label=STRUCTURE_DISPLAY.get(key, key),
        )

    ax.set_xlabel("Dose (% of plan max)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    title = "Cumulative DVHs — Monoenergetic Proton Beam"
    if n_replicates:
        title += f"\nMean ± 1σ over {n_replicates} independent seeds"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=400)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    # Summary table
    print()
    print("--- DVH ensemble summary (dose as % of plan max) ---")
    print(f'{"Structure":<14s} {"D95":>14s} {"D50":>14s} {"D02":>14s} {"Dmax":>14s}')
    print("-" * 74)
    for key in LEGEND_ORDER:
        if key not in agg:
            continue
        stats = agg[key]

        def fmt(dname):
            m = stats.get(f"{dname}_mean", float("nan"))
            s = stats.get(f"{dname}_std",  float("nan"))
            if not np.isfinite(m):
                return "      —       "
            return f"{m:6.2f} ± {s:4.2f}"
        print(f'{STRUCTURE_DISPLAY.get(key, key):<14s} '
              f'{fmt("D95"):>14s} {fmt("D50"):>14s} {fmt("D02"):>14s} '
              f'{fmt("Dmax"):>14s}')


if __name__ == "__main__":
    main()
