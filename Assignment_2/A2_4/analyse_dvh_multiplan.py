"""
Multi-plan cumulative DVH plotting from VICTORIA exports.

Usage examples:
    cd A2_4 && python3 analyse_dvh_multiplan.py
    python3 A2_4/analyse_dvh_multiplan.py

Default behaviour:
    - looks in output/
    - expects any of these files if present:
        dvh_2beam.csv
        dvh_3beam.csv
        dvh_4beam.csv
        dvh_1field.csv

You can also edit PLAN_FILES below.

Expected VICTORIA export format:
    one dose column + one column per structure, tab/comma/semicolon delimited.

Outputs:
    - one comparison figure per structure across plans
    - one per-plan all-structures DVH figure
    - console summary table with D95/D50/D02 style metrics
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Edit these names to match your VICTORIA exports.
PLAN_FILES = {
    "1-field": "dvh_1field.csv",
    "2-beam": "dvh_2beam.csv",
    "3-beam": "dvh_3beam.csv",
    "4-beam": "dvh_4beam.csv",
}

SMOOTH_DVH = True
SMOOTH_WINDOW = 3

STRUCTURE_COLORS = {
    "GTVp": "red",
    "PTV": "purple",
    "Lung_R": "orange",
    "Lung_L": "olive",
    "Heart": "blue",
    "SpinalCord": "green",
    "BODY": "grey",
    "Body": "grey",
}

STRUCTURE_DISPLAY_NAMES = {
    "GTVp": "GTVp",
    "PTV": "PTV",
    "Lung_R": "Right Lung",
    "Lung_L": "Left Lung",
    "Heart": "Heart",
    "SpinalCord": "Spinal Cord",
    "BODY": "Body",
    "Body": "Body",
}

PLAN_LINESTYLES = {
    "1-field": "--",
    "2-beam": "-",
    "3-beam": "-.",
    "4-beam": ":",
}

LEGEND_ORDER = ["GTVp", "PTV", "Lung_R", "Lung_L", "Heart", "SpinalCord", "BODY", "Body"]
COMPARE_STRUCTURES = ["GTVp", "Heart", "SpinalCord", "Lung_R", "BODY"]


# ------------------------------------------------------------------
# Flexible DVH parser
# ------------------------------------------------------------------
def detect_delimiter(filepath):
    with open(filepath, "r") as f:
        sample_lines = []
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                sample_lines.append(stripped)
            if len(sample_lines) >= 5:
                break
    sample = "\n".join(sample_lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return "\t"


def is_dose_column(name):
    lower = name.strip().lower()
    return "dose" in lower or lower.endswith("gy") or "cgy" in lower


def load_dvh_file(filepath):
    delimiter = detect_delimiter(filepath)
    header = None
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = [fld.strip() for fld in stripped.split(delimiter) if fld.strip()]
            if not fields:
                continue

            if header is None:
                non_numeric = 0
                for field in fields:
                    try:
                        float(field)
                    except ValueError:
                        non_numeric += 1
                if non_numeric > 0:
                    header = fields
                    continue
                else:
                    header = [f"col_{i}" for i in range(len(fields))]

            row = []
            for field in fields:
                try:
                    row.append(float(field))
                except ValueError:
                    row.append(np.nan)
            data_lines.append(row)

    if not data_lines:
        raise ValueError(f"No numeric data found in {filepath}")

    max_cols = max(len(r) for r in data_lines)
    for row in data_lines:
        while len(row) < max_cols:
            row.append(np.nan)

    data = np.array(data_lines)
    while len(header) < data.shape[1]:
        header.append(f"col_{len(header)}")
    header = header[: data.shape[1]]

    dose_idx = None
    for i, name in enumerate(header):
        if is_dose_column(name):
            dose_idx = i
            break
    if dose_idx is None:
        dose_idx = 0

    dose = data[:, dose_idx]
    structures = {}
    for i, name in enumerate(header):
        if i == dose_idx:
            continue
        col = data[:, i]
        if np.all(np.isnan(col)):
            continue
        structures[name] = col

    return dose, structures


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def smooth(y, window):
    if window < 2 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def sort_structures(structures):
    ordered = [s for s in LEGEND_ORDER if s in structures]
    extra = [s for s in structures if s not in LEGEND_ORDER]
    return ordered + sorted(extra)


def display_name(raw_name):
    return STRUCTURE_DISPLAY_NAMES.get(raw_name, raw_name)


def dose_at_volume(dose_pct, volume_pct, target_vol):
    above = volume_pct >= target_vol
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0][-1]
    if idx >= len(dose_pct) - 1:
        return float(dose_pct[idx])
    v0, v1 = volume_pct[idx], volume_pct[idx + 1]
    d0, d1 = dose_pct[idx], dose_pct[idx + 1]
    if v0 == v1:
        return float(d0)
    frac = (target_vol - v0) / (v1 - v0)
    return float(d0 + frac * (d1 - d0))


def max_nonzero_dose(dose_pct, volume_pct):
    nonzero = volume_pct > 0.5
    if np.any(nonzero):
        return float(dose_pct[np.where(nonzero)[0][-1]])
    return 0.0


def get_color(structure_name, fallback_idx=0):
    if structure_name in STRUCTURE_COLORS:
        return STRUCTURE_COLORS[structure_name]
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return default_cycle[fallback_idx % len(default_cycle)]


def normalise_dose(dose_raw):
    dose_max = np.nanmax(dose_raw)
    if dose_max > 0:
        return dose_raw / dose_max * 100.0
    return dose_raw


def normalise_volume(vol):
    return vol * 100.0 if np.nanmax(vol) <= 1.01 else vol


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
def print_plan_summary(plan_name, dose_pct, structures):
    print(f"\n--- {plan_name} ---")
    print(f"{'Structure':<16s} {'D95 (%)':>10s} {'D50 (%)':>10s} {'D02 (%)':>10s} {'Max (%)':>10s}")
    print("-" * 60)
    for name in sort_structures(structures):
        vol_pct = normalise_volume(structures[name])
        d95 = dose_at_volume(dose_pct, vol_pct, 95.0)
        d50 = dose_at_volume(dose_pct, vol_pct, 50.0)
        d02 = dose_at_volume(dose_pct, vol_pct, 2.0)
        dmax = max_nonzero_dose(dose_pct, vol_pct)
        def fmt(x):
            return "—" if np.isnan(x) else f"{x:10.1f}"
        print(f"{display_name(name):<16s} {fmt(d95)} {fmt(d50)} {fmt(d02)} {fmt(dmax)}")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_all_structures_for_plan(plan_name, dose_pct, structures, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered_names = sort_structures(structures)

    for i, name in enumerate(ordered_names):
        vol_plot = normalise_volume(structures[name])
        if SMOOTH_DVH:
            vol_plot = smooth(vol_plot, SMOOTH_WINDOW)
        ax.plot(dose_pct, vol_plot, color=get_color(name, i), linewidth=1.6, label=display_name(name))

    ax.set_xlabel("Dose (% of plan max)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title(f"Cumulative DVHs — {plan_name}", fontsize=14)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_structure_across_plans(structure_name, plan_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for i, (plan_name, payload) in enumerate(plan_data.items()):
        dose_pct = payload["dose_pct"]
        structures = payload["structures"]
        if structure_name not in structures:
            continue
        vol_plot = normalise_volume(structures[structure_name])
        if SMOOTH_DVH:
            vol_plot = smooth(vol_plot, SMOOTH_WINDOW)
        ax.plot(
            dose_pct,
            vol_plot,
            linestyle=PLAN_LINESTYLES.get(plan_name, "-"),
            linewidth=2.0,
            label=plan_name,
        )
        plotted = True

    if not plotted:
        return

    ax.set_xlabel("Dose (% of plan max)")
    ax.set_ylabel("Volume (%)")
    ax.set_title(f"Cumulative DVH comparison — {display_name(structure_name)}")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gtv_and_oars(plan_data, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for plan_idx, (plan_name, payload) in enumerate(plan_data.items()):
        dose_pct = payload["dose_pct"]
        structures = payload["structures"]
        for struct_name in ["GTVp", "Heart", "SpinalCord", "Lung_R"]:
            if struct_name not in structures:
                continue
            vol_plot = normalise_volume(structures[struct_name])
            if SMOOTH_DVH:
                vol_plot = smooth(vol_plot, SMOOTH_WINDOW)
            ax.plot(
                dose_pct,
                vol_plot,
                color=get_color(struct_name, plan_idx),
                linestyle=PLAN_LINESTYLES.get(plan_name, "-"),
                linewidth=1.7,
                label=f"{plan_name} — {display_name(struct_name)}",
            )
            plotted = True
    if not plotted:
        return
    ax.set_xlabel("Dose (% of plan max)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("Cumulative DVH comparison — key structures across plans")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plan_data = {}
    print("Searching for DVH files in:", INPUT_DIR)

    for plan_name, filename in PLAN_FILES.items():
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"Missing: {filepath}  (skipping {plan_name})")
            continue
        dose_raw, structures = load_dvh_file(filepath)
        plan_data[plan_name] = {
            "filename": filename,
            "dose_raw": dose_raw,
            "structures": structures,
        }
        print(f"Loaded {plan_name}: {filename}")

    # Normalise dose to % of each plan's own maximum, so DVH curves
    # span 0-100% on the x-axis regardless of absolute dose scale.
    # This matches how VICTORIA displays DVH comparisons.
    for payload in plan_data.values():
        dmax = np.nanmax(payload["dose_raw"])
        if dmax > 0:
            payload["dose_pct"] = payload["dose_raw"] / dmax * 100.0
        else:
            payload["dose_pct"] = payload["dose_raw"]

    if not plan_data:
        print("No DVH files found. Update PLAN_FILES to match your exports.")
        return

    for plan_name, payload in plan_data.items():
        print_plan_summary(plan_name, payload["dose_pct"], payload["structures"])
        out_single = os.path.join(OUTPUT_DIR, f"dvh_{plan_name.replace('-', '_')}_all_structures.png")
        plot_all_structures_for_plan(plan_name, payload["dose_pct"], payload["structures"], out_single)

    for structure_name in COMPARE_STRUCTURES:
        out_compare = os.path.join(OUTPUT_DIR, f"dvh_compare_{structure_name}.png")
        plot_structure_across_plans(structure_name, plan_data, out_compare)

    out_key = os.path.join(OUTPUT_DIR, "dvh_compare_key_structures.png")
    plot_gtv_and_oars(plan_data, out_key)

    print("\nDone.")
    print("If your VICTORIA export filenames differ, edit PLAN_FILES near the top of the script.")


if __name__ == "__main__":
    main()
