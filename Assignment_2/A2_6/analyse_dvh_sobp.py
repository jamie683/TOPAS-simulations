"""
Section 2.6: Cumulative DVH plotting from VICTORIA SOBP export.

Usage:
    cd A2_6 && python3 analyse_dvh_sobp.py
    or from project root:
    python3 A2_6/analyse_dvh_sobp.py

Expects one DVH CSV file exported from VICTORIA in output/.
Default filename: dvh_sobp.csv

Dose axis is normalised to % of maximum dose for plotting.

Outputs:
    output/dvh_sobp.png                 cumulative DVH plot
    Printed summary table with D95, D50, D02, and max dose per structure
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
DVH_FILE = "dvh_sobp.csv"

SMOOTH_DVH = True
SMOOTH_WINDOW = 3

STRUCTURE_COLORS = {
    "GTVp":       "red",
    "PTV":        "purple",
    "Lung_R":     "orange",
    "Lung_L":     "olive",
    "Heart":      "blue",
    "SpinalCord": "green",
    "BODY":       "grey",
    "Body":       "grey",
}

STRUCTURE_DISPLAY_NAMES = {
    "GTVp":       "GTVp",
    "PTV":        "PTV",
    "Lung_R":     "Right Lung",
    "Lung_L":     "Left Lung",
    "Heart":      "Heart",
    "SpinalCord": "Spinal Cord",
    "BODY":       "Body",
    "Body":       "Body",
}

LEGEND_ORDER = ["GTVp", "PTV", "Lung_R", "Lung_L", "Heart", "SpinalCord", "BODY", "Body"]


# ------------------------------------------------------------------
# DVH file parser
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
    """
    Parse a single VICTORIA DVH export.
    Returns (dose_array, {structure_name: volume_array}).
    """
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
                non_numeric = sum(1 for fld in fields if not _is_float(fld))
                if non_numeric > 0:
                    header = fields
                    continue
                else:
                    header = [f"col_{i}" for i in range(len(fields))]

            row = []
            for fld in fields:
                try:
                    row.append(float(fld))
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
    header = header[:data.shape[1]]

    dose_idx = None
    for i, name in enumerate(header):
        if is_dose_column(name):
            dose_idx = i
            break
    if dose_idx is None:
        print(f"Warning: no 'dose' column found in header {header}. Using column 0.")
        dose_idx = 0

    dose = data[:, dose_idx]

    structures = {}
    for i, name in enumerate(header):
        if i == dose_idx:
            continue
        col = data[:, i]
        if not np.all(np.isnan(col)):
            structures[name] = col

    print(f"Loaded {filepath}:")
    print(f"  Dose column: '{header[dose_idx]}' ({len(dose)} points)")
    print(f"  Structures:  {list(structures.keys())}")

    return dose, structures


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def smooth(y, window):
    if window < 2 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def sort_structures(structures):
    ordered = [s for s in LEGEND_ORDER if s in structures]
    extra = [s for s in structures if s not in LEGEND_ORDER]
    return ordered + sorted(extra)


def display_name(raw):
    return STRUCTURE_DISPLAY_NAMES.get(raw, raw)


def normalise_volume(vol):
    return vol * 100.0 if np.nanmax(vol) <= 1.01 else vol


def dose_at_volume(dose_pct, vol_pct, target_vol):
    """Interpolate dose at which cumulative DVH crosses target_vol (%)."""
    above = vol_pct >= target_vol
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0][-1]
    if idx >= len(dose_pct) - 1:
        return float(dose_pct[idx])
    v0, v1 = vol_pct[idx], vol_pct[idx + 1]
    d0, d1 = dose_pct[idx], dose_pct[idx + 1]
    if v0 == v1:
        return float(d0)
    frac = (target_vol - v0) / (v1 - v0)
    return float(d0 + frac * (d1 - d0))


def max_nonzero_dose(dose_pct, vol_pct):
    """Highest dose bin where volume > 0.5% (ignores noise floor)."""
    nonzero = vol_pct > 0.5
    if np.any(nonzero):
        return float(dose_pct[np.where(nonzero)[0][-1]])
    return 0.0


def get_color(name, fallback_idx):
    if name in STRUCTURE_COLORS:
        return STRUCTURE_COLORS[name]
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return default_cycle[fallback_idx % len(default_cycle)]


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
def print_summary(dose_pct, structures):
    print("\n" + "=" * 68)
    print("SOBP PROTON DVH SUMMARY (dose as % of maximum)")
    print("=" * 68)
    print(f"  {'Structure':<16s} {'Max dose (%)':<14s} "
          f"{'D95 (%)':<10s} {'D50 (%)':<10s} {'D02 (%)':<10s}")
    print(f"  {'-' * 60}")

    for name in sort_structures(structures):
        vol_pct = normalise_volume(structures[name])
        md = max_nonzero_dose(dose_pct, vol_pct)
        d95 = dose_at_volume(dose_pct, vol_pct, 95.0)
        d50 = dose_at_volume(dose_pct, vol_pct, 50.0)
        d02 = dose_at_volume(dose_pct, vol_pct, 2.0)

        def fmt(x):
            return f"{x:.1f}" if not np.isnan(x) else "—"

        print(f"  {display_name(name):<16s} {md:<14.1f} "
              f"{fmt(d95):<10s} {fmt(d50):<10s} {fmt(d02):<10s}")

    print()


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_cumulative_dvh(dose_pct, structures, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(sort_structures(structures)):
        vol_plot = normalise_volume(structures[name])
        if SMOOTH_DVH:
            vol_plot = smooth(vol_plot, SMOOTH_WINDOW)

        ax.plot(dose_pct, vol_plot, color=get_color(name, i),
                linewidth=1.5, label=display_name(name))

    ax.set_xlabel("Dose (% of maximum)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("Cumulative DVHs \u2014 SOBP Proton Treatment (Normalised Dose)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filepath = os.path.join(INPUT_DIR, DVH_FILE)
    if not os.path.isfile(filepath):
        print(f"DVH file not found: {filepath}")
        print(f"Export DVH data from VICTORIA and save as: output/{DVH_FILE}")
        return

    dose_raw, structures = load_dvh_file(filepath)

    if not structures:
        print("No structure columns found. Nothing to plot.")
        return

    # Normalise dose to % of maximum
    dose_max = np.nanmax(dose_raw)
    if dose_max > 0:
        dose_pct = dose_raw / dose_max * 100.0
    else:
        print("Warning: maximum dose is zero. Plotting raw values.")
        dose_pct = dose_raw

    print_summary(dose_pct, structures)

    output_path = os.path.join(OUTPUT_DIR, "dvh_sobp.png")
    plot_cumulative_dvh(dose_pct, structures, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
