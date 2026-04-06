"""
Section 2.3: Cumulative DVH plotting from a single VICTORIA export.

Usage:
    cd A2_3 && python3 analyse_dvh.py
    or from project root:
    python3 A2_3/analyse_dvh.py

Expects one DVH CSV file exported from VICTORIA in output/.
Default filename: dvh_export.csv (change DVH_FILE below if needed).

Expected format (tab or comma delimited):
    dose    GTVp    Lung_R    Heart    SpinalCord    ...
    0       1       1         1        1
    4.99E-07  0.295  0.118    0        0

The script auto-detects:
  - Delimiter (tab, comma, semicolon)
  - Dose column (name containing "dose" or "gy")
  - Structure columns (all other columns with numeric data)

Dose axis is normalised to % of maximum dose for plotting.
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
DVH_FILE = "dvh_export.csv"

# Set to True to apply a mild 3-point moving average to DVH curves.
# Reduces Monte Carlo jaggedness without distorting shape.
SMOOTH_DVH = True
SMOOTH_WINDOW = 3

# Preferred colours for known structures.
STRUCTURE_COLORS = {
    "GTVp":       "red",
    "PTV":        "purple",
    "Lung_R":     "orange",
    "Lung_L":     "olive",
    "Heart":      "blue",
    "SpinalCord": "green",
    "BODY":       "grey",
}

# Display-friendly names for legend.
STRUCTURE_DISPLAY_NAMES = {
    "GTVp":       "GTVp",
    "PTV":        "PTV",
    "Lung_R":     "Right Lung",
    "Lung_L":     "Left Lung",
    "Heart":      "Heart",
    "SpinalCord": "Spinal Cord",
    "BODY":       "Body",
}

# Preferred legend order. Structures not listed appear afterwards.
LEGEND_ORDER = ["GTVp", "PTV", "Lung_R", "Lung_L", "Heart", "SpinalCord", "BODY"]


# ------------------------------------------------------------------
# Flexible single-file DVH parser
# ------------------------------------------------------------------
def detect_delimiter(filepath):
    """Sniff the delimiter from the first non-comment lines."""
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
    """Check if a column name looks like the dose column."""
    lower = name.strip().lower()
    return "dose" in lower or "gy" in lower or "cgy" in lower


def load_dvh_file(filepath):
    """
    Parse a single VICTORIA DVH export containing one dose column
    and multiple structure columns.

    Returns:
        dose: 1D numpy array (raw values as exported)
        structures: dict of {column_name: 1D numpy array}
    """
    delimiter = detect_delimiter(filepath)

    header = None
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = [f.strip() for f in stripped.split(delimiter) if f.strip()]

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
        if np.all(np.isnan(col)):
            continue
        structures[name] = col

    print(f"Loaded {filepath}:")
    print(f"  Dose column: '{header[dose_idx]}' ({len(dose)} points)")
    print(f"  Structures:  {list(structures.keys())}")

    return dose, structures


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def smooth(y, window):
    """Uniform moving average. Returns array of same length."""
    if window < 2 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    # Pad edges to preserve array length
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def sort_structures(structures):
    """Return structure names sorted by preferred legend order."""
    ordered = [s for s in LEGEND_ORDER if s in structures]
    extra = [s for s in structures if s not in LEGEND_ORDER]
    return ordered + sorted(extra)


def display_name(raw_name):
    """Return a display-friendly name, or the raw name if unknown."""
    return STRUCTURE_DISPLAY_NAMES.get(raw_name, raw_name)


def dose_at_volume(dose_pct, volume_pct, target_vol):
    """
    Interpolate the dose at which the cumulative DVH crosses a given
    volume fraction. Returns NaN if the curve never reaches that level.
    """
    # volume_pct should be decreasing; find where it crosses target_vol
    above = volume_pct >= target_vol
    if not np.any(above):
        return np.nan
    # Last index where volume >= target
    idx = np.where(above)[0][-1]
    if idx >= len(dose_pct) - 1:
        return float(dose_pct[idx])
    # Linear interpolation between idx and idx+1
    v0, v1 = volume_pct[idx], volume_pct[idx + 1]
    d0, d1 = dose_pct[idx], dose_pct[idx + 1]
    if v0 == v1:
        return float(d0)
    frac = (target_vol - v0) / (v1 - v0)
    return float(d0 + frac * (d1 - d0))


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
def print_summary(dose_pct, structures):
    """Print a concise quantitative summary for each structure."""
    print("\n--- DVH Summary (dose as % of maximum) ---")
    print(f"{'Structure':<16s} {'Max dose (%)':<14s} {'D50 (%)':<10s}")
    print("-" * 42)

    for name in sort_structures(structures):
        vol = structures[name]
        vol_pct = vol * 100.0 if np.nanmax(vol) <= 1.0 else vol

        # Max dose: highest dose bin where volume > 0
        nonzero = vol_pct > 0.5  # threshold at 0.5% to ignore noise
        if np.any(nonzero):
            max_dose = float(dose_pct[np.where(nonzero)[0][-1]])
        else:
            max_dose = 0.0

        d50 = dose_at_volume(dose_pct, vol_pct, 50.0)

        d50_str = f"{d50:.1f}" if not np.isnan(d50) else "—"
        print(f"{display_name(name):<16s} {max_dose:<14.1f} {d50_str:<10s}")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def get_color(structure_name, fallback_idx):
    """Return a colour for the structure, falling back to the default cycle."""
    if structure_name in STRUCTURE_COLORS:
        return STRUCTURE_COLORS[structure_name]
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return default_cycle[fallback_idx % len(default_cycle)]


def plot_cumulative_dvh(dose_pct, structures, output_path):
    """Plot cumulative DVHs with dose normalised to % of maximum."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ordered_names = sort_structures(structures)

    for i, name in enumerate(ordered_names):
        volume = structures[name]
        vol_plot = volume * 100.0 if np.nanmax(volume) <= 1.0 else volume

        if SMOOTH_DVH:
            vol_plot = smooth(vol_plot, SMOOTH_WINDOW)

        color = get_color(name, i)
        ax.plot(dose_pct, vol_plot, color=color, linewidth=1.5,
                label=display_name(name))

    ax.set_xlabel("Dose (% of maximum)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title("Cumulative DVHs \u2014 Single 1 MeV Photon Field (Normalised Dose)", fontsize=14)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filepath = os.path.join(INPUT_DIR, DVH_FILE)
    if not os.path.isfile(filepath):
        print(f"DVH file not found: {filepath}")
        print(f"Export DVH data from VICTORIA and save as {filepath}")
        return

    dose_raw, structures = load_dvh_file(filepath)

    if not structures:
        print("No structure columns found. Nothing to plot.")
        return

    # Normalise dose to % of maximum for plotting and summary
    dose_max = np.nanmax(dose_raw)
    if dose_max > 0:
        dose_pct = dose_raw / dose_max * 100.0
    else:
        print("Warning: maximum dose is zero. Plotting raw values.")
        dose_pct = dose_raw

    print_summary(dose_pct, structures)

    output_path = os.path.join(OUTPUT_DIR, "dvh_photon_1field.png")
    plot_cumulative_dvh(dose_pct, structures, output_path)


if __name__ == "__main__":
    main()
