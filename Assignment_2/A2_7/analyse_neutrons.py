"""
Section 2.7: Neutron Production Analysis

Reads TOPAS output from the brass collimator simulation and produces:
  - Beam profile plot confirming ~8 cm collimated spot
  - Depth-dose comparison (total vs neutron-incident)
  - Neutron energy spectrum (log-scaled)
  - ICRP 60 weighted-average RBE of incident neutrons

Normalisation convention:
  "1 Gy delivered to the phantom" means 1 Gy *mean* dose averaged over
  the entire water phantom volume.  All plotted quantities are scaled by
  the factor (1 Gy / raw_mean_dose).

Usage:
    python3 analyse_neutrons.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

BEAM_PROFILE_CSV   = os.path.join(OUTPUT_DIR, "beam_profile.csv")
TOTAL_DOSE_CSV     = os.path.join(OUTPUT_DIR, "total_dose.csv")
NEUTRON_DOSE_CSV   = os.path.join(OUTPUT_DIR, "neutron_dose.csv")
NEUTRON_PS_PHSP    = os.path.join(OUTPUT_DIR, "neutron_phasespace.phsp")
NEUTRON_PS_HDR     = os.path.join(OUTPUT_DIR, "neutron_phasespace.header")
TOTAL_DOSE_XY_CSV  = os.path.join(OUTPUT_DIR, "total_dose_xy.csv")
NEUTRON_DOSE_XY_CSV = os.path.join(OUTPUT_DIR, "neutron_dose_xy.csv")

# Collimator thickness sensitivity study (3 cm variant)
THIN_PREFIX = "_3cm"
TOTAL_DOSE_CSV_3CM     = os.path.join(OUTPUT_DIR, "total_dose_3cm.csv")
NEUTRON_DOSE_CSV_3CM   = os.path.join(OUTPUT_DIR, "neutron_dose_3cm.csv")
NEUTRON_PS_PHSP_3CM    = os.path.join(OUTPUT_DIR, "neutron_phasespace_3cm.phsp")
NEUTRON_PS_HDR_3CM     = os.path.join(OUTPUT_DIR, "neutron_phasespace_3cm.header")

# XY scorer dimensions (must match TOPAS file)
PHANTOM_NXY = 100  # XY bins in radial dose scorers

# Phantom geometry (must match TOPAS file)
PHANTOM_HLX = 10.0   # cm
PHANTOM_HLY = 10.0
PHANTOM_HLZ = 10.0
PHANTOM_NZ  = 200     # Z bins in dose scorers
PHANTOM_FRONT_Z = 0.0  # cm (front face of phantom in world coords)

# ICRP 60 neutron RBE weighting factors
# (E_low MeV, E_high MeV, wR)
ICRP60_BINS = [
    (0.0,   0.01,   5),    # < 10 keV
    (0.01,  0.1,   10),    # 10 keV to < 100 keV
    (0.1,   2.0,   20),    # 100 keV to < 2 MeV
    (2.0,  20.0,   10),    # 2 MeV to < 20 MeV
    (20.0, np.inf,  5),    # >= 20 MeV
]


# ------------------------------------------------------------------
# TOPAS CSV parser
# ------------------------------------------------------------------
def load_topas_1d_dose(csv_path, n_bins):
    """Load a TOPAS CSV dose scorer with 1x1xN binning.

    Returns depth bin centres (cm, relative to phantom front face)
    and dose values (Gy per history, as TOPAS outputs).
    """
    values = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                int(parts[0])  # bin index — confirms it's a data line
                values.append(float(parts[3]))
            except (ValueError, IndexError):
                continue

    dose = np.array(values)
    # Bin centres: phantom extends from 0 to 2*HLZ in depth
    dz = 2.0 * PHANTOM_HLZ / n_bins
    depth_centres = np.arange(n_bins) * dz + dz / 2.0  # cm from front face
    return depth_centres[:len(dose)], dose


def load_topas_2d_fluence(csv_path, nx, ny):
    """Load a TOPAS CSV fluence scorer with NxNx1 binning.

    Returns x_centres, y_centres (cm), and 2D fluence array.
    """
    data = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                ix, iy = int(parts[0]), int(parts[1])
                data[(ix, iy)] = float(parts[3])
            except (ValueError, IndexError):
                continue

    fluence = np.zeros((ny, nx))
    for (ix, iy), val in data.items():
        if 0 <= ix < nx and 0 <= iy < ny:
            fluence[iy, ix] = val

    dx = 2.0 * PHANTOM_HLX / nx
    dy = 2.0 * PHANTOM_HLY / ny
    x_centres = np.arange(nx) * dx + dx / 2.0 - PHANTOM_HLX
    y_centres = np.arange(ny) * dy + dy / 2.0 - PHANTOM_HLY
    return x_centres, y_centres, fluence


# ------------------------------------------------------------------
# Phase space parser
# ------------------------------------------------------------------
def load_neutron_energies(phsp_path, header_path):
    """Load neutron energies from a TOPAS ASCII phase space file.

    Returns array of kinetic energies (MeV) for neutrons heading
    toward the phantom (+Z direction only).
    """
    # Parse the header to find column layout
    # Default TOPAS ASCII columns:
    #   X Y Z CosX CosY E Weight PDG IsNewHistory ...
    # Energy sign convention: negative E means particle travels in -Z.
    # We keep only +Z-directed particles (positive E or CosZ > 0).

    energies = []

    if os.path.isfile(phsp_path):
        with open(phsp_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    energy = float(parts[5])  # column 5 = energy (MeV)
                except (ValueError, IndexError):
                    continue
                # TOPAS convention: negative energy => travelling in -Z
                # We want neutrons heading toward phantom (+Z direction)
                if energy > 0:
                    energies.append(energy)

    if not energies:
        print(f"Warning: no +Z neutrons found in {phsp_path}")
    return np.array(energies)


# ------------------------------------------------------------------
# ICRP 60 RBE calculation
# ------------------------------------------------------------------
def compute_icrp60_rbe(energies_mev):
    """Compute fluence-weighted average RBE from ICRP 60 table.

    Parameters:
        energies_mev : 1D array of individual neutron energies (MeV)

    Returns:
        average_rbe  : float
        bin_counts   : list of (E_low, E_high, wR, count, fraction)
    """
    if len(energies_mev) == 0:
        return 0.0, []

    total = len(energies_mev)
    weighted_sum = 0.0
    bin_info = []

    for e_low, e_high, wr in ICRP60_BINS:
        if np.isinf(e_high):
            mask = energies_mev >= e_low
        else:
            mask = (energies_mev >= e_low) & (energies_mev < e_high)
        count = int(np.sum(mask))
        frac = count / total if total > 0 else 0.0
        weighted_sum += count * wr
        bin_info.append((e_low, e_high, wr, count, frac))

    avg_rbe = weighted_sum / total if total > 0 else 0.0
    return avg_rbe, bin_info


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def compute_beam_diameter_2d(x, y, fluence):
    """Compute beam diameter from the 2D fluence map using an area method.

    For a collimated flat-top beam the 1D profile can be misleading due
    to scatter halo.  Instead we:
      1. Lightly smooth the 2D fluence to suppress pixel noise.
      2. Estimate the plateau from a small central region (r < 2 cm).
      3. Threshold at 50% of that plateau (standard field-edge definition).
      4. Compute the enclosed area and derive diameter = 2 * sqrt(A / pi).

    Returns (diameter_cm, plateau_level).
    """
    from scipy.ndimage import uniform_filter

    # Light 2D smooth (3×3 box filter)
    smoothed = uniform_filter(fluence, size=3)

    # Build radial distance grid
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    # Plateau: median of smoothed fluence within r < 2 cm (well inside the
    # expected ~4 cm beam radius, avoids penumbra and halo entirely)
    core_mask = rr < 2.0
    if np.sum(core_mask) < 5:
        core_mask = rr < 3.0  # fallback if bins are very coarse
    plateau = np.median(smoothed[core_mask])

    if plateau <= 0:
        return 0.0, 0.0

    # Field boundary: 50% of plateau on the smoothed map
    field_mask = smoothed >= 0.5 * plateau

    # Pixel area
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    field_area = np.sum(field_mask) * dx * dy  # cm²

    # Equivalent circular diameter
    diameter = 2.0 * np.sqrt(field_area / np.pi)

    return diameter, plateau


def plot_beam_profile(x, y, fluence, filepath):
    """Plot 2D beam fluence with field-edge contour, and 1D X-profile."""
    from scipy.ndimage import uniform_filter

    diameter, plateau = compute_beam_diameter_2d(x, y, fluence)
    beam_radius = diameter / 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2D fluence map with field-edge contour
    extent = [x[0] - (x[1]-x[0])/2, x[-1] + (x[1]-x[0])/2,
              y[0] - (y[1]-y[0])/2, y[-1] + (y[1]-y[0])/2]
    im = ax1.imshow(fluence, extent=extent, origin="lower",
                    aspect="equal", cmap="inferno")
    # Overlay 50%-plateau contour
    smoothed = uniform_filter(fluence, size=3)
    ax1.contour(x, y, smoothed, levels=[0.5 * plateau],
                colors="lime", linewidths=1.5, linestyles="--")
    # Overlay equivalent circle
    circle = plt.Circle((0, 0), beam_radius, fill=False, color="white",
                         linewidth=1.5, linestyle=":")
    ax1.add_patch(circle)
    ax1.set_xlabel("X (cm)", fontsize=14)
    ax1.set_ylabel("Y (cm)", fontsize=14)
    ax1.set_title("Beam Fluence at Phantom Surface", fontsize=15)
    ax1.tick_params(labelsize=12)
    fig.colorbar(im, ax=ax1, label="Fluence (per cm$^2$ per history)")

    # 1D X-profile through centre
    mid_y = fluence.shape[0] // 2
    profile = fluence[mid_y, :]
    ax2.plot(x, profile, "k-", linewidth=1.5)
    ax2.set_xlabel("X (cm)", fontsize=14)
    ax2.set_ylabel("Fluence (per cm$^2$ per history)", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    # Mark plateau and 50% threshold on 1D profile
    ax2.axhline(plateau, color="green", linestyle="-", alpha=0.4,
                label=f"Central plateau")
    ax2.axhline(0.5 * plateau, color="red", linestyle="--", alpha=0.6,
                label=f"50% plateau (field edge)")
    # Mark the derived beam edges on the 1D plot
    ax2.axvline(-beam_radius, color="blue", linestyle=":", alpha=0.6)
    ax2.axvline(beam_radius, color="blue", linestyle=":", alpha=0.6,
                label=f"Beam edge (d = {diameter:.1f} cm)")

    ax2.set_title(f"Central X-Profile — Beam Diameter = {diameter:.1f} cm",
                  fontsize=15)
    ax2.legend(fontsize=10, loc="upper right")

    print(f"  Beam diameter (2D area method): {diameter:.1f} cm")
    print(f"  Central plateau level: {plateau:.2e}")

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_depth_dose(depth, total_dose, neutron_dose, filepath):
    """Plot total and neutron depth-dose profiles (normalised to 1 Gy mean)."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(depth, total_dose, "b-", linewidth=2.0, label="Total dose")
    ax.plot(depth, neutron_dose, "r-", linewidth=1.5, label="Neutron dose")

    ax.set_xlabel("Depth in water (cm)", fontsize=14)
    ax.set_ylabel("Dose (Gy)", fontsize=14)
    ax.set_title("Depth-Dose Profile — Total vs Neutron\n"
                 "(Normalised to 1 Gy mean phantom dose)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_neutron_spectrum(energies_mev, filepath):
    """Plot neutron energy spectrum with log-spaced bins.

    Thermal neutrons (< 1 eV = 1e-6 MeV) are grouped into a single bin
    and annotated separately, since they are below the lowest ICRP 60
    energy boundary and do not contribute significant dose.
    """
    if len(energies_mev) == 0:
        print("No neutron energies to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))

    thermal_cut = 1e-6  # 1 eV in MeV
    n_thermal = int(np.sum(energies_mev < thermal_cut))
    above_thermal = energies_mev[energies_mev >= thermal_cut]

    # Log-spaced bins from 1 eV to above max energy
    e_min = thermal_cut
    e_max = max(above_thermal.max() * 1.2, 250.0) if len(above_thermal) else 250.0
    bin_edges = np.logspace(np.log10(e_min), np.log10(e_max), 70)

    counts, edges = np.histogram(above_thermal, bins=bin_edges)
    bin_centres = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    bin_widths = edges[1:] - edges[:-1]

    # dN/dE spectral density; mask empty bins to avoid log-scale issues
    spectrum = np.where(counts > 0, counts / bin_widths, np.nan)

    ax.step(bin_centres, spectrum, where="mid", color="darkblue", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Neutron Energy (MeV)", fontsize=14)
    ax.set_ylabel("dN/dE (neutrons per MeV)", fontsize=14)
    ax.set_title("Neutron Energy Spectrum at Phantom Surface", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, which="both")

    # Shade ICRP 60 energy bands and label
    band_colors = ["#cce5ff", "#ffe0cc", "#ffcccc", "#ffe0cc", "#cce5ff"]
    for (e_lo, e_hi, wr), col in zip(ICRP60_BINS, band_colors):
        hi = min(e_hi, e_max) if not np.isinf(e_hi) else e_max
        lo = max(e_lo, e_min)
        if lo >= hi:
            continue
        ax.axvspan(lo, hi, alpha=0.15, color=col)
        label_x = np.sqrt(lo * hi)
        # Place wR labels at top of plot
        ax.text(label_x, 0.97, f"$w_R$={wr}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, alpha=0.7)

    # Annotate thermal neutrons
    if n_thermal > 0:
        total = len(energies_mev)
        ax.text(0.02, 0.02,
                f"Thermal neutrons (< 1 eV): {n_thermal} "
                f"({n_thermal/total*100:.1f}% of total)\n"
                f"Negligible dose contribution — grouped below plot range",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Extension 1: Radial dose profile
# ------------------------------------------------------------------
def load_topas_2d_dose(csv_path, nx, ny):
    """Load a TOPAS CSV dose scorer with NxNx1 binning (same format as fluence)."""
    return load_topas_2d_fluence(csv_path, nx, ny)


def compute_radial_profile(x, y, dose_2d, n_radial_bins=50):
    """Average a 2D XY dose map into radial bins from the beam axis.

    Returns r_centres (cm), mean dose in each annular bin.
    """
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    r_max = min(PHANTOM_HLX, PHANTOM_HLY)
    r_edges = np.linspace(0, r_max, n_radial_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    profile = np.zeros(n_radial_bins)
    for i in range(n_radial_bins):
        mask = (rr >= r_edges[i]) & (rr < r_edges[i + 1])
        vals = dose_2d[mask]
        profile[i] = np.mean(vals) if len(vals) > 0 else 0.0

    return r_centres, profile


def plot_radial_dose(r, total_profile, neutron_profile, beam_radius, filepath):
    """Plot radial dose profile showing neutron dose extending beyond the beam."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(r, total_profile, "b-", linewidth=2.0, label="Total dose")
    ax.plot(r, neutron_profile, "r-", linewidth=1.5, label="Neutron dose")

    ax.axvline(beam_radius, color="green", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Beam edge ({2*beam_radius:.0f} cm diameter)")

    ax.set_xlabel("Radial distance from beam axis (cm)", fontsize=14)
    ax.set_ylabel("Dose (Gy)", fontsize=14)
    ax.set_title("Radial Dose Profile — Total vs Neutron\n"
                 "(Normalised to 1 Gy mean phantom dose)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Extension 2: Angular distribution of incident neutrons
# ------------------------------------------------------------------
def load_neutron_phasespace(phsp_path):
    """Load full neutron phase space: energies, positions, directions.

    Returns dict with arrays, or None if no data.
    Only includes +Z directed neutrons (heading toward phantom).
    """
    records = []
    if not os.path.isfile(phsp_path):
        return None

    with open(phsp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                cos_x = float(parts[3])
                cos_y = float(parts[4])
                energy = float(parts[5])
            except (ValueError, IndexError):
                continue
            if energy > 0:  # +Z direction
                cos_z_sq = max(0.0, 1.0 - cos_x**2 - cos_y**2)
                records.append((energy, x, y, cos_x, cos_y, np.sqrt(cos_z_sq)))

    if not records:
        return None

    arr = np.array(records)
    return {
        "energy": arr[:, 0],
        "x": arr[:, 1],
        "y": arr[:, 2],
        "cos_x": arr[:, 3],
        "cos_y": arr[:, 4],
        "cos_z": arr[:, 5],
    }


def plot_angular_distribution(ps_data, filepath):
    """Plot polar angle distribution of neutrons incident on the phantom."""
    cos_z = np.clip(ps_data["cos_z"], 0.0, 1.0)
    theta = np.degrees(np.arccos(cos_z))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.hist(theta, bins=60, range=(0, 90), color="steelblue", edgecolor="navy",
            alpha=0.8, linewidth=0.5)
    ax.set_xlabel("Polar angle from beam axis (degrees)", fontsize=14)
    ax.set_ylabel("Neutron count", fontsize=14)
    ax.set_title("Angular Distribution of Neutrons Incident on Phantom", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    median_theta = np.median(theta)
    mean_theta = np.mean(theta)
    ax.axvline(median_theta, color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {median_theta:.1f}°")
    ax.axvline(mean_theta, color="orange", linestyle=":", linewidth=1.5,
               label=f"Mean = {mean_theta:.1f}°")
    ax.legend(fontsize=12, frameon=True)

    print(f"  Neutron angular distribution: mean = {mean_theta:.1f}°, "
          f"median = {median_theta:.1f}°")

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Extension 3: Collimator thickness comparison
# ------------------------------------------------------------------
def analyse_thickness_variant(label, dose_csv, neutron_csv, ps_phsp, ps_hdr):
    """Compute key metrics for one collimator thickness.

    Returns dict with neutron_fraction, avg_rbe, n_neutrons, scale, or None.
    """
    if not os.path.isfile(dose_csv):
        return None

    _, total_raw = load_topas_1d_dose(dose_csv, PHANTOM_NZ)
    _, neutron_raw = load_topas_1d_dose(neutron_csv, PHANTOM_NZ)

    mean_total = np.mean(total_raw)
    if mean_total <= 0:
        return None

    scale = 1.0 / mean_total
    nfrac = np.sum(neutron_raw) / np.sum(total_raw)

    energies = load_neutron_energies(ps_phsp, ps_hdr)
    avg_rbe, _ = compute_icrp60_rbe(energies)

    return {
        "label": label,
        "neutron_fraction": nfrac,
        "avg_rbe": avg_rbe,
        "bio_fraction": nfrac * avg_rbe,
        "n_neutrons": len(energies),
        "n_per_gy": len(energies) * scale,
        "scale": scale,
    }


def print_thickness_comparison(results):
    """Print side-by-side comparison table for collimator thickness study."""
    print(f"\n{'='*72}")
    print("COLLIMATOR THICKNESS SENSITIVITY STUDY")
    print(f"{'='*72}")
    print(f"  {'Metric':<40s}", end="")
    for r in results:
        print(f"  {r['label']:>12s}", end="")
    print()
    print(f"  {'-'*68}")

    rows = [
        ("Neutron yield per Gy",          "n_per_gy",          ",.0f"),
        ("Neutron physical dose fraction", "neutron_fraction",  ".4f"),
        ("Average neutron RBE (ICRP 60)",  "avg_rbe",           ".2f"),
        ("RBE-weighted biological fraction","bio_fraction",      ".4f"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:<40s}", end="")
        for r in results:
            val = r[key]
            print(f"  {val:>12{fmt}}", end="")
        print()
    print()


def plot_thickness_comparison(results, filepath):
    """Bar chart comparing key metrics between collimator thicknesses."""
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalise each metric to the baseline (first result) for comparison
    baseline = results[0]
    metrics = [
        ("Neutron yield / Gy", "n_per_gy", "steelblue"),
        ("Physical dose frac", "neutron_fraction", "coral"),
        ("RBE-weighted frac", "bio_fraction", "goldenrod"),
    ]

    for i, (mlabel, key, color) in enumerate(metrics):
        base_val = baseline[key]
        if base_val > 0:
            vals = [r[key] / base_val for r in results]
        else:
            vals = [0] * len(results)
        ax.bar(x + (i - 1) * width, vals, width, label=mlabel, color=color,
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Collimator thickness", fontsize=14)
    ax.set_ylabel("Relative to 5 cm collimator", fontsize=14)
    ax.set_title("Collimator Thickness Sensitivity — Neutron Metrics", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ----
    print("Loading TOPAS outputs...")

    if not os.path.isfile(TOTAL_DOSE_CSV):
        print(f"Total dose file not found: {TOTAL_DOSE_CSV}")
        print("Run the TOPAS simulation first:  topas neutron_collimator.txt")
        return

    depth, total_dose_raw = load_topas_1d_dose(TOTAL_DOSE_CSV, PHANTOM_NZ)
    _, neutron_dose_raw = load_topas_1d_dose(NEUTRON_DOSE_CSV, PHANTOM_NZ)

    # ---- Normalise to 1 Gy mean dose ----
    # "1 Gy delivered to the phantom" = 1 Gy mean dose across the full phantom.
    mean_total = np.mean(total_dose_raw)
    if mean_total <= 0:
        print("Error: total dose is zero. Check simulation output.")
        return

    scale = 1.0 / mean_total  # Gy / raw_unit → gives 1 Gy mean
    total_dose = total_dose_raw * scale
    neutron_dose = neutron_dose_raw * scale

    neutron_fraction = np.sum(neutron_dose_raw) / np.sum(total_dose_raw)

    print(f"\n  Raw mean total dose per history: {mean_total:.4e} Gy")
    print(f"  Scale factor for 1 Gy normalisation: {scale:.4e}")

    # ---- Beam profile ----
    if os.path.isfile(BEAM_PROFILE_CSV):
        print("\nLoading beam profile...")
        x, y, fluence = load_topas_2d_fluence(BEAM_PROFILE_CSV, 200, 200)
        plot_beam_profile(x, y, fluence,
                          os.path.join(OUTPUT_DIR, "beam_profile.png"))
    else:
        print(f"Beam profile not found: {BEAM_PROFILE_CSV}")

    # ---- Depth-dose plot ----
    print("\nPlotting depth-dose profiles...")
    plot_depth_dose(depth, total_dose, neutron_dose,
                    os.path.join(OUTPUT_DIR, "depth_dose.png"))

    # ---- Neutron spectrum + ICRP RBE ----
    n_total = 0
    avg_rbe = 0.0
    if os.path.isfile(NEUTRON_PS_PHSP):
        print("\nLoading neutron phase space...")
        energies = load_neutron_energies(NEUTRON_PS_PHSP, NEUTRON_PS_HDR)
        n_total = len(energies)

        # Yield: report per Gy of mean phantom dose (our normalisation convention).
        # This is specific to this geometry/beam — not a universal physical constant.
        n_per_gy = n_total * scale
        print(f"  Neutrons incident on phantom (+Z direction): {n_total}")
        print(f"  Neutrons incident on phantom per Gy "
              f"(mean phantom dose normalisation): {n_per_gy:.0f}")

        plot_neutron_spectrum(energies,
                              os.path.join(OUTPUT_DIR, "neutron_spectrum.png"))

        # ICRP 60 RBE
        avg_rbe, bin_info = compute_icrp60_rbe(energies)

        print(f"\n{'='*65}")
        print("ICRP 60 NEUTRON RBE ANALYSIS")
        print(f"{'='*65}")
        print(f"  {'Energy range':<22s} {'wR':>5s} {'Count':>10s} {'Fraction':>10s}")
        print(f"  {'-'*50}")
        for e_lo, e_hi, wr, count, frac in bin_info:
            if np.isinf(e_hi):
                label = f">= {e_lo*1000:.0f} keV" if e_lo < 1 else f">= {e_lo:.0f} MeV"
            elif e_hi <= 0.01:
                label = f"< {e_hi*1000:.0f} keV"
            elif e_lo < 1:
                label = f"{e_lo*1000:.0f} keV – {e_hi*1000:.0f} keV" if e_hi < 1 \
                    else f"{e_lo*1000:.0f} keV – {e_hi:.0f} MeV"
            else:
                label = f"{e_lo:.0f} MeV – {e_hi:.0f} MeV"
            print(f"  {label:<22s} {wr:>5d} {count:>10d} {frac:>10.4f}")

        print(f"\n  Fluence-weighted average RBE: {avg_rbe:.2f}")
    else:
        print(f"\nNeutron phase space not found: {NEUTRON_PS_PHSP}")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Neutron physical dose fraction:  {neutron_fraction*100:.2f}%")
    print(f"  Average neutron RBE (ICRP 60):   {avg_rbe:.2f}")
    if avg_rbe > 0:
        bio_frac = neutron_fraction * avg_rbe
        print(f"  RBE-weighted biological contribution: {bio_frac*100:.2f}%")
        print()
        print(f"  INTERPRETATION:")
        print(f"    The neutron physical dose is a small fraction of the total")
        print(f"    (~{neutron_fraction*100:.1f}%), but the biological impact is")
        print(f"    substantially larger (~{bio_frac*100:.1f}%) because neutrons")
        print(f"    have an elevated RBE (average wR ~ {avg_rbe:.1f}).  This means")
        print(f"    each Gray of neutron dose is ~{avg_rbe:.0f}x more biologically")
        print(f"    damaging than the same dose from the primary proton beam.")
        print(f"    In passive-scattering proton therapy, this collimator-generated")
        print(f"    neutron dose — though physically small — is a clinically")
        print(f"    relevant concern for secondary cancer risk.")
    print()

    # ================================================================
    # Extension 1: Radial dose profile
    # ================================================================
    if os.path.isfile(TOTAL_DOSE_XY_CSV) and os.path.isfile(NEUTRON_DOSE_XY_CSV):
        print("\n--- Extension 1: Radial dose profile ---")
        x_xy, y_xy, total_2d = load_topas_2d_dose(TOTAL_DOSE_XY_CSV,
                                                    PHANTOM_NXY, PHANTOM_NXY)
        _, _, neutron_2d = load_topas_2d_dose(NEUTRON_DOSE_XY_CSV,
                                               PHANTOM_NXY, PHANTOM_NXY)

        # Apply same 1 Gy normalisation
        total_2d_norm = total_2d * scale
        neutron_2d_norm = neutron_2d * scale

        r, total_radial = compute_radial_profile(x_xy, y_xy, total_2d_norm)
        _, neutron_radial = compute_radial_profile(x_xy, y_xy, neutron_2d_norm)

        # Estimate beam radius from the 2D fluence map (or use 4 cm fallback)
        beam_radius = 4.0  # cm (nominal from collimator design)
        if os.path.isfile(BEAM_PROFILE_CSV):
            xbp, ybp, flbp = load_topas_2d_fluence(BEAM_PROFILE_CSV, 200, 200)
            d, _ = compute_beam_diameter_2d(xbp, ybp, flbp)
            beam_radius = d / 2.0

        plot_radial_dose(r, total_radial, neutron_radial, beam_radius,
                         os.path.join(OUTPUT_DIR, "radial_dose.png"))

        # Report fraction of neutron dose outside beam field
        outside = r > beam_radius
        if np.any(outside) and np.sum(neutron_radial) > 0:
            frac_outside = np.sum(neutron_radial[outside]) / np.sum(neutron_radial)
            print(f"  Neutron dose outside beam field (r > {beam_radius:.1f} cm): "
                  f"{frac_outside*100:.1f}% of total neutron dose")
    else:
        print("\n  Radial dose XY files not found — skipping Extension 1.")
        print(f"  (Need {TOTAL_DOSE_XY_CSV} and {NEUTRON_DOSE_XY_CSV})")

    # ================================================================
    # Extension 2: Angular distribution
    # ================================================================
    if os.path.isfile(NEUTRON_PS_PHSP):
        print("\n--- Extension 2: Neutron angular distribution ---")
        ps_data = load_neutron_phasespace(NEUTRON_PS_PHSP)
        if ps_data is not None:
            plot_angular_distribution(
                ps_data, os.path.join(OUTPUT_DIR, "neutron_angular.png"))
        else:
            print("  No phase space data loaded.")

    # ================================================================
    # Extension 3: Collimator thickness comparison
    # ================================================================
    print("\n--- Extension 3: Collimator thickness sensitivity ---")
    baseline_metrics = {
        "label": "5 cm",
        "neutron_fraction": neutron_fraction,
        "avg_rbe": avg_rbe,
        "bio_fraction": neutron_fraction * avg_rbe,
        "n_neutrons": n_total if os.path.isfile(NEUTRON_PS_PHSP) else 0,
        "n_per_gy": (n_total * scale) if os.path.isfile(NEUTRON_PS_PHSP) else 0,
        "scale": scale,
    }

    thin_metrics = analyse_thickness_variant(
        "3 cm", TOTAL_DOSE_CSV_3CM, NEUTRON_DOSE_CSV_3CM,
        NEUTRON_PS_PHSP_3CM, NEUTRON_PS_HDR_3CM,
    )

    if thin_metrics is not None:
        comparison = [baseline_metrics, thin_metrics]
        print_thickness_comparison(comparison)
        plot_thickness_comparison(
            comparison, os.path.join(OUTPUT_DIR, "thickness_comparison.png"))
    else:
        print("  3 cm collimator data not found — run neutron_collimator_3cm.txt first.")
        print(f"  (Need {TOTAL_DOSE_CSV_3CM})")


if __name__ == "__main__":
    main()
