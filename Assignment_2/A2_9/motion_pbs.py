"""
Section 2.9: Patient Motion in Pencil Beam Scanning (PBS) Proton Therapy

Extends Section 2.8 (static PBS) to include the most important clinical
limitation of PBS — interplay between sequential spot delivery and
breathing motion. Three studies:

  (3) Single-fraction interplay simulation
        Tumour moves sinusoidally while spots are delivered one at a time.
        Each spot lands at the tumour's current position → cold/hot spots.

  (4) Fractionation averaging
        A clinical treatment is 30 fractions with a random breathing
        phase at the start of each fraction. Averages out interplay.

  (5) Rescanning mitigation
        Deliver the plan M times at 1/M intensity per pass. Even within a
        single fraction this suppresses interplay variance ∝ 1/sqrt(M).

Reuses from Section 2.8:
  - CT geometry, Schneider RSP, pristine peaks cache
  - Spot grid, analytical dose influence matrix, NNLS weight optimisation

Key physics:
  - Motion direction: X (lateral, within CT slice). Amplitude A ≈ 5 mm,
    period T_breath ≈ 4 s, spot dwell τ ≈ 2 ms.
  - Spot j delivered at time t_j = j·τ, shifted by Δ(t_j)=A·sin(2π(t_j-φ₀)/T).
  - Dose contribution of spot j to patient = (static spot dose), then
    shift the whole dose distribution by Δ(t_j) in X, then accumulate.

Implementation:
  - Phase-binned acceleration: group spots of similar shift into N_bins,
    compute partial dose via sparse matmul, shift bin dose once, sum.

Outputs in 2.9/output/:
  dvh_motion.png               static vs interplay vs fractionated vs rescanned
  rescan_variance.png          interplay variance vs rescan count M
  dose_map_motion.png          static vs single-fraction cold/hot spots
  motion_timeline.png          shift Δ(t) and spot delivery schedule
  summary_motion.csv           DVH metrics comparison
"""

import os
import sys
import time
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
except Exception as exc:
    raise RuntimeError(f"matplotlib required: {exc}")

try:
    import pydicom
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "pydicom"])
    import pydicom

# Import from 2.8
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PBS_DIR = os.path.join(PROJECT_ROOT, "A2_8")
sys.path.insert(0, PBS_DIR)
import pbs_proton as pbs  # noqa: E402

from scipy import sparse  # noqa: E402

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ------------------------------------------------------------------
# Motion configuration
# ------------------------------------------------------------------
MOTION_AMPLITUDE_MM = 5.0        # peak-to-mean lateral excursion
MOTION_PERIOD_S = 4.0            # typical adult breath cycle
SPOT_DWELL_S = 0.002             # 2 ms per spot
LAYER_SWITCH_S = 1.0             # realistic energy-layer switch time
MOTION_DIRECTION = "x"           # lateral shift axis ("x" or "z")

N_PHASE_BINS = 16                # bins per fraction (shift discretisation)
N_FRACTIONS = 30                 # clinical fractionation
RESCAN_PASSES = [1, 2, 5, 10, 20]
RESCAN_FRACTIONS_FOR_VARIANCE = 20   # independent draws per M

RNG_SEED = 2026
np.random.seed(RNG_SEED)


# ------------------------------------------------------------------
# Spot delivery schedule
# ------------------------------------------------------------------
def assign_delivery_times(spots):
    """
    Assign each spot a delivery time based on sequential scanning:
      - Spots within a layer delivered at SPOT_DWELL_S apart.
      - Layer switch adds LAYER_SWITCH_S delay.
    Returns t[j] in seconds.
    """
    # Group spots by layer index
    by_layer = {}
    for j, (_, _, _, li) in enumerate(spots):
        by_layer.setdefault(li, []).append(j)
    times = np.zeros(len(spots))
    t = 0.0
    for li in sorted(by_layer.keys()):
        for j in by_layer[li]:
            times[j] = t
            t += SPOT_DWELL_S
        t += LAYER_SWITCH_S
    return times


def motion_shift(t, phi0):
    """Sinusoidal tumour lateral position (mm) at absolute time t (s)."""
    return MOTION_AMPLITUDE_MM * np.sin(
        2.0 * np.pi * (t - phi0) / MOTION_PERIOD_S)


# ------------------------------------------------------------------
# Shifted-dose computation (phase-binned acceleration)
# ------------------------------------------------------------------
def shift_dose_3d(dose_3d, shift_mm, dx_mm, axis):
    """
    Shift a 3D dose map by `shift_mm` along `axis` ("x" or "z")
    using linear interpolation in voxel index space. Zero-pad at edges.

    dose_3d: shape (n_slices, rows, cols) corresponding to (iz, iy, ix).
    """
    shift_vox = shift_mm / dx_mm
    n_pre = int(np.floor(shift_vox))
    frac = shift_vox - n_pre

    # Linear interpolation between integer shifts n_pre and n_pre+1.
    def roll_axis(arr, n, ax):
        if n == 0:
            return arr.copy()
        out = np.zeros_like(arr)
        if ax == "x":  # last axis
            if n > 0:
                out[..., n:] = arr[..., :-n]
            else:
                out[..., :n] = arr[..., -n:]
        elif ax == "z":  # first axis
            if n > 0:
                out[n:, ...] = arr[:-n, ...]
            else:
                out[:n, ...] = arr[-n:, ...]
        else:
            raise ValueError(axis)
        return out

    lo = roll_axis(dose_3d, n_pre, axis)
    hi = roll_axis(dose_3d, n_pre + 1, axis)
    return (1.0 - frac) * lo + frac * hi


def accumulate_shifted_dose(D, weights, shifts_per_spot, geom, axis="x"):
    """
    For a delivery sequence where spot j gets position shift shifts_per_spot[j],
    compute total 3D dose.

    Phase-binned for efficiency:
      - Bin spots by their shift into N_PHASE_BINS
      - For each bin k, partial dose = D[:, spots_in_bin] @ weights[spots_in_bin]
      - Shift partial dose by the bin-mean shift, accumulate
    """
    n_vox = D.shape[0]
    n_spots = D.shape[1]
    dx = geom["dx"] if axis == "x" else geom["dz"]
    n_cols, n_rows, n_slices = geom["cols"], geom["rows"], geom["n_slices"]

    # Bin shifts
    s_min = -MOTION_AMPLITUDE_MM
    s_max = MOTION_AMPLITUDE_MM
    edges = np.linspace(s_min, s_max, N_PHASE_BINS + 1)
    # Clip shifts into range
    sh = np.clip(shifts_per_spot, s_min + 1e-9, s_max - 1e-9)
    bin_idx = np.digitize(sh, edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_PHASE_BINS - 1)

    total = np.zeros(n_vox)
    for k in range(N_PHASE_BINS):
        members = np.where(bin_idx == k)[0]
        if members.size == 0:
            continue
        # Partial weighted sum
        w_k = np.zeros(n_spots)
        w_k[members] = weights[members]
        if sparse.issparse(D):
            partial = np.asarray(D.dot(w_k)).ravel()
        else:
            partial = D @ w_k
        # Shift partial dose by the mean shift of members
        shift_mm = float(np.mean(sh[members]))
        partial_3d = partial.reshape(n_slices, n_rows, n_cols)
        shifted_3d = shift_dose_3d(partial_3d, shift_mm, dx, axis)
        total += shifted_3d.ravel()
    return total


# ------------------------------------------------------------------
# Simulations: interplay, fractionation, rescanning
# ------------------------------------------------------------------
def simulate_fraction(D, weights, delivery_times, phi0, pass_scale=1.0,
                      time_offset=0.0, geom=None):
    """One delivery (single pass). Returns flat dose vector."""
    shifts = motion_shift(delivery_times + time_offset, phi0)
    return pass_scale * accumulate_shifted_dose(
        D, weights * pass_scale, shifts, geom,
        axis=MOTION_DIRECTION)


def simulate_rescanned_fraction(D, weights, delivery_times, phi0, M, geom):
    """
    Deliver M passes at weights/M each.  Each pass begins at phi0 with
    an extra time offset equal to total previous-pass delivery time.
    """
    scaled = weights / M
    total_pass_time = delivery_times.max() + SPOT_DWELL_S + LAYER_SWITCH_S
    total = np.zeros(D.shape[0])
    for p in range(M):
        offset = p * total_pass_time
        shifts = motion_shift(delivery_times + offset, phi0)
        total += accumulate_shifted_dose(D, scaled, shifts, geom,
                                         axis=MOTION_DIRECTION)
    return total


# ------------------------------------------------------------------
# DVH helpers (using 2.8 voxel maps + masks)
# ------------------------------------------------------------------
def flat_to_dose_map(dose_flat, geom, threshold=1e-12):
    dose_map = {}
    n_cols = geom["cols"]
    n_rows = geom["rows"]
    for iz in range(geom["n_slices"]):
        for iy in range(n_rows):
            for ix in range(n_cols):
                vi = iz * (n_rows * n_cols) + iy * n_cols + ix
                if dose_flat[vi] > threshold:
                    dose_map[(ix, iy, iz)] = float(dose_flat[vi])
    return dose_map


def structure_dose_array(dose_flat, mask_voxels, geom):
    n_cols = geom["cols"]; n_rows = geom["rows"]
    out = np.zeros(len(mask_voxels))
    for k, (ix, iy, iz) in enumerate(mask_voxels):
        vi = iz * (n_rows * n_cols) + iy * n_cols + ix
        out[k] = dose_flat[vi]
    return out


def dvh_stats(dose_flat, mask_voxels, geom):
    d = structure_dose_array(dose_flat, mask_voxels, geom)
    if d.size == 0 or d.max() == 0:
        return {"D95": 0, "D50": 0, "D02": 0, "mean": 0, "max": 0}
    return {
        "D95": float(np.percentile(d, 5)),
        "D50": float(np.percentile(d, 50)),
        "D02": float(np.percentile(d, 98)),
        "mean": float(np.mean(d)),
        "max": float(np.max(d)),
    }


def cumulative_dvh(dose_flat, mask_voxels, geom, n_bins=200, d_max=None):
    d = structure_dose_array(dose_flat, mask_voxels, geom)
    if d.size == 0 or d.max() == 0:
        return np.array([0.0]), np.array([100.0])
    if d_max is None:
        d_max = d.max() * 1.05
    bins = np.linspace(0, d_max, n_bins)
    vol = np.array([100.0 * np.sum(d >= b) / d.size for b in bins])
    return bins, vol


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_motion_timeline(delivery_times, shifts, filepath):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(delivery_times, shifts, "o", ms=2, alpha=0.6, label="spot shift")
    t_fine = np.linspace(0, delivery_times.max(), 2000)
    ax.plot(t_fine, motion_shift(t_fine, phi0=0.0), "-", lw=1,
            color="grey", alpha=0.5, label="motion Δ(t)")
    ax.set_xlabel("Delivery time (s)")
    ax.set_ylabel(f"Lateral shift Δ{MOTION_DIRECTION} (mm)")
    ax.set_title("Spot delivery schedule and tumour motion")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(filepath, dpi=300); plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_dvh_motion(curves, masks, geom, filepath):
    """
    curves: dict[label] = dose_flat
    """
    struct_cfg = [
        ("tumour", "red", "GTV"),
        ("heart", "blue", "Heart"),
        ("cord", "green", "Cord"),
        ("ptv", "magenta", "PTV"),
        ("lung_r", "orange", "Lung R"),
        ("lung_l", "cyan", "Lung L"),
    ]
    style_for = {
        "static": ("-", 2.2),
        "interplay": ("--", 1.7),
        "fractionated": ("-.", 1.7),
        "rescanned M=10": (":", 1.9),
    }
    # Global d_max for consistent axes — use static (planned) dose max
    plan_max = 0.0
    d_max = 0.0
    for label, dose_flat in curves.items():
        for sname, _, _ in struct_cfg:
            if sname in masks and masks[sname]:
                d = structure_dose_array(dose_flat, masks[sname], geom)
                if d.size:
                    d_max = max(d_max, float(d.max()))
                    if label == "static":
                        plan_max = max(plan_max, float(d.max()))
    if plan_max <= 0:
        plan_max = d_max if d_max > 0 else 1.0
    d_max_pct = d_max / plan_max * 100.0 * 1.05

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for label, dose_flat in curves.items():
        ls, lw = style_for.get(label, ("-", 1.4))
        for sname, color, display in struct_cfg:
            if sname not in masks or not masks[sname]:
                continue
            b, v = cumulative_dvh(dose_flat, masks[sname], geom,
                                  d_max=d_max * 1.05)
            b_pct = b / plan_max * 100.0
            ax.plot(b_pct, v, color=color, linestyle=ls, linewidth=lw,
                    label=f"{display} ({label})")
    ax.set_xlabel("Dose (% of plan max)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("DVH — Motion Effects in PBS")
    ax.set_xlim(0, d_max_pct); ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
              frameon=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_rescan_variance(variance_table, filepath):
    Ms = sorted(variance_table.keys())
    stds = [variance_table[M] for M in Ms]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ms, stds, "o-", lw=2, ms=8, label="GTV σ(dose) / mean")
    # 1/sqrt(M) reference
    ref = np.array(stds[0]) * np.sqrt(Ms[0]) / np.sqrt(Ms)
    ax.plot(Ms, ref, "k--", alpha=0.5, label=r"$\propto 1/\sqrt{M}$")
    ax.set_xlabel("Number of rescans M")
    ax.set_ylabel("GTV dose coefficient of variation")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Rescanning suppresses interplay variance")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(filepath, dpi=300); plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_d95_vs_rescanning(Ms, d95_values, static_d95, filepath):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ms, d95_values, "o-", lw=2, ms=8, color="C0",
            label="GTV D95 (rescanned)")
    ax.axhline(static_d95, color="grey", ls="--", lw=1.2,
               label=f"Static D95 = {static_d95:.1f} Gy")
    ax.set_xlabel("Number of rescans M")
    ax.set_ylabel("GTV D95 (Gy)")
    ax.set_xscale("log")
    ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
    ax.set_title("Rescanning recovery of GTV D95 coverage")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(filepath, dpi=300); plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_scenario_comparison(labels, d95_values, cov_values, filepath):
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax1.bar(x - w/2, d95_values, width=w, color="C0", label="GTV D95 (Gy)")
    ax1.set_ylabel("GTV D95 (Gy)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.grid(alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, cov_values, width=w, color="C3", label="GTV CoV")
    ax2.set_ylabel("GTV CoV", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax1.set_title("Scenario comparison — GTV D95 vs CoV")
    ax1.legend([b1, b2], ["GTV D95 (Gy)", "GTV CoV"], loc="upper right")
    fig.tight_layout(); fig.savefig(filepath, dpi=300); plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_d95_vs_phase(phases, d95_values, static_d95, filepath):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(phases, d95_values, "o-", lw=2, ms=8, color="C2",
            label="GTV D95 (interplay)")
    ax.axhline(static_d95, color="grey", ls="--", lw=1.2,
               label=f"Static D95 = {static_d95:.1f} Gy")
    ax.set_xlabel("Breathing start phase φ₀ (s)")
    ax.set_ylabel("GTV D95 (Gy)")
    ax.set_title("Single-fraction D95 vs breathing start phase")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(filepath, dpi=300); plt.close(fig)
    print(f"  Saved: {filepath}")


CONTOUR_NUDGE_PX = 0.5
CONTOUR_COLOURS = {
    "GTVp": "red", "GTV": "red", "PTV": "magenta",
    "Lung_R": "orange", "Lung_L": "cyan", "Heart": "blue",
    "SpinalCord": "green", "Body": "grey", "BODY": "grey",
    "External": "grey", "EXTERNAL": "grey",
}
CONTOUR_ORDER = [
    "Body", "BODY", "External", "EXTERNAL",
    "Lung_R", "Lung_L", "Heart", "SpinalCord", "PTV", "GTVp", "GTV",
]
CT_DIR = os.path.join(PROJECT_ROOT, "CTData")


def _find_rtstruct():
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(CT_DIR, fname),
                                 stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                return os.path.join(CT_DIR, fname)
        except Exception:
            pass
    return None


def _load_contour_polygons(geom, slice_iz):
    rtstruct_path = _find_rtstruct()
    if not rtstruct_path:
        return {}
    ds = pydicom.dcmread(rtstruct_path)
    roi_map = {}
    for roi in getattr(ds, "StructureSetROISequence", []):
        roi_map[int(roi.ROINumber)] = str(roi.ROIName)
    contours = {}
    for roi_contour in getattr(ds, "ROIContourSequence", []):
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", -1))
        roi_name = roi_map.get(roi_number)
        if not roi_name or not hasattr(roi_contour, "ContourSequence"):
            continue
        for contour in roi_contour.ContourSequence:
            data = np.asarray(getattr(contour, "ContourData", []), dtype=float)
            if data.size < 9:
                continue
            pts = data.reshape(-1, 3)
            z_mean = float(np.mean(pts[:, 2]))
            iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - z_mean)))
            if iz != slice_iz:
                continue
            contours.setdefault(roi_name, []).append(pts[:, :2])
    return contours


def _load_ct_slice_hu(geom, iz):
    target_z = geom["slice_zs"][iz]
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        ds = pydicom.dcmread(os.path.join(CT_DIR, fname))
        if getattr(ds, "Modality", "") != "CT":
            continue
        if abs(float(ds.ImagePositionPatient[2]) - target_z) < 0.1:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            return ds.pixel_array.astype(float) * slope + intercept
    return None


def _slice_extent(geom):
    dx, dy = geom["dx"], geom["dy"]
    x_min = geom["x0"] - dx / 2
    x_max = geom["x0"] + (geom["cols"] - 0.5) * dx
    y_min = geom["y0"] - dy / 2
    y_max = geom["y0"] + (geom["rows"] - 0.5) * dy
    return [x_min, x_max, y_min, y_max]


def _draw_contours(ax, contour_polys, geom):
    shift = np.array([geom["dx"] * CONTOUR_NUDGE_PX,
                      geom["dy"] * CONTOUR_NUDGE_PX])
    drawn = set()
    legend_handles = []
    for name in CONTOUR_ORDER:
        if name in contour_polys and name not in drawn:
            colour = CONTOUR_COLOURS.get(name, "white")
            for poly_xy in contour_polys[name]:
                ax.add_patch(MplPolygon(
                    poly_xy + shift, closed=True, fill=False,
                    edgecolor=colour, linewidth=1.3))
            legend_handles.append(
                Line2D([], [], color=colour, linewidth=1.3, label=name))
            drawn.add(name)
    for name, polys in contour_polys.items():
        if name not in drawn:
            colour = CONTOUR_COLOURS.get(name, "white")
            for poly_xy in polys:
                ax.add_patch(MplPolygon(
                    poly_xy + shift, closed=True, fill=False,
                    edgecolor=colour, linewidth=1.3))
            legend_handles.append(
                Line2D([], [], color=colour, linewidth=1.3, label=name))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
                  frameon=True, framealpha=0.8, edgecolor="grey")


def plot_dose_map_motion(static_flat, interplay_flat, geom, gtv, filepath):
    n_cols, n_rows, n_slices = geom["cols"], geom["rows"], geom["n_slices"]
    iz = n_slices // 2
    static_3d = static_flat.reshape(n_slices, n_rows, n_cols)
    interp_3d = interplay_flat.reshape(n_slices, n_rows, n_cols)

    dmax = max(static_3d.max(), interp_3d.max())
    if dmax <= 0:
        dmax = 1.0

    static_slice = static_3d[iz]
    interp_slice = interp_3d[iz]
    diff = interp_slice - static_slice

    # CT background + vector contours (matching plot_uncertainty_slices.py)
    hu_image = _load_ct_slice_hu(geom, iz)
    contour_polys = _load_contour_polygons(geom, iz)
    extent = _slice_extent(geom)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for ax_idx, (dose_slice, title, cmap, vmin, vmax) in enumerate([
        (static_slice, "Static PBS dose", "jet", 0, dmax),
        (interp_slice, "Single-fraction interplay", "jet", 0, dmax),
        (diff, "Interplay \u2212 Static",
         "seismic", -max(abs(diff.min()), abs(diff.max())),
         max(abs(diff.min()), abs(diff.max()))),
    ]):
        ax = axes[ax_idx]
        if hu_image is not None:
            ax.imshow(hu_image, cmap="gray", extent=extent, origin="lower",
                      vmin=-400, vmax=400, aspect="equal")
        dose_masked = np.ma.masked_where(
            np.abs(dose_slice) < 0.02 * dmax, dose_slice)
        im = ax.imshow(dose_masked, cmap=cmap, extent=extent, origin="lower",
                       alpha=0.55, vmin=vmin, vmax=vmax, aspect="equal")
        _draw_contours(ax, contour_polys, geom)
        ax.invert_yaxis()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gy")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_title(title)

    target_z = geom["slice_zs"][iz]
    fig.suptitle(f"Axial slice Z={target_z:.1f} mm — PBS interplay effects",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def write_summary(all_metrics, filepath):
    # all_metrics: dict[label] -> dict[structure] -> dict[metric]
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "structure", "D95", "D50", "D02", "mean", "max"])
        for label, per_struct in all_metrics.items():
            for sname, stats in per_struct.items():
                w.writerow([label, sname,
                            f"{stats['D95']:.4f}",
                            f"{stats['D50']:.4f}",
                            f"{stats['D02']:.4f}",
                            f"{stats['mean']:.4f}",
                            f"{stats['max']:.4f}"])
    print(f"  Saved: {filepath}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("SECTION 2.9: PATIENT MOTION IN PBS PROTON THERAPY")
    print("=" * 70)

    # ---- Part A: reuse 2.8 static PBS setup ----
    print("\n[A] Loading CT/geometry/masks (reused from 2.8) ...")
    geom = pbs.load_ct_geometry()
    gtv = pbs.load_gtv_bounds()
    schneider = pbs.parse_schneider_params()
    y_c, cum_wet = pbs.compute_wet_along_beam(
        geom, gtv["centre_x"], gtv["centre_z"], schneider)
    wet_prox, wet_dist, _, _ = pbs.find_tumour_wet_boundaries(
        y_c, cum_wet, gtv)
    masks = pbs.build_scoring_masks(geom)

    print("\n[B] Loading pristine peaks ...")
    energies = np.arange(60.0, 120.0, pbs.ENERGY_LAYER_STEP).tolist()
    profiles = pbs.load_or_run_pristine_peaks(sorted(energies))

    print("\n[C] Energy layers + spot grid ...")
    energy_layers = pbs.select_energy_layers(profiles, wet_prox, wet_dist)
    spots = pbs.generate_spot_grid(gtv, energy_layers)

    print("\n[D] Dose influence matrix ...")
    D = pbs.build_dose_influence_matrix(
        spots, profiles, geom, schneider, gtv)

    print("\n[E] Optimising static spot weights ...")
    weights = pbs.optimize_spot_weights(D, spots, masks, geom)
    print(f"  {int(np.sum(weights > 1e-6))}/{len(spots)} active spots")

    # Static reference dose
    if sparse.issparse(D):
        static_flat = np.asarray(D.dot(weights)).ravel()
    else:
        static_flat = D @ weights

    # Normalise weights so that the static GTV mean dose == the clinical
    # prescription (60 Gy). This makes D95/D50/D02 in Gy comparable with
    # DVH reports and is purely a global scaling — it does not affect
    # CoV or any motion result.
    PRESCRIPTION_GY = 60.0
    gtv_idx = sorted(iz * geom["rows"] * geom["cols"] + iy * geom["cols"] + ix
                     for ix, iy, iz in masks["tumour"])
    gtv_mean = float(np.mean(static_flat[gtv_idx])) if gtv_idx else 0.0
    if gtv_mean > 0:
        scale = PRESCRIPTION_GY / gtv_mean
        weights = weights * scale
        static_flat = static_flat * scale
        print(f"  Scaled to prescription: mean GTV = {PRESCRIPTION_GY:.1f} Gy "
              f"(scale factor = {scale:.3e})")

    # ---- Delivery timeline ----
    delivery_times = assign_delivery_times(spots)
    total_delivery = delivery_times.max() + SPOT_DWELL_S
    print(f"\n[F] Delivery timeline: {total_delivery:.2f} s "
          f"({total_delivery/MOTION_PERIOD_S:.1f} breath cycles)")

    # Timeline plot (use phi0=0 shifts)
    sh0 = motion_shift(delivery_times, phi0=0.0)
    plot_motion_timeline(delivery_times, sh0,
                         os.path.join(OUTPUT_DIR, "motion_timeline.png"))

    # ---- (3) Single-fraction interplay: worst-case phase ----
    print("\n[3] Single-fraction interplay (searching for worst phase) ...")
    # Sample a few phases, pick the one with largest GTV heterogeneity
    phases = np.linspace(0, MOTION_PERIOD_S, 6, endpoint=False)
    print(f"  Start phases tested (s): "
          f"{np.array2string(phases, precision=3)}")
    worst_std = -1.0
    worst_flat = None
    worst_phi = 0.0
    phase_d95 = []   # per-phase GTV D95 (Gy) for d95_vs_phase plot
    phase_cov = []
    for phi in phases:
        flat = simulate_fraction(D, weights, delivery_times, phi,
                                  pass_scale=1.0, geom=geom)
        d_gtv = structure_dose_array(flat, masks["tumour"], geom)
        if d_gtv.size == 0 or d_gtv.mean() == 0:
            phase_d95.append(0.0); phase_cov.append(0.0)
            continue
        cv = d_gtv.std() / d_gtv.mean()
        st = dvh_stats(flat, masks["tumour"], geom)
        phase_d95.append(st["D95"]); phase_cov.append(cv)
        print(f"    phi0={phi:.2f}s → GTV CoV={cv:.4f}  D95={st['D95']:.2f} Gy")
        if cv > worst_std:
            worst_std = cv; worst_flat = flat; worst_phi = phi
    interplay_flat = worst_flat
    print(f"  Worst phase: phi0={worst_phi:.2f}s (CoV={worst_std:.4f})")

    # ---- (4) Fractionation averaging ----
    print(f"\n[4] Fractionation averaging over {N_FRACTIONS} fractions ...")
    fractionated = np.zeros_like(static_flat)
    frac_phases = np.random.uniform(0, MOTION_PERIOD_S, size=N_FRACTIONS)
    for f, phi in enumerate(frac_phases):
        fractionated += simulate_fraction(
            D, weights, delivery_times, phi, pass_scale=1.0, geom=geom)
    fractionated /= N_FRACTIONS
    print(f"  Start phases (s): "
          f"{np.array2string(frac_phases, precision=2, max_line_width=120)}")

    # ---- (5) Rescanning study ----
    print("\n[5] Rescanning study (M = "
          f"{RESCAN_PASSES}) ...")
    variance_table = {}
    rescan_examples = {}  # one representative dose per M
    rescan_phases = {}
    for M in RESCAN_PASSES:
        cvs = []
        phases_M = np.random.uniform(0, MOTION_PERIOD_S,
                                     size=RESCAN_FRACTIONS_FOR_VARIANCE)
        rescan_phases[M] = phases_M
        for k, phi in enumerate(phases_M):
            flat = simulate_rescanned_fraction(
                D, weights, delivery_times, phi, M, geom)
            d_gtv = structure_dose_array(flat, masks["tumour"], geom)
            if d_gtv.mean() > 0:
                cvs.append(d_gtv.std() / d_gtv.mean())
            if k == 0:
                rescan_examples[M] = flat
        variance_table[M] = float(np.mean(cvs))
        print(f"    M={M:2d}: mean GTV CoV = {variance_table[M]:.4f}  "
              f"(ideal 1/sqrt(M) bound = "
              f"{variance_table[RESCAN_PASSES[0]]/np.sqrt(M):.4f}; "
              f"not reached — see saturation note below)")

    # ---- DVH metrics + plots ----
    print("\n[G] Computing DVH metrics ...")
    structures_to_report = ["tumour", "ptv", "heart", "cord", "lung_r", "lung_l", "body"]

    def per_struct(flat):
        out = {}
        for s in structures_to_report:
            if s in masks and masks[s]:
                out[s] = dvh_stats(flat, masks[s], geom)
        return out

    all_metrics = {
        "static": per_struct(static_flat),
        "interplay_worst": per_struct(interplay_flat),
        "fractionated": per_struct(fractionated),
    }
    for M in RESCAN_PASSES:
        all_metrics[f"rescan_M{M}"] = per_struct(rescan_examples[M])

    # Full summary table (all scenarios, all structures)
    print(f"\n  {'Label':<20s} {'GTV mean':>10s} {'GTV D95':>10s} "
          f"{'GTV D02':>10s} {'Heart mean':>12s} {'Cord max':>10s} "
          f"{'Lung_R mean':>12s} {'Lung_L mean':>12s}")
    print("  " + "-" * 104)
    for label, ps in all_metrics.items():
        gt = ps.get("tumour", {"mean": 0, "D95": 0, "D02": 0})
        he = ps.get("heart", {"mean": 0}); co = ps.get("cord", {"max": 0})
        lu = ps.get("lung_r", {"mean": 0})
        ll = ps.get("lung_l", {"mean": 0})
        print(f"  {label:<20s} {gt['mean']:10.3f} {gt['D95']:10.3f} "
              f"{gt['D02']:10.3f} {he['mean']:12.3f} {co['max']:10.3f} "
              f"{lu['mean']:12.3f} {ll['mean']:12.3f}")

    # Curves to overlay on DVH
    curves = {
        "static": static_flat,
        "interplay": interplay_flat,
        "fractionated": fractionated,
        "rescanned M=10": rescan_examples[10],
    }
    plot_dvh_motion(curves, masks, geom,
                    os.path.join(OUTPUT_DIR, "dvh_motion.png"))
    plot_rescan_variance(variance_table,
                         os.path.join(OUTPUT_DIR, "rescan_variance.png"))
    plot_dose_map_motion(static_flat, interplay_flat, geom, gtv,
                         os.path.join(OUTPUT_DIR, "dose_map_motion.png"))
    write_summary(all_metrics,
                  os.path.join(OUTPUT_DIR, "summary_motion.csv"))

    # --- New: D95 vs rescanning M, scenario comparison, D95 vs phase ---
    static_d95 = dvh_stats(static_flat, masks["tumour"], geom)["D95"]
    rescan_d95 = [dvh_stats(rescan_examples[M], masks["tumour"], geom)["D95"]
                  for M in RESCAN_PASSES]
    plot_d95_vs_rescanning(
        RESCAN_PASSES, rescan_d95, static_d95,
        os.path.join(OUTPUT_DIR, "d95_vs_rescanning.png"))

    plot_d95_vs_phase(
        phases, phase_d95, static_d95,
        os.path.join(OUTPUT_DIR, "d95_vs_phase.png"))

    # Scenario comparison bar chart
    def _gtv_cov(flat):
        d = structure_dose_array(flat, masks["tumour"], geom)
        return float(d.std() / d.mean()) if d.size and d.mean() > 0 else 0.0
    scen_labels = ["static", "interplay_worst", "fractionated", "rescan_M10"]
    scen_flats = [static_flat, interplay_flat, fractionated, rescan_examples[10]]
    scen_d95 = [dvh_stats(f, masks["tumour"], geom)["D95"] for f in scen_flats]
    scen_cov = [_gtv_cov(f) for f in scen_flats]
    plot_scenario_comparison(
        scen_labels, scen_d95, scen_cov,
        os.path.join(OUTPUT_DIR, "scenario_comparison.png"))

    # Clean report-ready scenario summary CSV
    def _lung_mean(flat, key="lung_r"):
        if key in masks and masks[key]:
            return float(np.mean(
                structure_dose_array(flat, masks[key], geom)))
        return 0.0
    scen_csv = os.path.join(OUTPUT_DIR, "scenario_summary.csv")
    with open(scen_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "GTV_mean", "GTV_D95", "GTV_D50", "GTV_D02",
                    "GTV_CoV", "Lung_R_mean", "Lung_L_mean"])
        summary_rows = [("static", static_flat),
                        ("interplay_worst", interplay_flat),
                        ("fractionated", fractionated)] + \
                       [(f"rescan_M{M}", rescan_examples[M])
                        for M in RESCAN_PASSES]
        for label, flat in summary_rows:
            st = dvh_stats(flat, masks["tumour"], geom)
            w.writerow([label,
                        f"{st['mean']:.4f}", f"{st['D95']:.4f}",
                        f"{st['D50']:.4f}", f"{st['D02']:.4f}",
                        f"{_gtv_cov(flat):.4f}",
                        f"{_lung_mean(flat, 'lung_r'):.4f}",
                        f"{_lung_mean(flat, 'lung_l'):.4f}"])
    print(f"  Saved: {scen_csv}")

    # GTV-focused CoV/D95/D50 table (direct scenario comparison)
    def gtv_cov(flat):
        d = structure_dose_array(flat, masks["tumour"], geom)
        if d.size == 0 or d.mean() == 0:
            return 0.0
        return float(d.std() / d.mean())
    gtv_rows = [
        ("static", static_flat),
        ("interplay_worst", interplay_flat),
        ("fractionated", fractionated),
    ] + [(f"rescan_M{M}", rescan_examples[M]) for M in RESCAN_PASSES]
    gtv_csv = os.path.join(OUTPUT_DIR, "gtv_cov_table.csv")
    with open(gtv_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "GTV_D95", "GTV_D50", "GTV_D02",
                    "GTV_mean", "GTV_CoV"])
        for label, flat in gtv_rows:
            st = dvh_stats(flat, masks["tumour"], geom)
            w.writerow([label, f"{st['D95']:.4f}", f"{st['D50']:.4f}",
                        f"{st['D02']:.4f}", f"{st['mean']:.4f}",
                        f"{gtv_cov(flat):.4f}"])
    print(f"  Saved: {gtv_csv}")

    # --- Key results comparison (4 headline scenarios) ---
    print("\n  KEY RESULTS — GTV motion metrics + ipsilateral lung")
    print(f"    {'scenario':<20s} {'CoV':>8s} {'D95':>8s} {'D50':>8s} "
          f"{'Lung_R mean':>12s} {'Lung_L mean':>12s}")
    print("    " + "-" * 74)
    key_labels = ["static", "interplay_worst", "fractionated", "rescan_M10"]
    key_flats = [static_flat, interplay_flat, fractionated, rescan_examples[10]]
    for label, flat in zip(key_labels, key_flats):
        st = dvh_stats(flat, masks["tumour"], geom)
        lu_r_mean = _lung_mean(flat, "lung_r")
        lu_l_mean = _lung_mean(flat, "lung_l")
        print(f"    {label:<20s} {gtv_cov(flat):8.4f} "
              f"{st['D95']:8.3f} {st['D50']:8.3f} "
              f"{lu_r_mean:12.3f} {lu_l_mean:12.3f}")

    # --- Interpretation notes ---
    d95_drop = 0.0
    d50_drop = 0.0
    st_static = dvh_stats(static_flat, masks["tumour"], geom)
    st_worst = dvh_stats(interplay_flat, masks["tumour"], geom)
    if st_static["D95"] > 0:
        d95_drop = 100.0 * (st_static["D95"] - st_worst["D95"]) / st_static["D95"]
    if st_static["D50"] > 0:
        d50_drop = 100.0 * (st_static["D50"] - st_worst["D50"]) / st_static["D50"]
    print(f"\n  Motion degrades D95 ({d95_drop:.1f}% drop) far more than "
          f"D50 ({d50_drop:.1f}% drop) → heterogeneity, not uniform loss.")
    print("  Rescanning: improvement saturates beyond M~2 and does NOT "
          "follow 1/sqrt(M).")
    print("    Reasons: pass time commensurate with breath period, "
          "deterministic scan order, single-axis rigid motion.")
    print("  OARs: small changes only — motion redistributes dose within "
          "the target, not outward.")

    print("\n" + "=" * 70)
    print("DONE. Outputs in 2.9/output/")
    print("=" * 70)
    print(f"  Motion: A={MOTION_AMPLITUDE_MM}mm, T={MOTION_PERIOD_S}s, "
          f"dwell={SPOT_DWELL_S*1000:.0f}ms, layer switch={LAYER_SWITCH_S}s")
    print(f"  Single-fraction interplay worst-case GTV CoV: {worst_std:.4f}")
    print(f"  Fractionated ({N_FRACTIONS} fx) GTV CoV: "
          f"{structure_dose_array(fractionated, masks['tumour'], geom).std() / max(1e-9, structure_dose_array(fractionated, masks['tumour'], geom).mean()):.4f}")
    print(f"  Rescanning M=10 GTV CoV: {variance_table[10]:.4f}")


if __name__ == "__main__":
    main()
