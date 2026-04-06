"""
Section 2.6: Spread Out Bragg Peak (SOBP) Proton Treatment

Three-stage pipeline:
  Part 0 — Compute WEPL along the central beam axis through the patient CT
            to determine the tumour's proximal and distal WET boundaries.
  Part 1 — Design the SOBP in a simple water phantom, using the WET-derived
            target region so that Bragg peak depths in water correspond to
            the actual range required in the patient.
  Part 2 — Quantify flatness and produce summary plots
  Part 3 — Apply the optimised SOBP to the patient CT

Usage:
    cd PHY4004_A2 && python3 A2_6/sobp_proton.py

Outputs (all in A2_6/output/):
    pristine_peaks.png                  individual Bragg peaks in water
    sobp_water.png                      combined SOBP in water + target region
    sobp_summary.csv                    energies, weights, flatness
    patient_sobp.txt                    final multi-beam patient TOPAS file
    dose_sobp_patient.dcm               DICOM dose for VICTORIA
"""

import os
import sys
import csv
import time
import shutil
import subprocess

import numpy as np

try:
    import pydicom
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydicom"])
    import pydicom

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
except Exception as exc:
    raise RuntimeError(f"Matplotlib is required: {exc}")

try:
    from scipy.optimize import nnls, curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
TOPAS_EXE = "/home/jamie/shellScripts/topas"
CT_DIR = os.path.join(PROJECT_ROOT, "CTData")
SCHNEIDER_FILE = os.path.join(PROJECT_ROOT, "HUtoMaterialSchneider.txt")

# Tumour geometry — read from RTStruct at runtime (see load_gtv_bounds()).
# Fallback values if RTStruct is missing (from GTV contour inspection).
TUMOUR_X_FALLBACK = -48.9   # mm
TUMOUR_Y_FALLBACK = 43.1    # mm
TUMOUR_Z_FALLBACK = 0.0     # mm

# SOBP parameters
# The distal energy and target depth are NO LONGER hardcoded — they are
# computed from the WEPL along the central beam axis through the patient CT.
ENERGY_SPREAD = 0.01          # 1% fractional spread per beam (default; may be tuned)
ENERGY_SPREAD_TEST = [0.0, 0.01, 0.02, 0.03]  # values to test in water (0%, 1%, 2%, 3%)
NARROW_BEAM_CUTOFF_X = 6.0  # mm — deliberately narrow cutoff for "initial" patient plot

# ------------------------------------------------------------------
# Bortfeld analytical Bragg peak model
# ------------------------------------------------------------------
# Fit each MC pristine peak to Bortfeld's analytical form to obtain
# a smooth, noise-free depth-dose curve.  The fitted profiles are
# used for NNLS weight optimisation; MC profiles are retained for
# validation plots.
#
# Reference:
#   Bortfeld (1997) "An analytical approximation of the Bragg curve
#   for therapeutic proton beams" Med. Phys. 24(12):2024–2033.
#
# Literature values for water:
#   alpha = 0.0022 cm·MeV^(-p) = 0.022 mm·MeV^(-p)
#   p     = 1.77
# ------------------------------------------------------------------
USE_BORTFELD_FIT = False
BORTFELD_P_LITERATURE = 1.77       # Bortfeld 1997 (water)
BORTFELD_ALPHA_LIT_MM = 0.022      # mm · MeV^(-p)  (0.0022 cm)
BORTFELD_FIT_GRID_STEP = 0.25      # mm — resampling grid for analytical profiles

# Pristine peak energies to test in water (MeV).
# Coarse grid covers the full plausible range; a fine grid (0.5 MeV steps)
# is added at runtime around the WET-selected target region.
PRISTINE_ENERGIES_COARSE = sorted(np.arange(60.0, 120.0, 2.0).tolist())
PRISTINE_ENERGY_FINE_STEP = 0.5  # MeV — fine grid step within target region
PRISTINE_ENERGY_FINE_MARGIN = 4.0  # MeV — extend fine grid beyond target edges

# Water phantom geometry
WATER_HLX = 50.0   # mm (half-length, 10 cm total lateral)
WATER_HLY = 50.0
WATER_HLZ = 100.0  # mm (half-length, 20 cm total depth)
WATER_Z_BINS = 800  # 0.25 mm depth resolution
WATER_HISTORIES = 80000
PROFILE_SMOOTH_WINDOW = 3  # depth-bin moving average applied to profiles

# Patient beam geometry (from A2_5 verified setup)
BEAM_TRANS_X = -46.0    # mm
BEAM_TRANS_Y = 140.0    # mm
BEAM_TRANS_Z = 0.0      # mm
BEAM_ROT_X = -90.0      # deg
BEAM_ROT_Y = 0.0        # deg
BEAM_ROT_Z = 0.0        # deg
BEAM_POS_CUTOFF_X = 12.0  # mm (covers 30mm tumour + 5mm margin)
BEAM_POS_CUTOFF_Y = 12.0  # mm
BEAM_ANG_CUTOFF = 5.0     # deg
BEAM_ANG_SPREAD = 0.5     # deg

PATIENT_HISTORIES = 40000   # low-stat debug run (bump to 500000 for production)

# Beam width sweep (Part 3b)
BEAM_WIDTH_SWEEP_X = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]  # mm
BEAM_WIDTH_SWEEP_Y = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]        # mm
SWEEP_HISTORIES = 4000   # low-stat debug sweeps; increase for production

# Structure aliases for RTStruct DVH scoring
STRUCTURE_ALIASES = {
    "tumour": ["GTVp", "GTV", "Tumour", "Tumor"],
    "ptv":    ["PTV"],
    "lung_r": ["Lung_R", "Right Lung", "LungR"],
    "heart":  ["Heart"],
    "cord":   ["SpinalCord", "Spinal Cord", "Cord"],
    "body":   ["Body", "External", "BODY", "EXTERNAL"],
}


# ==================================================================
# PART 0 — WET COMPUTATION ALONG CENTRAL BEAM AXIS
# ==================================================================
# The SOBP target region in water must correspond to the Water
# Equivalent Path Length (WEPL) through the patient at the tumour's
# proximal and distal edges.  WEPL is computed as:
#
#     WEPL = integral of RSP(x) dx
#
# where RSP is the Relative Stopping Power at each point along the
# beam.  We approximate RSP from the CT using the Schneider
# calibration that TOPAS itself uses (see justification below).
#
# This computation is performed along the central beam axis only
# (a single ray at X=TUMOUR_X, Z=TUMOUR_Z).  This is a first-order
# range model; it does not account for lateral density variations
# across the beam cross-section.
# ------------------------------------------------------------------

def load_ct_geometry():
    """Read CT DICOM headers to get voxel grid dimensions."""
    slices = []
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(CT_DIR, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "CT":
                slices.append(ds)
        except Exception:
            pass
    if not slices:
        raise FileNotFoundError(f"No CT slices found in {CT_DIR}")

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    ds0 = slices[0]
    zs = [float(s.ImagePositionPatient[2]) for s in slices]

    return {
        "rows": int(ds0.Rows),
        "cols": int(ds0.Columns),
        "n_slices": len(zs),
        "dx": float(ds0.PixelSpacing[1]),   # column spacing (X)
        "dy": float(ds0.PixelSpacing[0]),   # row spacing (Y)
        "dz": abs(zs[1] - zs[0]) if len(zs) > 1 else 1.0,
        "x0": float(ds0.ImagePositionPatient[0]),
        "y0": float(ds0.ImagePositionPatient[1]),
        "z0": zs[0],
        "slice_zs": zs,
    }


def load_gtv_bounds():
    """
    Read the GTVp contour from the RTStruct DICOM and return the
    bounding box of the tumour in patient coordinates.

    Returns a dict with keys:
        x_min, x_max, y_min, y_max, z_min, z_max,
        centre_x, centre_y, centre_z
    """
    rtstruct_path = None
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(CT_DIR, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                rtstruct_path = fpath
                break
        except Exception:
            pass

    if rtstruct_path is None:
        print("  WARNING: no RTStruct found — using fallback tumour coordinates")
        r = 15.0
        return {
            "x_min": TUMOUR_X_FALLBACK - r, "x_max": TUMOUR_X_FALLBACK + r,
            "y_min": TUMOUR_Y_FALLBACK - r, "y_max": TUMOUR_Y_FALLBACK + r,
            "z_min": TUMOUR_Z_FALLBACK - r, "z_max": TUMOUR_Z_FALLBACK + r,
            "centre_x": TUMOUR_X_FALLBACK,
            "centre_y": TUMOUR_Y_FALLBACK,
            "centre_z": TUMOUR_Z_FALLBACK,
        }

    ds = pydicom.dcmread(rtstruct_path)
    roi_map = {}
    for roi in ds.StructureSetROISequence:
        roi_map[int(roi.ROINumber)] = str(roi.ROIName)

    # Find GTVp contour
    gtv_aliases = ["GTVp", "GTV", "Tumour", "Tumor"]
    for rc in ds.ROIContourSequence:
        rn = int(getattr(rc, "ReferencedROINumber", -1))
        name = roi_map.get(rn, "")
        if not any(name.lower() == a.lower() for a in gtv_aliases):
            continue
        all_pts = []
        for c in rc.ContourSequence:
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            all_pts.append(pts)
        all_pts = np.vstack(all_pts)
        return {
            "x_min": float(all_pts[:, 0].min()),
            "x_max": float(all_pts[:, 0].max()),
            "y_min": float(all_pts[:, 1].min()),
            "y_max": float(all_pts[:, 1].max()),
            "z_min": float(all_pts[:, 2].min()),
            "z_max": float(all_pts[:, 2].max()),
            "centre_x": float(all_pts[:, 0].mean()),
            "centre_y": float(all_pts[:, 1].mean()),
            "centre_z": float(all_pts[:, 2].mean()),
        }

    raise RuntimeError("GTVp contour not found in RTStruct")


def load_ct_hu_column(geom, x_mm, z_mm):
    """
    Extract a column of HU values along the Y axis (all rows) at the
    nearest (X, Z) position to the requested coordinates.

    Returns (y_centres, hu_values) as 1-D numpy arrays sorted by
    ascending Y.
    """
    # Find the CT slice closest to z_mm
    iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - z_mm)))
    target_z = geom["slice_zs"][iz]

    # Find the file for that slice
    ct_files = sorted(
        f for f in os.listdir(CT_DIR) if f.lower().endswith(".dcm")
    )
    target_ds = None
    for fname in ct_files:
        ds = pydicom.dcmread(os.path.join(CT_DIR, fname))
        if getattr(ds, "Modality", "") != "CT":
            continue
        if abs(float(ds.ImagePositionPatient[2]) - target_z) < 0.1:
            target_ds = ds
            break
    if target_ds is None:
        raise FileNotFoundError(f"Could not find CT slice at Z={target_z:.1f}")

    # Column index for X
    ix = int(round((x_mm - geom["x0"]) / geom["dx"]))
    ix = max(0, min(ix, geom["cols"] - 1))

    # Extract the column (all rows = Y positions)
    slope = float(getattr(target_ds, "RescaleSlope", 1.0))
    intercept = float(getattr(target_ds, "RescaleIntercept", 0.0))
    pixel_column = target_ds.pixel_array[:, ix].astype(float)
    hu_column = pixel_column * slope + intercept

    # Y centres for each row
    y_centres = geom["y0"] + (np.arange(geom["rows"]) + 0.5) * geom["dy"]

    return y_centres, hu_column


def parse_schneider_params():
    """
    Parse the Schneider HU-to-density parameters and DensityCorrection
    factors from HUtoMaterialSchneider.txt.

    Returns a dict with keys:
        hu_sections   : list of 8 boundary HU values (7 segments)
        offsets       : list of 7 floats
        factors       : list of 7 floats
        factor_offsets: list of 7 floats
        density_corr  : list of 3996 floats (index 0 = HU -1000)
    """
    params = {}

    # Map TOPAS parameter keys to our dict keys + expected counts.
    # Ordered longest-first so that "SchneiderDensityFactorOffset"
    # is tested before "SchneiderDensityFactor" (substring match).
    key_map = [
        ("Ge/Patient/SchneiderHounsfieldUnitSections", "hu_sections", 8),
        ("Ge/Patient/SchneiderDensityFactorOffset", "factor_offsets", 7),
        ("Ge/Patient/SchneiderDensityOffset", "offsets", 7),
        ("Ge/Patient/SchneiderDensityFactor", "factors", 7),
        ("Ge/Patient/DensityCorrection", "density_corr", 3996),
    ]

    with open(SCHNEIDER_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for topas_key, our_key, expected_n in key_map:
                if topas_key in line:
                    # Format: "type:path/key = N val1 val2 ..."
                    rhs = line.split("=", 1)[1].strip()
                    tokens = rhs.split()
                    count = int(tokens[0])
                    values = [float(v) for v in tokens[1:count + 1]]
                    params[our_key] = values
                    break

    for key in ("hu_sections", "offsets", "factors", "factor_offsets", "density_corr"):
        if key not in params:
            raise ValueError(f"Missing Schneider parameter: {key}")

    return params


def hu_to_rsp(hu_array, schneider):
    """
    Convert an array of HU values to approximate Relative Stopping
    Power (RSP) using the Schneider calibration.

    Approximation used:
        RSP ≈ base_density × DensityCorrection[HU + 1000]

    where base_density comes from the Schneider piecewise linear
    HU-to-density formula:
        base_density = Offset_i + Factor_i × (FactorOffset_i + HU)

    and DensityCorrection is described in the Schneider file as
    "Correction Factor for the relative stopping power of Geant4
    and the XiO planning system".

    This is a Schneider-based approximation, not an exact stopping-
    power calculation.  The DensityCorrection factors encode stopping-
    power corrections that TOPAS uses internally for dose transport.
    Using the same correction ensures our WEPL estimate is consistent
    with the simulation's energy-loss model.  For biological tissues
    the RSP-density relationship is well correlated, making this a
    physically reasonable first-order approximation.

    Limitation: this does not account for I-value (mean excitation
    energy) variations between tissues at the same density.  A more
    rigorous treatment would use a dedicated stoichiometric HU-to-RSP
    calibration curve.
    """
    hu = np.asarray(hu_array, dtype=float)
    hu_clamped = np.clip(hu, -1000.0, 2995.0)

    sections = schneider["hu_sections"]   # 8 boundaries → 7 segments
    offsets = schneider["offsets"]
    factors = schneider["factors"]
    foffsets = schneider["factor_offsets"]
    dcorr = np.array(schneider["density_corr"])

    # Determine which segment each HU value falls into
    base_density = np.zeros_like(hu, dtype=float)
    for i in range(len(offsets)):   # 7 segments
        lo = sections[i]
        hi = sections[i + 1]
        mask = (hu_clamped >= lo) & (hu_clamped < hi)
        base_density[mask] = (
            offsets[i] + factors[i] * (foffsets[i] + hu_clamped[mask])
        )

    # Look up DensityCorrection (index 0 = HU -1000)
    corr_idx = np.clip(np.round(hu_clamped).astype(int) + 1000, 0, len(dcorr) - 1)
    correction = dcorr[corr_idx]

    rsp = base_density * correction
    return rsp


def compute_wet_along_beam(geom, x_mm, z_mm, schneider):
    """
    Compute cumulative Water Equivalent Path Length (WEPL) along the
    central beam axis through the patient CT.

    The beam fires in the −Y direction (RotX = −90°), so we traverse
    voxels from high Y (beam entrance) to low Y (distal).

    WEPL is computed as:
        WEPL = Σ RSP_i × Δy
    where RSP_i is the approximate relative stopping power of voxel i
    and Δy is the voxel thickness along the beam axis (mm).

    Returns:
        y_centres : 1-D array of Y positions in beam order (high → low Y)
        cum_wet   : 1-D array of cumulative WET (mm), starting at 0
    """
    y_centres, hu_values = load_ct_hu_column(geom, x_mm, z_mm)
    rsp = hu_to_rsp(hu_values, schneider)

    # Reverse to beam order: high Y (entrance) first
    order = np.argsort(-y_centres)
    y_ordered = y_centres[order]
    rsp_ordered = rsp[order]

    wet_increments = rsp_ordered * geom["dy"]
    cum_wet = np.cumsum(wet_increments)
    # Prepend 0 so cum_wet[i] is the WET at the entrance face of voxel i
    cum_wet = np.insert(cum_wet, 0, 0.0)[:-1]

    return y_ordered, cum_wet


def find_tumour_wet_boundaries(y_centres, cum_wet, gtv_bounds):
    """
    Interpolate the cumulative WET at the tumour's proximal and distal
    Y-edges (from the RTStruct bounding box).

    The beam travels in −Y, so:
        proximal edge (beam hits first) = y_max (highest Y)
        distal edge   (beam hits last)  = y_min (lowest Y)

    Returns (wet_proximal, wet_distal, proximal_y, distal_y).
    """
    proximal_y = gtv_bounds["y_max"]
    distal_y = gtv_bounds["y_min"]

    # y_centres is in descending order; np.interp needs ascending x
    y_asc = y_centres[::-1]
    wet_asc = cum_wet[::-1]

    wet_proximal = float(np.interp(proximal_y, y_asc, wet_asc))
    wet_distal = float(np.interp(distal_y, y_asc, wet_asc))

    return wet_proximal, wet_distal, proximal_y, distal_y


def select_distal_energy(profiles, wet_distal):
    """
    Select the pristine-peak energy whose Bragg peak depth in water
    best matches the tumour's distal-edge WET.

    Prefers an energy whose peak is at or slightly beyond wet_distal
    to ensure full distal coverage.
    """
    best_e = None
    best_diff = float("inf")

    for e in sorted(profiles.keys()):
        bp = find_bragg_peak_depth(*profiles[e])
        diff = bp - wet_distal   # positive = peak is beyond target
        # Prefer slight overshoot (diff >= 0) over undershoot
        penalty = abs(diff) if diff >= 0 else abs(diff) * 2.0
        if penalty < best_diff:
            best_diff = penalty
            best_e = e

    return best_e


# ==================================================================
# PART 1 — WATER PHANTOM SOBP DESIGN
# ==================================================================

# ------------------------------------------------------------------
# TOPAS file generation: water phantom
# ------------------------------------------------------------------
def generate_water_topas(energy_mev, output_basename, n_histories,
                         energy_spread=None):
    """
    Generate a TOPAS file for a single pristine Bragg peak in a
    water phantom with fine depth binning.

    Uses the same beam convention as A2_5 (RotX=-90, beam along -Y
    from +Y side), which is verified to work.  The depth axis is Y,
    binned into WATER_Y_BINS steps of 0.5 mm.
    """
    if energy_spread is None:
        energy_spread = ENERGY_SPREAD
    lines = [
        f"# Pristine Bragg peak: {energy_mev:.1f} MeV proton in water",
        "",
        "# World",
        's:Ge/World/Type     = "TsBox"',
        's:Ge/World/Material = "Vacuum"',
        "d:Ge/World/HLX      = 300.0 mm",
        "d:Ge/World/HLY      = 300.0 mm",
        "d:Ge/World/HLZ      = 300.0 mm",
        "",
        "# Water phantom — depth along Y, binned for scoring",
        's:Ge/WaterPhantom/Type     = "TsBox"',
        's:Ge/WaterPhantom/Parent   = "World"',
        's:Ge/WaterPhantom/Material = "G4_WATER"',
        f"d:Ge/WaterPhantom/HLX      = {WATER_HLX:.1f} mm",
        f"d:Ge/WaterPhantom/HLY      = {WATER_HLZ:.1f} mm",  # depth axis
        f"d:Ge/WaterPhantom/HLZ      = {WATER_HLX:.1f} mm",
        "d:Ge/WaterPhantom/TransX    = 0.0 mm",
        "d:Ge/WaterPhantom/TransY    = 0.0 mm",
        "d:Ge/WaterPhantom/TransZ    = 0.0 mm",
        "i:Ge/WaterPhantom/XBins     = 1",
        f"i:Ge/WaterPhantom/YBins     = {WATER_Z_BINS}",  # depth bins along Y
        "i:Ge/WaterPhantom/ZBins     = 1",
        "",
        "# Run control",
        f"i:Ts/ShowHistoryCountAtInterval = {max(1000, n_histories // 10)}",
        "i:Ts/NumberOfThreads            = 0",
        'b:Ts/PauseBeforeQuit            = "False"',
        "",
        "# Beam source — same convention as A2_5 (RotX=-90, fires along -Y)",
        's:Ge/BeamPos/Type   = "Group"',
        's:Ge/BeamPos/Parent = "World"',
        "d:Ge/BeamPos/TransX = 0.0 mm",
        f"d:Ge/BeamPos/TransY = {WATER_HLZ + 50.0:.1f} mm",
        "d:Ge/BeamPos/TransZ = 0.0 mm",
        f"d:Ge/BeamPos/RotX   = {BEAM_ROT_X:.1f} deg",
        "d:Ge/BeamPos/RotY   = 0.0 deg",
        "d:Ge/BeamPos/RotZ   = 0.0 deg",
        "",
        's:So/Beam/Type                     = "Beam"',
        's:So/Beam/Component                = "BeamPos"',
        's:So/Beam/BeamParticle             = "proton"',
        f"d:So/Beam/BeamEnergy               = {energy_mev:.1f} MeV",
        f"u:So/Beam/BeamEnergySpread         = {energy_spread * 100.0}",
        f"i:So/Beam/NumberOfHistoriesInRun   = {n_histories}",
        's:So/Beam/BeamPositionDistribution = "Flat"',
        's:So/Beam/BeamPositionCutoffShape  = "Ellipse"',
        f"d:So/Beam/BeamPositionCutoffX      = {BEAM_POS_CUTOFF_X:.1f} mm",
        f"d:So/Beam/BeamPositionCutoffY      = {BEAM_POS_CUTOFF_Y:.1f} mm",
        's:So/Beam/BeamAngularDistribution  = "Gaussian"',
        f"d:So/Beam/BeamAngularCutoffX       = {BEAM_ANG_CUTOFF:.1f} deg",
        f"d:So/Beam/BeamAngularCutoffY       = {BEAM_ANG_CUTOFF:.1f} deg",
        f"d:So/Beam/BeamAngularSpreadX       = {BEAM_ANG_SPREAD:.1f} deg",
        f"d:So/Beam/BeamAngularSpreadY       = {BEAM_ANG_SPREAD:.1f} deg",
        "",
        "# Depth-dose scorer (bins inherited from WaterPhantom geometry)",
        's:Sc/DepthDose/Quantity                  = "DoseToMedium"',
        's:Sc/DepthDose/Component                 = "WaterPhantom"',
        's:Sc/DepthDose/OutputType                = "csv"',
        f's:Sc/DepthDose/OutputFile                = "{output_basename}"',
        's:Sc/DepthDose/IfOutputFileAlreadyExists = "Overwrite"',
        'b:Sc/DepthDose/Active                    = "True"',
    ]

    filepath = os.path.join(PROJECT_ROOT, output_basename + "_run.txt")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return filepath


# ------------------------------------------------------------------
# TOPAS execution
# ------------------------------------------------------------------
def run_topas(param_file):
    rel = os.path.relpath(param_file, PROJECT_ROOT)
    result = subprocess.run(
        [TOPAS_EXE, rel],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-800:]
        raise RuntimeError(f"TOPAS failed for {rel}:\n{tail}")


# ------------------------------------------------------------------
# Read water phantom depth-dose CSV
# ------------------------------------------------------------------
def read_water_csv(csv_path):
    """
    Read TOPAS CSV from the water phantom scorer.

    Returns (depths_mm, doses) as numpy arrays, where depth is
    measured from the beam entrance (+Y face of the phantom).

    The phantom depth axis is Y, spanning -HLZ to +HLZ (HLZ used as
    the depth half-length).  The beam enters at +Y (+HLZ).
    iy = 0 is at -HLZ (deepest), iy = N-1 is at +HLZ (entrance).
    Depth from entrance = HLZ - y_centre(iy).
    """
    dy = 2.0 * WATER_HLZ / WATER_Z_BINS  # bin width in mm

    dose_by_iy = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                iy = int(parts[1])  # Y-bin index (depth axis)
                dose = float(parts[3])
                dose_by_iy[iy] = dose
            except (ValueError, IndexError):
                continue

    if not dose_by_iy:
        print(f"  WARNING: could not parse dose data from {csv_path}")

    depths = []
    doses = []
    for iy in range(WATER_Z_BINS):
        y_centre = -WATER_HLZ + (iy + 0.5) * dy
        depth = WATER_HLZ - y_centre  # distance from +Y entrance face
        depths.append(depth)
        doses.append(dose_by_iy.get(iy, 0.0))

    # Sort by increasing depth (entrance → deep)
    order = np.argsort(depths)
    depths_arr = np.array(depths)[order]
    doses_arr = np.array(doses)[order]

    # Light moving-average smoothing to suppress per-bin MC noise while
    # preserving the Bragg peak shape (0.75 mm window at 3 bins).
    if PROFILE_SMOOTH_WINDOW >= 2 and len(doses_arr) >= PROFILE_SMOOTH_WINDOW:
        w = PROFILE_SMOOTH_WINDOW
        kernel = np.ones(w) / w
        pad = w // 2
        padded = np.pad(doses_arr, (pad, pad), mode="edge")
        doses_arr = np.convolve(padded, kernel, mode="valid")[: len(depths_arr)]

    return depths_arr, doses_arr


# ------------------------------------------------------------------
# Run all pristine peaks
# ------------------------------------------------------------------
def run_pristine_peaks(energies, energy_spread=None, tag_suffix=""):
    """
    Run a pristine Bragg peak in the water phantom for each energy.
    Returns {energy: (depths_mm, doses_array)}.
    """
    profiles = {}
    n = len(energies)
    for i, energy in enumerate(energies, 1):
        basename = os.path.join("A2_6", "output", "_water",
                                f"pristine_{energy:.1f}MeV{tag_suffix}")
        csv_path = os.path.join(PROJECT_ROOT, basename + ".csv")
        if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
            depths, doses = read_water_csv(csv_path)
            profiles[energy] = (depths, doses)
            print(f"  [{i}/{n}] {energy:.1f} MeV ... cached "
                  f"(peak at {depths[np.argmax(doses)]:.1f} mm)")
            continue
        print(f"  [{i}/{n}] {energy:.1f} MeV ... ", end="", flush=True)
        t0 = time.time()
        param_file = generate_water_topas(energy, basename, WATER_HISTORIES,
                                          energy_spread=energy_spread)
        run_topas(param_file)
        depths, doses = read_water_csv(csv_path)
        profiles[energy] = (depths, doses)
        try:
            os.remove(param_file)
        except OSError:
            pass
        print(f"peak at {depths[np.argmax(doses)]:.1f} mm  ({time.time()-t0:.1f}s)")
    return profiles


# ------------------------------------------------------------------
# Identify target region and optimise weights
# ------------------------------------------------------------------
def find_bragg_peak_depth(depths, doses):
    """Return the depth (mm) of the Bragg peak maximum.
    Only searches beyond the entrance region (depth > 5 mm) to avoid
    picking up entrance dose noise as the 'peak'."""
    past_entrance = depths > 5.0
    if past_entrance.any() and np.max(doses[past_entrance]) > 0:
        return float(depths[past_entrance][np.argmax(doses[past_entrance])])
    return float(depths[np.argmax(doses)])


def define_target_region(wet_proximal, wet_distal):
    """
    Define the SOBP target region using WET boundaries computed from
    the patient CT.  The proximal and distal values are Water
    Equivalent Path Lengths (mm) to the tumour edges.
    """
    return wet_proximal, wet_distal


def optimize_weights(energies, profiles, target_proximal, target_distal):
    """
    Find non-negative beam weights that produce a flat combined
    dose profile across the target region.

    Uses non-negative least squares: minimise ||A w - b||^2, w >= 0,
    where A contains the pristine dose profiles in the target region
    and b is a uniform target dose.
    """
    # All profiles share the same depth grid
    depths = profiles[energies[0]][0]

    # Target mask: exactly the tumour WET region.
    target_mask = (depths >= target_proximal) & (depths <= target_distal)
    n_target = int(np.sum(target_mask))

    # Build the dose matrix: only the target region matters.
    # Entrance dose (proximal) and distal falloff are physics — not
    # degrees of freedom the optimizer should fight against.
    A_target = np.column_stack([profiles[e][1][target_mask] for e in energies])

    # Prescription: median plateau dose of a mid-target beam.
    mid_e = energies[len(energies) // 2]
    mid_dose = profiles[mid_e][1][target_mask]
    above = mid_dose[mid_dose > np.max(mid_dose) * 0.3]
    target_level = float(np.median(above)) if len(above) else 1.0
    b_target = np.full(n_target, target_level)

    # Light smoothness regularisation (Tikhonov first-difference)
    # prevents noisy weight oscillations without distorting the SOBP.
    n_e = len(energies)
    SMOOTH_WEIGHT = 0.02 * target_level
    L = np.zeros((n_e - 1, n_e))
    for j in range(n_e - 1):
        L[j, j] = -1.0
        L[j, j + 1] = 1.0
    L *= SMOOTH_WEIGHT

    # Stack: [ A_target ] w ≈ [ b_target ]
    #        [ λ·L      ]     [ 0        ]
    A = np.vstack([A_target, L])
    b = np.concatenate([b_target, np.zeros(n_e - 1)])

    # Solve with NNLS
    if HAS_SCIPY:
        w, _ = nnls(A, b)
    else:
        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        for _ in range(50):
            w = np.maximum(w, 0.0)
            active = w > 0
            if not active.any():
                break
            w_sub, _, _, _ = np.linalg.lstsq(A[:, active], b, rcond=None)
            w[active] = np.maximum(w_sub, 0.0)

    # Normalise so weights sum to 1
    if w.sum() > 0:
        w = w / w.sum()

    return w


# ------------------------------------------------------------------
# Bortfeld analytical Bragg peak fitting
# ------------------------------------------------------------------
def bortfeld_unstraggled(z, R0, epsilon, p=BORTFELD_P_LITERATURE):
    """
    Unstraggled depth-dose (Bortfeld 1997, pre-Gaussian-convolution).

    D_0(z) ∝ (R0 - z)^(1/p - 1) + ε · (R0 - z)^(1/p)    for z < R0
           = 0                                             for z >= R0

    The first term is the plateau / Bragg peak tail (stopping power)
    and the second term is the nuclear-interaction low-energy tail.
    """
    d = np.zeros_like(z, dtype=float)
    mask = z < R0
    dr = R0 - z[mask]
    d[mask] = dr ** (1.0 / p - 1.0) + epsilon * dr ** (1.0 / p)
    return d


def bortfeld_peak(z, R0, sigma, epsilon, phi, p=BORTFELD_P_LITERATURE):
    """
    Straggled pristine proton Bragg peak (numerical-convolution form).

    Convolves the unstraggled curve with a Gaussian kernel of width
    sigma to account for range straggling.  Mathematically equivalent
    to the Bortfeld parabolic-cylinder expression but simpler to
    evaluate and numerically stable.

    Parameters:
        z       : depth array (mm), uniform spacing
        R0      : range (mm)
        sigma   : range straggling width (mm)
        epsilon : nuclear interaction fraction (dimensionless)
        phi     : overall fluence/normalisation
        p       : range-energy exponent (default: Bortfeld literature)

    Returns:
        dose array on the input z grid (same units as phi × base term)
    """
    if len(z) < 2:
        return np.zeros_like(z)
    dz = z[1] - z[0]

    # Evaluate on a fine sub-grid and bin-average back, so that the
    # integrable divergence of (R0-z)^(1/p-1) near z=R0 is integrated
    # (not point-sampled) — otherwise a single grid point falling very
    # close to R0 produces a spurious spike that the Gaussian kernel
    # cannot wash out.
    sub = 10
    dz_fine = dz / sub
    z_fine = np.arange(z[0] - 0.5 * dz + 0.5 * dz_fine,
                       z[-1] + 0.5 * dz, dz_fine)
    d0_fine = bortfeld_unstraggled(z_fine, R0, epsilon, p)

    # Gaussian kernel on fine grid (5 sigma half-width)
    n_half = max(1, int(np.ceil(5.0 * sigma / dz_fine)))
    zk = np.arange(-n_half, n_half + 1) * dz_fine
    kernel = np.exp(-0.5 * (zk / sigma) ** 2)
    kernel /= kernel.sum()

    d_fine = np.convolve(d0_fine, kernel, mode="same")

    # Bin-average back to the input grid
    n_bins = len(z)
    d_fine = d_fine[: n_bins * sub]
    d_conv = d_fine.reshape(n_bins, sub).mean(axis=1)
    return phi * np.maximum(d_conv, 0.0)


def fit_bortfeld_peak(depths, doses, energy_mev):
    """
    Fit Bortfeld analytical model to a single MC pristine peak.

    Initial guesses:
        R0    : MC peak depth (argmax)
        sigma : 0.012 * R0^0.935 (Bortfeld range-straggling relation, mm)
        eps   : 0.10 (typical nuclear fraction at therapeutic energies)
        phi   : from rough normalisation at the peak

    Returns:
        dict with fitted parameters and fit quality R²
    """
    if not HAS_SCIPY:
        return None

    from scipy.optimize import least_squares

    depths = np.asarray(depths, dtype=float)
    doses = np.asarray(doses, dtype=float)

    # Initial guesses
    R0_init = float(depths[int(np.argmax(doses))])
    # Bortfeld straggling: sigma ≈ 0.012 · R^0.935 (cm)  →  mm scaling
    sigma_init = 0.012 * (R0_init / 10.0) ** 0.935 * 10.0  # mm
    sigma_init = max(sigma_init, 0.3)
    eps_init = 0.02

    # Rough phi from peak height: run the model once with phi=1 to get
    # the characteristic peak amplitude, then scale to match MC.
    probe = bortfeld_peak(depths, R0_init, sigma_init, eps_init, 1.0)
    probe_peak = float(np.max(probe))
    phi_init = float(np.max(doses)) / max(probe_peak, 1e-30)
    phi_init = max(phi_init, 1e-18)

    # Peak-focused window: [R0 - 25mm, R0 + 3mm].
    window_lo = R0_init - 25.0
    window_hi = R0_init + 3.0
    wmask = (depths >= window_lo) & (depths <= window_hi)
    if wmask.sum() < 10:
        wmask = np.ones_like(depths, dtype=bool)
    z_win = depths[wmask]
    d_win = doses[wmask]

    # Normalised linear residuals: (model − data) / sqrt(data + floor).
    # Balances peak and plateau while remaining well-behaved near R0.
    d_floor = 0.05 * float(np.max(d_win)) if np.max(d_win) > 0 else 1e-9

    def residuals(params):
        R0, sigma, epsilon, phi = params
        m = bortfeld_peak(z_win, R0, sigma, epsilon, phi)
        return (m - d_win) / np.sqrt(d_win + d_floor)

    # Explicit parameter scaling — without this, |phi|~1e-6 ruins the
    # Jacobian conditioning and the optimiser stalls at the initial point.
    x_scale = [5.0, 1.0, 0.1, phi_init]

    bounds = (
        [R0_init - 3.0, 0.1, 0.0, phi_init * 1e-3],
        [R0_init + 3.0, 8.0, 0.5, phi_init * 1e3],
    )

    try:
        result = least_squares(
            residuals,
            x0=[R0_init, sigma_init, eps_init, phi_init],
            bounds=bounds, x_scale=x_scale, method="trf",
            diff_step=0.01, loss="soft_l1", f_scale=0.3,
            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=5000,
        )
        popt = result.x
    except (RuntimeError, ValueError) as exc:
        print(f"    fit failed for {energy_mev} MeV: {exc}")
        return None

    # Compute R² on the full profile (not just the fit window)
    d_fit = bortfeld_peak(depths, *popt)
    ss_res = float(np.sum((doses - d_fit) ** 2))
    ss_tot = float(np.sum((doses - np.mean(doses)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "energy": float(energy_mev),
        "R0": float(popt[0]),
        "sigma": float(popt[1]),
        "epsilon": float(popt[2]),
        "phi": float(popt[3]),
        "r2": r2,
        "d_fit": d_fit,
    }


def fit_alpha_p_from_ranges(fits):
    """
    Given a list of per-energy fits (with 'energy' and 'R0' keys),
    fit the range-energy relation R0(E) = alpha · E^p via log-log
    linear regression with iterative outlier rejection.

    Returns (alpha_mm, p, r2_loglog).
    """
    energies = np.array([f["energy"] for f in fits])
    R0s = np.array([f["R0"] for f in fits])
    mask = (energies > 0) & (R0s > 0)
    if mask.sum() < 3:
        return None

    # Iterative outlier rejection: fit, remove >3σ outliers, refit
    for _ in range(3):
        log_E = np.log(energies[mask])
        log_R = np.log(R0s[mask])
        p_fit, log_alpha = np.polyfit(log_E, log_R, 1)
        log_R_pred = log_alpha + p_fit * log_E
        residuals = np.abs(log_R - log_R_pred)
        sigma = np.std(residuals)
        if sigma == 0:
            break
        good = residuals < 3.0 * sigma
        new_mask = mask.copy()
        new_mask[mask] &= good
        if new_mask.sum() == mask.sum() or new_mask.sum() < 3:
            break
        mask = new_mask

    log_E = np.log(energies[mask])
    log_R = np.log(R0s[mask])
    p_fit, log_alpha = np.polyfit(log_E, log_R, 1)
    alpha_mm = float(np.exp(log_alpha))
    R_pred = alpha_mm * energies[mask] ** p_fit
    ss_res = np.sum((R0s[mask] - R_pred) ** 2)
    ss_tot = np.sum((R0s[mask] - R0s[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha_mm, float(p_fit), float(r2)


def build_analytical_profiles(mc_profiles, fits, depth_step=BORTFELD_FIT_GRID_STEP,
                              r2_threshold=0.90):
    """
    Build smooth profiles on a fine uniform depth grid.

    Uses the fitted Bortfeld parameters where the fit is good
    (R² ≥ r2_threshold).  Falls back to a cubic-spline resampling
    of the smoothed MC profile for energies whose fit is poor,
    so downstream NNLS never sees a degenerate curve.
    """
    if not fits and not mc_profiles:
        return {}

    # Use the MC depth range as reference
    any_e = next(iter(mc_profiles))
    mc_depths = mc_profiles[any_e][0]
    d_min = float(mc_depths.min())
    d_max = float(mc_depths.max())
    z_fine = np.arange(d_min, d_max + 0.5 * depth_step, depth_step)

    try:
        from scipy.interpolate import CubicSpline
        HAS_SPLINE = True
    except ImportError:
        HAS_SPLINE = False

    analytical = {}
    n_bortfeld, n_spline = 0, 0
    for fit in fits:
        e = fit["energy"]
        if fit.get("r2", 0.0) >= r2_threshold:
            doses = bortfeld_peak(z_fine, fit["R0"], fit["sigma"],
                                  fit["epsilon"], fit["phi"])
            analytical[e] = (z_fine, doses)
            n_bortfeld += 1
        else:
            # Cubic-spline fallback on MC data
            mc_d, mc_dose = mc_profiles[e]
            if HAS_SPLINE:
                cs = CubicSpline(mc_d, mc_dose, extrapolate=False)
                doses = cs(z_fine)
                doses = np.nan_to_num(doses, nan=0.0)
            else:
                doses = np.interp(z_fine, mc_d, mc_dose)
            doses = np.maximum(doses, 0.0)
            analytical[e] = (z_fine, doses)
            n_spline += 1
    print(f"    Analytical profiles: {n_bortfeld} Bortfeld, "
          f"{n_spline} spline fallback (R² < {r2_threshold})")
    return analytical


def plot_bortfeld_validation(mc_profiles, fits, filepath, n_show=6):
    """
    Validation plot: MC profile vs Bortfeld fit for a representative
    subset of energies, with residuals below.
    """
    if not fits:
        return

    fits_sorted = sorted(fits, key=lambda f: f["energy"])
    if len(fits_sorted) > n_show:
        # Pick evenly spaced fits
        idx = np.linspace(0, len(fits_sorted) - 1, n_show).astype(int)
        fits_show = [fits_sorted[i] for i in idx]
    else:
        fits_show = fits_sorted

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    gridspec_kw={"height_ratios": [2.2, 1]},
                                    sharex=True)
    cmap = plt.cm.viridis
    e_min = fits_show[0]["energy"]
    e_max = fits_show[-1]["energy"]
    import matplotlib as mpl
    norm_e = mpl.colors.Normalize(vmin=e_min, vmax=e_max)

    for fit in fits_show:
        e = fit["energy"]
        if e not in mc_profiles:
            continue
        depths, mc = mc_profiles[e]
        d_fit = fit["d_fit"]
        colour = cmap(norm_e(e))
        ax1.plot(depths, mc, ".", color=colour, markersize=3, alpha=0.5)
        ax1.plot(depths, d_fit, "-", color=colour, linewidth=1.5,
                 label=f"{e:.1f} MeV (R²={fit['r2']:.3f})")
        resid = mc - d_fit
        peak = np.max(mc)
        dose_region = mc > 0.10 * peak
        ax2.plot(depths[dose_region], resid[dose_region], "-",
                 color=colour, linewidth=0.8)

    ax1.set_ylabel("Dose (Gy)", fontsize=12)
    ax1.set_title("Bortfeld analytical fit vs MC pristine peaks", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9, frameon=True)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Depth in water (mm)", fontsize=12)
    ax2.set_ylabel("MC − fit (Gy)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    # Clip residual axis using the 95th percentile of absolute residuals
    # Only consider depths where dose exceeds 10% of peak to exclude
    # noise in the zero-dose region
    all_resid = []
    for fit in fits_show:
        e = fit["energy"]
        if e not in mc_profiles:
            continue
        depths, mc = mc_profiles[e]
        resid = mc - fit["d_fit"]
        peak = np.max(mc)
        dose_mask = mc > 0.10 * peak
        all_resid.extend(np.abs(resid[dose_mask]))
    if all_resid:
        clip = float(np.percentile(all_resid, 95)) * 2.0
        if clip > 0:
            ax2.set_ylim(-clip, clip)

    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_range_energy_relation(fits, alpha_fit, p_fit, r2, filepath):
    """Plot R0 vs E with fitted power law and literature curve."""
    if not fits:
        return
    fits_sorted = sorted(fits, key=lambda f: f["energy"])
    energies = np.array([f["energy"] for f in fits_sorted])
    R0s = np.array([f["R0"] for f in fits_sorted])

    e_grid = np.linspace(energies.min(), energies.max(), 200)
    R_fit = alpha_fit * e_grid ** p_fit
    R_lit = BORTFELD_ALPHA_LIT_MM * e_grid ** BORTFELD_P_LITERATURE

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(energies, R0s, "o", color="tab:blue", markersize=6,
            label="MC Bragg peak R$_0$")
    ax.plot(e_grid, R_fit, "-", color="tab:red", linewidth=1.8,
            label=fr"Fit: α={alpha_fit:.5f}, p={p_fit:.3f} (R²={r2:.4f})")
    ax.plot(e_grid, R_lit, "--", color="grey", linewidth=1.3,
            label=f"Literature: α={BORTFELD_ALPHA_LIT_MM}, p={BORTFELD_P_LITERATURE}")

    ax.set_xlabel("Proton energy E (MeV)", fontsize=12)
    ax.set_ylabel("Bragg peak range R$_0$ (mm)", fontsize=12)
    ax.set_title("Range–energy relation: MC Bragg peaks vs literature", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def build_sobp(energies, weights, profiles):
    """Combine pristine peaks using the given weights."""
    depths = profiles[energies[0]][0]
    combined = np.zeros_like(depths, dtype=float)
    for e, w in zip(energies, weights):
        combined += w * profiles[e][1]
    return depths, combined


# ==================================================================
# PART 2 — QUANTIFICATION
# ==================================================================

def compute_flatness(depths, combined, target_proximal, target_distal):
    """
    Flatness metric: standard deviation / mean of the dose across
    the target region.  Lower is better; 0 = perfectly flat.
    """
    mask = (depths >= target_proximal) & (depths <= target_distal)
    target_dose = combined[mask]
    if len(target_dose) == 0 or np.mean(target_dose) == 0:
        return np.nan
    return float(np.std(target_dose) / np.mean(target_dose))


def _pristine_peak_figure(energies, profiles, target_proximal, target_distal,
                          filepath, title, subtitle=None):
    """Helper: plot a single set of pristine Bragg peak profiles."""
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(11, 6.5))

    cmap = plt.cm.viridis
    e_min = min(energies)
    e_max = max(energies)
    norm_e = mpl.colors.Normalize(vmin=e_min, vmax=e_max)

    for e in sorted(energies):
        if e not in profiles:
            continue
        depths, doses = profiles[e]
        ax.plot(depths, doses, linewidth=1.0, color=cmap(norm_e(e)))

    ax.axvspan(target_proximal, target_distal, alpha=0.15, color="red",
               label="Target region")

    ax.set_xlabel("Depth in water (mm)", fontsize=13)
    ax.set_ylabel("Dose (Gy)", fontsize=13)
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_e)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Proton energy (MeV)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_pristine_peaks_mc(energies, mc_profiles, target_proximal,
                           target_distal, filepath):
    """Plot MC (TOPAS) pristine Bragg peaks — raw simulation data."""
    _pristine_peak_figure(
        energies, mc_profiles, target_proximal, target_distal, filepath,
        title="Pristine Bragg Peaks — Monte Carlo (TOPAS)",
    )


def plot_pristine_peaks_analytical(energies, profiles, target_proximal,
                                   target_distal, filepath):
    """Plot Bortfeld analytical pristine Bragg peaks — used for SOBP
    weight optimisation."""
    _pristine_peak_figure(
        energies, profiles, target_proximal, target_distal, filepath,
        title="Pristine Bragg Peaks — Bortfeld Analytical Model",
        subtitle="(used for SOBP weight optimisation)",
    )


def plot_sobp(energies, weights, profiles, depths, combined,
              target_proximal, target_distal, flatness, filepath):
    """Plot the combined SOBP with weighted individual contributions.

    # NOTE: The large number of small-weight energy components is an idealised
    # computational SOBP construction for this assignment, not necessarily a
    # literal clinical delivery scheme.  Clinical proton systems may use fewer
    # energy layers with hardware-specific constraints.
    """
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Plot weighted individual peaks, coloured by energy
    active = [(e, w) for e, w in zip(energies, weights) if w > 1e-6]
    cmap = plt.cm.viridis
    if active:
        e_min = min(e for e, _ in active)
        e_max = max(e for e, _ in active)
    else:
        e_min, e_max = 0.0, 1.0
    norm_e = mpl.colors.Normalize(vmin=e_min, vmax=e_max)

    for e, w in active:
        d, dose = profiles[e]
        ax.plot(d, w * dose, linewidth=0.9, alpha=0.55, color=cmap(norm_e(e)))

    # Combined SOBP — bold black
    ax.plot(depths, combined, "k-", linewidth=2.5, label="SOBP (combined)")

    # Target region
    ax.axvspan(target_proximal, target_distal, alpha=0.15, color="red",
               label="Target region")

    ax.set_xlabel("Depth in water (mm)", fontsize=13)
    ax.set_ylabel("Dose (Gy)", fontsize=13)
    ax.set_title(f"Spread Out Bragg Peak — Flatness = {flatness:.4f} "
                 f"(SD/mean in target)", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11, frameon=True)

    # Colorbar encodes the energy of each weighted peak
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_e)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Proton energy (MeV)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def write_summary_csv(energies, weights, flatness, profiles, filepath):
    """Write a summary table of the SOBP energies, weights, and peak depths."""
    rows = []
    for e, w in zip(energies, weights):
        peak_depth = find_bragg_peak_depth(*profiles[e])
        rows.append({
            "energy_MeV": e,
            "weight": round(w, 6),
            "bragg_peak_mm": round(peak_depth, 1),
        })
    rows.append({
        "energy_MeV": "SOBP",
        "weight": round(sum(weights), 6),
        "bragg_peak_mm": f"flatness={flatness:.4f}",
    })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["energy_MeV", "weight", "bragg_peak_mm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {filepath}")


# ==================================================================
# PART 3 — APPLY SOBP TO PATIENT
# ==================================================================

def generate_patient_sobp(energies, weights, output_basename, n_histories,
                          cutoff_x=None, cutoff_y=None,
                          energy_spread=None, output_type="dicom"):
    """
    Generate a multi-beam TOPAS file that applies the optimised SOBP
    to the patient CT, using the same geometry as Section 2.5.

    Each beam has a different energy; the number of histories per beam
    is proportional to its weight.

    Parameters:
        cutoff_x      : override BeamPositionCutoffX (mm). None = use global.
        cutoff_y      : override BeamPositionCutoffY (mm). None = use global.
        energy_spread : override beam energy spread. None = use global.
        output_type   : "dicom" for VICTORIA, "csv" for fast DVH scoring.
    """
    if cutoff_x is None:
        cutoff_x = BEAM_POS_CUTOFF_X
    if cutoff_y is None:
        cutoff_y = BEAM_POS_CUTOFF_Y
    if energy_spread is None:
        energy_spread = ENERGY_SPREAD
    # Distribute histories according to weights
    active = [(e, w) for e, w in zip(energies, weights) if w > 1e-6]
    if not active:
        raise ValueError("No beams with non-zero weight.")

    w_arr = np.array([w for _, w in active])
    w_arr = w_arr / w_arr.sum()
    raw_counts = w_arr * n_histories
    counts = np.floor(raw_counts).astype(int)
    # Distribute the remainder to the largest fractional parts
    remainder = int(n_histories - counts.sum())
    if remainder > 0:
        fracs = raw_counts - counts
        for idx in np.argsort(-fracs)[:remainder]:
            counts[idx] += 1

    lines = [
        "# SOBP patient plan — auto-generated by A2_6/sobp_proton.py",
        f"# {len(active)} beams, total {n_histories} histories",
        "",
        "includeFile = ct_geometry.txt",
        "",
        f"i:Ts/ShowHistoryCountAtInterval = {max(1000, n_histories // 20)}",
        "i:Ts/NumberOfThreads            = 0",
        'b:Ts/PauseBeforeQuit            = "False"',
        "",
    ]

    for i, ((energy, weight), nh) in enumerate(zip(active, counts), start=1):
        name = f"Beam{i}"
        lines += [
            f"# {name}: {energy:.1f} MeV, weight={weight:.4f}, histories={nh}",
            f's:Ge/{name}/Type   = "Group"',
            f's:Ge/{name}/Parent = "World"',
            f"d:Ge/{name}/TransX = {BEAM_TRANS_X:.1f} mm",
            f"d:Ge/{name}/TransY = {BEAM_TRANS_Y:.1f} mm",
            f"d:Ge/{name}/TransZ = {BEAM_TRANS_Z:.1f} mm",
            f"d:Ge/{name}/RotX   = {BEAM_ROT_X:.1f} deg",
            f"d:Ge/{name}/RotY   = {BEAM_ROT_Y:.1f} deg",
            f"d:Ge/{name}/RotZ   = {BEAM_ROT_Z:.1f} deg",
            "",
            f's:So/{name}/Type                     = "Beam"',
            f's:So/{name}/Component                = "{name}"',
            f's:So/{name}/BeamParticle             = "proton"',
            f"d:So/{name}/BeamEnergy               = {energy:.1f} MeV",
            f"u:So/{name}/BeamEnergySpread         = {energy_spread * 100.0}",
            f"i:So/{name}/NumberOfHistoriesInRun   = {nh}",
            f's:So/{name}/BeamPositionDistribution = "Flat"',
            f's:So/{name}/BeamPositionCutoffShape  = "Ellipse"',
            f"d:So/{name}/BeamPositionCutoffX      = {cutoff_x:.1f} mm",
            f"d:So/{name}/BeamPositionCutoffY      = {cutoff_y:.1f} mm",
            f's:So/{name}/BeamAngularDistribution  = "Gaussian"',
            f"d:So/{name}/BeamAngularCutoffX       = {BEAM_ANG_CUTOFF:.1f} deg",
            f"d:So/{name}/BeamAngularCutoffY       = {BEAM_ANG_CUTOFF:.1f} deg",
            f"d:So/{name}/BeamAngularSpreadX       = {BEAM_ANG_SPREAD:.1f} deg",
            f"d:So/{name}/BeamAngularSpreadY       = {BEAM_ANG_SPREAD:.1f} deg",
            "",
        ]

    lines += [
        f"# Dose scoring — {output_type} output",
        's:Sc/PatientDose/Quantity                  = "DoseToMedium"',
        's:Sc/PatientDose/Component                 = "Patient"',
        f's:Sc/PatientDose/OutputType                = "{output_type}"',
        f's:Sc/PatientDose/OutputFile                = "{output_basename}"',
        's:Sc/PatientDose/IfOutputFileAlreadyExists = "Overwrite"',
        'b:Sc/PatientDose/Active                    = "True"',
    ]

    filepath = os.path.join(PROJECT_ROOT, output_basename + ".txt")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return filepath


# ==================================================================
# PART 3b — LATERAL BEAM WIDTH OPTIMISATION
# ==================================================================
# After determining the SOBP energies and weights (Parts 0–2), this
# stage sweeps BeamPositionCutoffX to find the smallest lateral beam
# width that achieves adequate tumour coverage (D95/D50) without
# excessive normal-tissue dose (Lung_R, Body).
#
# Only CutoffX is varied; CutoffY is kept fixed.
# ------------------------------------------------------------------

def voxel_centres_xy(geom):
    """Return 1-D arrays of voxel-centre X and Y coordinates.
    ImagePositionPatient already gives the centre of pixel (0,0),
    so pixel (i) centre = x0 + i*dx."""
    xs = geom["x0"] + np.arange(geom["cols"]) * geom["dx"]
    ys = geom["y0"] + np.arange(geom["rows"]) * geom["dy"]
    return xs, ys


def closest_slice_index(z_value, slice_zs):
    return int(np.argmin(np.abs(np.asarray(slice_zs) - z_value)))


def contour_to_mask_slice(geom, iz, contour_xy):
    """Convert a single contour polygon to a set of (ix, iy, iz) voxel indices."""
    xs, ys = voxel_centres_xy(geom)
    poly = np.asarray(contour_xy, dtype=float)
    path = MplPath(poly)

    ixs = np.where((xs >= poly[:, 0].min() - geom["dx"]) &
                    (xs <= poly[:, 0].max() + geom["dx"]))[0]
    iys = np.where((ys >= poly[:, 1].min() - geom["dy"]) &
                    (ys <= poly[:, 1].max() + geom["dy"]))[0]
    if len(ixs) == 0 or len(iys) == 0:
        return set()

    xv, yv = np.meshgrid(xs[ixs], ys[iys], indexing="xy")
    points = np.column_stack([xv.ravel(), yv.ravel()])
    inside = path.contains_points(points, radius=1e-9).reshape(len(iys), len(ixs))

    voxels = set()
    for j, iy in enumerate(iys):
        for i, ix in enumerate(ixs):
            if inside[j, i]:
                voxels.add((int(ix), int(iy), int(iz)))
    return voxels


def build_rtstruct_masks(geom):
    """
    Build voxel masks for all structures in the RTStruct.
    Returns {structure_name: set of (ix, iy, iz)}.
    """
    from collections import defaultdict

    rtstruct_path = None
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(CT_DIR, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                rtstruct_path = fpath
                break
        except Exception:
            pass

    if rtstruct_path is None:
        print("  WARNING: no RTStruct found — cannot build structure masks")
        return {}

    ds = pydicom.dcmread(rtstruct_path)
    roi_map = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            roi_map[int(roi.ROINumber)] = str(roi.ROIName)

    if not hasattr(ds, "ROIContourSequence"):
        return {}

    masks_by_name = defaultdict(set)
    for roi_contour in ds.ROIContourSequence:
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", -1))
        roi_name = roi_map.get(roi_number)
        if not roi_name or not hasattr(roi_contour, "ContourSequence"):
            continue
        for contour in roi_contour.ContourSequence:
            data = np.asarray(getattr(contour, "ContourData", []), dtype=float)
            if data.size < 9:
                continue
            pts = data.reshape(-1, 3)
            iz = closest_slice_index(float(np.mean(pts[:, 2])), geom["slice_zs"])
            masks_by_name[roi_name].update(contour_to_mask_slice(geom, iz, pts[:, :2]))

    return dict(masks_by_name)


def build_scoring_masks(geom):
    """
    Build voxel masks for DVH scoring: tumour, PTV, Lung_R, Heart,
    SpinalCord, Body.  Uses RTStruct contours.
    """
    rt_masks = build_rtstruct_masks(geom)
    if not rt_masks:
        raise RuntimeError("RTStruct masks required for beam width optimisation")

    masks = {}
    for key, aliases in STRUCTURE_ALIASES.items():
        for alias in aliases:
            match = next((n for n in rt_masks if n.lower() == alias.lower()), None)
            if match:
                masks[key] = rt_masks[match]
                break

    found = list(masks.keys())
    print(f"  Scoring masks built: {found}")
    for k, v in masks.items():
        print(f"    {k}: {len(v)} voxels")

    if "tumour" not in masks:
        raise RuntimeError("GTVp mask not found in RTStruct — cannot score")

    return masks


def read_dose_csv(csv_path):
    """Read a TOPAS CSV dose file into a dict of {(ix,iy,iz): dose}."""
    dose_map = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                key = (int(parts[0]), int(parts[1]), int(parts[2]))
                dose_map[key] = float(parts[3])
            except (ValueError, IndexError):
                continue
    return dose_map


def compute_dvh_metrics(dose_map, masks):
    """
    Compute DVH metrics for each structure in masks.

    Returns a dict of {structure_name: {D95, D50, D02, mean, max}}.
    Dn = dose received by at least n% of the structure volume.
    """
    results = {}
    for name, voxels in masks.items():
        doses = np.array([dose_map.get(v, 0.0) for v in voxels])
        if len(doses) == 0:
            results[name] = {"D95": 0, "D50": 0, "D02": 0, "mean": 0, "max": 0}
            continue
        results[name] = {
            "D95": float(np.percentile(doses, 5)),
            "D50": float(np.percentile(doses, 50)),
            "D02": float(np.percentile(doses, 98)),
            "mean": float(np.mean(doses)),
            "max": float(np.max(doses)),
        }
    return results


def run_parameter_sweep(energies, weights, masks, n_histories,
                        sweep_values, sweep_param="cutoff_x",
                        fixed_cutoff_x=None, fixed_cutoff_y=None,
                        fixed_energy_spread=None):
    """
    Sweep a single beam parameter, running a patient TOPAS simulation
    for each value and scoring DVH metrics.

    Parameters:
        sweep_values  : list of values to test
        sweep_param   : "cutoff_x", "cutoff_y", or "energy_spread"
        fixed_*       : fixed values for the non-swept parameters

    Returns a list of dicts: [{param_value, metrics, csv_path}, ...].
    """
    results = []
    n = len(sweep_values)

    for i, val in enumerate(sweep_values, 1):
        tag = f"sweep_{sweep_param}_{val:.1f}"
        basename = os.path.join("A2_6", "output", "_sweep", tag)
        print(f"  [{i}/{n}] {sweep_param} = {val:.1f} ... ", end="", flush=True)

        kwargs = {
            "cutoff_x": fixed_cutoff_x,
            "cutoff_y": fixed_cutoff_y,
            "energy_spread": fixed_energy_spread,
            "output_type": "csv",
        }
        kwargs[sweep_param] = val

        t0 = time.time()
        param_file = generate_patient_sobp(
            energies, weights, basename, n_histories, **kwargs
        )
        run_topas(param_file)

        csv_path = os.path.join(PROJECT_ROOT, basename + ".csv")
        dose_map = read_dose_csv(csv_path)
        metrics = compute_dvh_metrics(dose_map, masks)

        elapsed = time.time() - t0
        tumour_d95 = metrics.get("tumour", {}).get("D95", 0)
        print(f"GTVp D95={tumour_d95:.4e}  ({elapsed:.1f}s)")

        try:
            os.remove(param_file)
        except OSError:
            pass

        results.append({
            "param_value": val,
            "param_name": sweep_param,
            "metrics": metrics,
            "csv_path": csv_path,
        })

    return results


def print_sweep_table(sweep_results, title="PARAMETER SWEEP — DVH METRICS"):
    """Print a full table of DVH metrics for each swept parameter value."""
    if not sweep_results:
        return
    param_name = sweep_results[0].get("param_name", "value")
    label_map = {"tumour": "GTVp", "ptv": "PTV", "lung_r": "Lung_R", "body": "Body"}

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    # Header
    header = f"{param_name:>14s}"
    for struct in ["tumour", "ptv", "lung_r", "body"]:
        label = label_map.get(struct, struct)
        header += f"  {label+' D95':>10s} {label+' D50':>10s}"
        if struct == "lung_r":
            header += f" {label+' D02':>10s}"
    print(f"  {header}")
    print(f"  {'-' * 97}")

    for r in sweep_results:
        val = r["param_value"]
        m = r["metrics"]
        line = f"{val:14.2f}"
        for struct in ["tumour", "ptv", "lung_r", "body"]:
            sm = m.get(struct, {"D95": 0, "D50": 0, "D02": 0})
            line += f"  {sm['D95']:10.4e} {sm['D50']:10.4e}"
            if struct == "lung_r":
                line += f" {sm['D02']:10.4e}"
        print(f"  {line}")

    print()


def plot_sweep(sweep_results, filepath, xlabel=None, title=None):
    """Plot DVH metrics vs swept parameter."""
    if not sweep_results:
        return
    param_name = sweep_results[0].get("param_name", "value")
    if xlabel is None:
        xlabel = param_name
    if title is None:
        title = f"{param_name} Optimisation — Tumour Coverage vs OAR Dose"

    xvals = [r["param_value"] for r in sweep_results]
    tumour_d95 = [r["metrics"].get("tumour", {}).get("D95", 0) for r in sweep_results]
    tumour_d50 = [r["metrics"].get("tumour", {}).get("D50", 0) for r in sweep_results]
    lung_d02 = [r["metrics"].get("lung_r", {}).get("D02", 0) for r in sweep_results]
    body_mean = [r["metrics"].get("body", {}).get("mean", 0) for r in sweep_results]

    def norm(arr):
        mx = max(arr) if max(arr) > 0 else 1.0
        return [v / mx for v in arr]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(xvals, norm(tumour_d95), "o-", color="tab:red", label="GTVp D95")
    ax1.plot(xvals, norm(tumour_d50), "s-", color="tab:blue", label="GTVp D50")
    ax1.set_ylabel("Normalised metric", fontsize=13)
    ax1.set_title(title, fontsize=15)
    ax1.tick_params(labelsize=11)
    ax1.legend(loc="best", fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(xvals, norm(lung_d02), "^-", color="tab:orange", label="Lung_R D02")
    ax2.plot(xvals, norm(body_mean), "D-", color="tab:green", label="Body mean")
    ax2.set_xlabel(xlabel, fontsize=13)
    ax2.set_ylabel("Normalised metric", fontsize=13)
    ax2.tick_params(labelsize=11)
    ax2.legend(loc="best", fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3)

    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ==================================================================
# REPORT FIGURES
# ==================================================================

def plot_wepl_profile(y_centres, cum_wet, prox_y, dist_y,
                      wet_proximal, wet_distal, filepath):
    """Figure 16 — Cumulative WEPL vs geometric depth along beam axis."""
    # Beam enters at highest Y and travels in -Y direction.  Depth from
    # entrance: max(y_centres) - y_centres.
    y_arr = np.asarray(y_centres)
    wet_arr = np.asarray(cum_wet)
    order = np.argsort(-y_arr)  # beam order: entrance first
    y_ord = y_arr[order]
    wet_ord = wet_arr[order]
    depth = y_ord[0] - y_ord  # geometric depth from entrance (mm)

    tumour_depth_prox = y_ord[0] - prox_y
    tumour_depth_dist = y_ord[0] - dist_y

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depth, wet_ord, color="tab:blue", linewidth=2.0,
            label="Cumulative WEPL")

    # Shade tumour region on depth axis
    ax.axvspan(tumour_depth_prox, tumour_depth_dist,
               alpha=0.18, color="red", label="Tumour (GTVp) region")

    # Mark tumour edges on both axes
    ax.axvline(tumour_depth_prox, color="red", linestyle="--", linewidth=1.0)
    ax.axvline(tumour_depth_dist, color="red", linestyle="--", linewidth=1.0)
    ax.axhline(wet_proximal, color="grey", linestyle=":", linewidth=1.0)
    ax.axhline(wet_distal, color="grey", linestyle=":", linewidth=1.0)

    # Annotate WEPL values at tumour edges
    ax.annotate(f"WEPL(prox)={wet_proximal:.1f} mm",
                xy=(tumour_depth_prox, wet_proximal),
                xytext=(tumour_depth_prox + 8, wet_proximal - 8),
                fontsize=10, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    ax.annotate(f"WEPL(dist)={wet_distal:.1f} mm",
                xy=(tumour_depth_dist, wet_distal),
                xytext=(tumour_depth_dist + 8, wet_distal - 15),
                fontsize=10, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    ax.set_xlabel("Geometric depth from beam entrance (mm)", fontsize=12)
    ax.set_ylabel("Cumulative WEPL (mm water-equivalent)", fontsize=12)
    ax.set_title("Water Equivalent Path Length along central beam axis",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_energy_spread_comparison(selected, weights, spread_profiles_map,
                                  target_prox, target_dist, filepath):
    """Figure 19 — SOBP profiles at different energy spreads, overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(spread_profiles_map)))
    for (spread, profiles), colour in zip(
            sorted(spread_profiles_map.items()), colours):
        depths, combined = build_sobp(selected, weights, profiles)
        flat = compute_flatness(depths, combined, target_prox, target_dist)
        ax.plot(depths, combined, linewidth=1.8, color=colour,
                label=f"{spread*100:.1f}% spread  (flatness={flat:.3f})")

    ax.axvspan(target_prox, target_dist, alpha=0.15, color="red",
               label="Target (WEPL) region")
    ax.set_xlabel("Depth in water (mm)", fontsize=12)
    ax.set_ylabel("Dose (Gy)", fontsize=12)
    ax.set_title("SOBP profiles for varying proton energy spread", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    # Zoom to plateau + transition
    xmax = target_dist + 20.0
    ax.set_xlim(0, xmax)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def load_ct_slice_hu(geom, z_mm):
    """Load full HU image for the CT slice nearest the requested Z (mm)."""
    iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - z_mm)))
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
            return ds.pixel_array.astype(float) * slope + intercept, iz
    raise FileNotFoundError(f"No CT slice near Z={z_mm}")


def plot_dose_on_ct(dose_map, geom, slice_iz, rtstruct_masks,
                    filepath, title="SOBP dose on CT"):
    """Figure 20 — dose overlay on axial CT slice with structure contours."""
    import matplotlib as mpl

    # CT slice HU
    target_z = geom["slice_zs"][slice_iz]
    hu_image, _ = load_ct_slice_hu(geom, target_z)

    # Build 2D dose image for this slice (sum over Z=slice_iz)
    dose_2d = np.zeros((geom["rows"], geom["cols"]), dtype=float)
    for (ix, iy, iz), d in dose_map.items():
        if iz == slice_iz and 0 <= ix < geom["cols"] and 0 <= iy < geom["rows"]:
            dose_2d[iy, ix] = d

    if dose_2d.max() <= 0:
        print(f"  WARNING: dose slice iz={slice_iz} is empty — skipping {filepath}")
        return

    # ImagePositionPatient gives the CENTRE of pixel (0,0).
    # imshow extent maps to image EDGES, so shift by half a pixel.
    dx = geom["dx"]; dy = geom["dy"]
    x_min = geom["x0"] - dx / 2
    x_max = geom["x0"] + (geom["cols"] - 0.5) * dx
    y_min = geom["y0"] - dy / 2
    y_max = geom["y0"] + (geom["rows"] - 0.5) * dy
    extent = [x_min, x_max, y_min, y_max]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(hu_image, cmap="gray", extent=extent, origin="lower",
              vmin=-400, vmax=400, aspect="equal")

    # Dose overlay: mask low values, jet colormap
    dose_max = float(dose_2d.max())
    dose_masked = np.ma.masked_where(dose_2d < 0.05 * dose_max, dose_2d)
    im = ax.imshow(dose_masked, cmap="jet", extent=extent, origin="lower",
                   alpha=0.55, vmin=0, vmax=dose_max, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Dose (Gy)", fontsize=11)

    # Voxel-centre coordinate arrays for contour overlay.
    cont_xs = geom["x0"] + np.arange(geom["cols"]) * dx
    cont_ys = geom["y0"] + np.arange(geom["rows"]) * dy
    contour_colours = {
        "GTVp": "red", "PTV": "purple", "Lung_R": "orange",
        "Heart": "blue", "SpinalCord": "green",
        "Body": "grey", "BODY": "grey", "External": "grey",
    }
    for name, voxels in rtstruct_masks.items():
        # Build a binary mask for this slice
        mask = np.zeros((geom["rows"], geom["cols"]), dtype=bool)
        for (ix, iy, iz) in voxels:
            if iz == slice_iz and 0 <= ix < geom["cols"] and 0 <= iy < geom["rows"]:
                mask[iy, ix] = True
        if not mask.any():
            continue
        colour = contour_colours.get(name, "white")
        ax.contour(cont_xs, cont_ys, mask.astype(int),
                   levels=[0.5], colors=[colour], linewidths=1.3)

    # Flip Y only so the orientation matches VICTORIA's display.
    ax.invert_yaxis()

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ==================================================================
# MAIN
# ==================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "_water"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "_sweep"), exist_ok=True)

    print("=" * 70)
    print("SECTION 2.6: SPREAD OUT BRAGG PEAK (SOBP)")
    print("=" * 70)

    # =================================================================
    # Part 0: Compute WEPL along central beam axis
    # =================================================================
    print("\n--- Part 0: Computing WEPL through patient CT ---")

    geom = load_ct_geometry()
    print(f"  CT grid: {geom['cols']}x{geom['rows']} voxels, "
          f"dx={geom['dx']:.1f} mm, dy={geom['dy']:.1f} mm")

    gtv = load_gtv_bounds()
    print(f"  GTV bounds: X=[{gtv['x_min']:.1f}, {gtv['x_max']:.1f}] "
          f"Y=[{gtv['y_min']:.1f}, {gtv['y_max']:.1f}] "
          f"Z=[{gtv['z_min']:.1f}, {gtv['z_max']:.1f}]")
    print(f"  GTV centre: ({gtv['centre_x']:.1f}, {gtv['centre_y']:.1f}, "
          f"{gtv['centre_z']:.1f})")

    ray_x = gtv["centre_x"]
    ray_z = gtv["centre_z"]
    print(f"  Beam axis: X={ray_x:.1f} mm, Z={ray_z:.1f} mm, direction −Y")

    schneider = parse_schneider_params()
    print(f"  Schneider params loaded ({len(schneider['density_corr'])} "
          f"correction factors)")

    y_centres, cum_wet = compute_wet_along_beam(geom, ray_x, ray_z, schneider)

    wet_proximal, wet_distal, prox_y, dist_y = find_tumour_wet_boundaries(
        y_centres, cum_wet, gtv
    )
    tumour_wet_extent = wet_distal - wet_proximal

    print(f"\n  Tumour proximal edge (Y={prox_y:.1f} mm): "
          f"WET = {wet_proximal:.1f} mm")
    print(f"  Tumour distal edge   (Y={dist_y:.1f} mm): "
          f"WET = {wet_distal:.1f} mm")
    print(f"  Tumour WET extent: {tumour_wet_extent:.1f} mm")

    # Figure 16 — WEPL along central beam axis
    plot_wepl_profile(
        y_centres, cum_wet, prox_y, dist_y, wet_proximal, wet_distal,
        os.path.join(OUTPUT_DIR, "wepl_profile.png"),
    )

    # =================================================================
    # Part 1: Coarse pristine peaks in water + fine grid refinement
    # =================================================================
    print(f"\n--- Part 1a: Coarse pristine peaks ({len(PRISTINE_ENERGIES_COARSE)} energies) ---")
    profiles = run_pristine_peaks(PRISTINE_ENERGIES_COARSE)

    # Identify the WET target region from coarse peaks
    distal_energy_coarse = select_distal_energy(profiles, wet_distal)
    target_prox, target_dist = define_target_region(wet_proximal, wet_distal)
    print(f"\n  Target region: {target_prox:.1f} – {target_dist:.1f} mm (WET-derived)")

    # Determine energy range that covers the target, then add fine grid
    coarse_in_target = []
    for e in sorted(profiles.keys()):
        bp = find_bragg_peak_depth(*profiles[e])
        if target_prox - 5.0 <= bp <= target_dist + 5.0:
            coarse_in_target.append(e)

    if coarse_in_target:
        fine_lo = min(coarse_in_target) - PRISTINE_ENERGY_FINE_MARGIN
        fine_hi = max(coarse_in_target) + PRISTINE_ENERGY_FINE_MARGIN
    else:
        fine_lo = 60.0
        fine_hi = 120.0

    fine_energies = np.arange(fine_lo, fine_hi + 0.1, PRISTINE_ENERGY_FINE_STEP)
    # Remove energies already in the coarse set
    coarse_set = set(round(e, 1) for e in PRISTINE_ENERGIES_COARSE)
    fine_new = sorted(e for e in fine_energies if round(e, 1) not in coarse_set)

    print(f"\n--- Part 1b: Fine grid ({len(fine_new)} new energies, "
          f"{PRISTINE_ENERGY_FINE_STEP} MeV steps, "
          f"{fine_lo:.1f}–{fine_hi:.1f} MeV) ---")
    if fine_new:
        fine_profiles = run_pristine_peaks(fine_new, tag_suffix="_fine")
        profiles.update(fine_profiles)

    # =================================================================
    # Part 1b.5: Bortfeld analytical fit to each MC pristine peak
    # =================================================================
    mc_profiles = {e: (d.copy(), s.copy()) for e, (d, s) in profiles.items()}
    bortfeld_fits = []
    if USE_BORTFELD_FIT and HAS_SCIPY:
        print(f"\n--- Part 1b.5: Fitting Bortfeld model to MC peaks "
              f"(Bortfeld 1997) ---")
        n_fit = len(profiles)
        for i, e in enumerate(sorted(profiles.keys()), 1):
            depths, doses = profiles[e]
            fit = fit_bortfeld_peak(depths, doses, e)
            if fit is None:
                continue
            bortfeld_fits.append(fit)
            if i <= 5 or i == n_fit or i % 10 == 0:
                print(f"  [{i:3d}/{n_fit}] {e:5.1f} MeV: "
                      f"R0={fit['R0']:6.2f} mm, σ={fit['sigma']:4.2f} mm, "
                      f"ε={fit['epsilon']:.3f}, R²={fit['r2']:.4f}")

        # Fit range-energy relation R0(E) = alpha · E^p
        re_result = fit_alpha_p_from_ranges(bortfeld_fits)
        if re_result is not None:
            alpha_fit, p_fit, re_r2 = re_result
            print(f"\n  Range-energy relation R0(E) = α·E^p:")
            print(f"    α (fitted) = {alpha_fit:.5f} mm·MeV^(-p)  "
                  f"(literature: {BORTFELD_ALPHA_LIT_MM})")
            print(f"    p (fitted) = {p_fit:.3f}               "
                  f"(literature: {BORTFELD_P_LITERATURE})")
            print(f"    R² (log-log) = {re_r2:.5f}")

            plot_range_energy_relation(
                bortfeld_fits, alpha_fit, p_fit, re_r2,
                os.path.join(OUTPUT_DIR, "bortfeld_range_energy.png"),
            )

        # Validation plot: MC vs analytical
        plot_bortfeld_validation(
            mc_profiles, bortfeld_fits,
            os.path.join(OUTPUT_DIR, "bortfeld_validation.png"),
        )

        # Write fit summary CSV
        bortfeld_csv = os.path.join(OUTPUT_DIR, "bortfeld_fit_summary.csv")
        with open(bortfeld_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["energy_MeV", "R0_mm", "sigma_mm", "epsilon",
                        "phi", "R2"])
            for fit in sorted(bortfeld_fits, key=lambda x: x["energy"]):
                w.writerow([fit["energy"], round(fit["R0"], 3),
                            round(fit["sigma"], 3),
                            round(fit["epsilon"], 4),
                            fit["phi"], round(fit["r2"], 5)])
        print(f"Saved: {bortfeld_csv}")

        # Replace MC profiles with smooth analytical versions for NNLS
        analytical_profiles = build_analytical_profiles(
            mc_profiles, bortfeld_fits, BORTFELD_FIT_GRID_STEP)
        profiles = analytical_profiles
        mean_r2 = float(np.mean([f["r2"] for f in bortfeld_fits]))
        print(f"  Replaced MC profiles with analytical "
              f"(mean fit R² = {mean_r2:.4f}, depth step = "
              f"{BORTFELD_FIT_GRID_STEP} mm)")

    # =================================================================
    # Range-energy relation from MC Bragg peak positions
    # =================================================================
    if not bortfeld_fits:
        print(f"\n--- Range-energy relation from MC peaks ---")
        mc_fits = []
        for e in sorted(profiles.keys()):
            bp = find_bragg_peak_depth(*profiles[e])
            mc_fits.append({"energy": e, "R0": bp})
        re_result = fit_alpha_p_from_ranges(mc_fits)
        if re_result is not None:
            alpha_fit, p_fit, re_r2 = re_result
            print(f"  R0(E) = α·E^p:")
            print(f"    α (fitted) = {alpha_fit:.5f} mm·MeV^(-p)  "
                  f"(literature: {BORTFELD_ALPHA_LIT_MM})")
            print(f"    p (fitted) = {p_fit:.3f}               "
                  f"(literature: {BORTFELD_P_LITERATURE})")
            print(f"    R² (log-log) = {re_r2:.5f}")
            plot_range_energy_relation(
                mc_fits, alpha_fit, p_fit, re_r2,
                os.path.join(OUTPUT_DIR, "range_energy.png"),
            )

    # Now select distal energy and target energies from the full set
    distal_energy = select_distal_energy(profiles, wet_distal)
    distal_bp = find_bragg_peak_depth(*profiles[distal_energy])
    print(f"\n  Selected distal energy: {distal_energy:.1f} MeV "
          f"(Bragg peak at {distal_bp:.1f} mm, target WET_distal = {wet_distal:.1f} mm)")

    # Select energies whose Bragg peaks fall inside the target region.
    # Include beams peaking up to 2mm beyond the distal edge — these are
    # needed to achieve uniform dose at the distal boundary (the Bragg
    # peak's sharp distal falloff would otherwise leave a gap).
    SELECT_MARGIN_PROX = 2.0  # mm
    SELECT_MARGIN_DIST = 2.0  # mm
    selected = []
    for e in sorted(profiles.keys()):
        bp = find_bragg_peak_depth(*profiles[e])
        if target_prox - SELECT_MARGIN_PROX <= bp <= target_dist + SELECT_MARGIN_DIST:
            selected.append(e)
    print(f"  Selected {len(selected)} energies for SOBP: "
          f"{selected[0]:.1f}–{selected[-1]:.1f} MeV" if selected else
          "  ERROR: no energies in target region")

    if not selected:
        print("  Consider widening PRISTINE_ENERGIES_COARSE or checking WET.")
        return

    # Optimise weights
    print("\n--- Part 1c: Optimising SOBP weights ---")
    weights = optimize_weights(selected, profiles, target_prox, target_dist)
    active_beams = [(e, w) for e, w in zip(selected, weights) if w > 1e-6]
    for e, w in active_beams:
        print(f"  {e:.1f} MeV: weight = {w:.4f}")

    # Build combined SOBP
    depths, combined = build_sobp(selected, weights, profiles)

    # =================================================================
    # Part 2: Quantify and plot (water phantom)
    # =================================================================
    flatness = compute_flatness(depths, combined, target_prox, target_dist)
    print(f"\n--- Part 2: SOBP flatness (SD/mean in target) = {flatness:.4f} ---")

    all_energies = sorted(profiles.keys())
    plot_pristine_peaks_mc(
        all_energies, profiles, target_prox, target_dist,
        os.path.join(OUTPUT_DIR, "pristine_peaks.png"),
    )
    plot_sobp(
        selected, weights, profiles, depths, combined,
        target_prox, target_dist, flatness,
        os.path.join(OUTPUT_DIR, "sobp_water.png"),
    )
    write_summary_csv(
        selected, weights, flatness, profiles,
        os.path.join(OUTPUT_DIR, "sobp_summary.csv"),
    )

    # =================================================================
    # Part 2b: Energy spread tuning (water phantom)
    # =================================================================
    RUN_ENERGY_SPREAD_TUNING = True    # Run energy spread sweep
    best_spread = ENERGY_SPREAD

    if RUN_ENERGY_SPREAD_TUNING:
        print(f"\n--- Part 2b: Energy spread tuning in water ---")
        print(f"  Testing energy spreads: {ENERGY_SPREAD_TEST}")
        best_flatness = flatness
        spread_results = []

        for es in ENERGY_SPREAD_TEST:
            print(f"\n  Energy spread = {es*100:.1f}%:")
            es_profiles = run_pristine_peaks(selected, energy_spread=es,
                                             tag_suffix=f"_es{es:.3f}")
            es_depths, es_combined = build_sobp(selected, weights, es_profiles)
            es_flat = compute_flatness(es_depths, es_combined, target_prox, target_dist)
            print(f"    Flatness = {es_flat:.4f}")
            spread_results.append({"spread": es, "flatness": es_flat,
                                    "profiles": es_profiles})
            if es_flat < best_flatness:
                best_flatness = es_flat
                best_spread = es

        print(f"\n  Best energy spread: {best_spread*100:.1f}% "
              f"(flatness = {best_flatness:.4f})")

        # Figure 19 — overlay of SOBP profiles for each energy spread
        spread_profiles_map = {r["spread"]: r["profiles"] for r in spread_results}
        plot_energy_spread_comparison(
            selected, weights, spread_profiles_map,
            target_prox, target_dist,
            os.path.join(OUTPUT_DIR, "sobp_energy_spread_comparison.png"),
        )

        # If the best spread differs, rebuild the SOBP with it
        if best_spread != ENERGY_SPREAD:
            print(f"  Re-optimising SOBP with energy spread = {best_spread*100:.1f}%")
            best_es_entry = next(r for r in spread_results if r["spread"] == best_spread)
            es_profiles = best_es_entry["profiles"]
            weights = optimize_weights(selected, es_profiles, target_prox, target_dist)
            depths, combined = build_sobp(selected, weights, es_profiles)
            flatness = compute_flatness(depths, combined, target_prox, target_dist)
            print(f"  Updated flatness: {flatness:.4f}")
            profiles.update(es_profiles)

            plot_sobp(
                selected, weights, es_profiles, depths, combined,
                target_prox, target_dist, flatness,
                os.path.join(OUTPUT_DIR, "sobp_water_tuned.png"),
            )
    else:
        print(f"\n--- Part 2b: Energy spread tuning SKIPPED ---")
        print(f"  Using default energy spread: {ENERGY_SPREAD*100:.1f}%")

    WATER_ONLY = False  # Set True to skip patient simulations
    if WATER_ONLY:
        print("\n*** WATER_ONLY mode — skipping patient simulations ***")
        return

    # =================================================================
    # Part 3a: Build structure masks for DVH scoring
    # =================================================================
    print("\n--- Part 3a: Building structure masks from RTStruct ---")
    scoring_masks = build_scoring_masks(geom)
    rt_masks_all = build_rtstruct_masks(geom)  # for contour overlay

    # =================================================================
    # Part 3a.1: Initial patient dose map (narrow cutoff, pre-sweep)
    # =================================================================
    print(f"\n--- Part 3a.1: Initial patient SOBP ({NARROW_BEAM_CUTOFF_X} mm "
          f"cutoff, pre-sweep) ---")
    init_basename = os.path.join("A2_6", "output", "_sweep", "initial_patient")
    init_param = generate_patient_sobp(
        selected, weights, init_basename, PATIENT_HISTORIES,
        cutoff_x=NARROW_BEAM_CUTOFF_X,
        cutoff_y=NARROW_BEAM_CUTOFF_X,
        energy_spread=best_spread,
        output_type="csv",
    )
    run_topas(init_param)
    init_csv = os.path.join(PROJECT_ROOT, init_basename + ".csv")
    init_dose = read_dose_csv(init_csv)

    # Pick the CT slice closest to the GTV centre for the dose overlay
    tumour_z = gtv["centre_z"]
    slice_iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - tumour_z)))
    plot_dose_on_ct(
        init_dose, geom, slice_iz, rt_masks_all,
        os.path.join(OUTPUT_DIR, "initial_sobp_patient.png"),
        title=f"Initial SOBP in patient "
              f"(narrow beam: CutoffX=CutoffY={NARROW_BEAM_CUTOFF_X:.0f} mm)",
    )
    try:
        os.remove(init_param)
    except OSError:
        pass

    # =================================================================
    # Part 3b: CutoffX sweep
    # =================================================================
    print(f"\n--- Part 3b: CutoffX sweep ---")
    print(f"  Values: {BEAM_WIDTH_SWEEP_X} mm")
    print(f"  CutoffY fixed at: {BEAM_POS_CUTOFF_Y:.1f} mm")
    print(f"  Energy spread: {best_spread*100:.1f}%")
    print(f"  Histories per run: {SWEEP_HISTORIES}")

    sweep_x = run_parameter_sweep(
        selected, weights, scoring_masks, SWEEP_HISTORIES,
        sweep_values=BEAM_WIDTH_SWEEP_X,
        sweep_param="cutoff_x",
        fixed_cutoff_y=BEAM_POS_CUTOFF_Y,
        fixed_energy_spread=best_spread,
    )
    print_sweep_table(sweep_x, title="CUTOFF-X SWEEP — DVH METRICS")
    plot_sweep(sweep_x,
               os.path.join(OUTPUT_DIR, "sweep_cutoff_x.png"),
               xlabel="BeamPositionCutoffX (mm)",
               title="CutoffX Sweep — Tumour Coverage vs OAR Dose")

    best_x_result = max(sweep_x,
                        key=lambda r: r["metrics"].get("tumour", {}).get("D95", 0))
    best_cutoff_x = best_x_result["param_value"]
    print(f"  Best CutoffX: {best_cutoff_x:.0f} mm "
          f"(GTVp D95 = {best_x_result['metrics']['tumour']['D95']:.4e})")

    # =================================================================
    # Part 3c: CutoffY sweep (with best CutoffX fixed)
    # =================================================================
    print(f"\n--- Part 3c: CutoffY sweep (CutoffX={best_cutoff_x:.0f} mm fixed) ---")
    print(f"  Values: {BEAM_WIDTH_SWEEP_Y} mm")

    sweep_y = run_parameter_sweep(
        selected, weights, scoring_masks, SWEEP_HISTORIES,
        sweep_values=BEAM_WIDTH_SWEEP_Y,
        sweep_param="cutoff_y",
        fixed_cutoff_x=best_cutoff_x,
        fixed_energy_spread=best_spread,
    )
    print_sweep_table(sweep_y, title="CUTOFF-Y SWEEP — DVH METRICS")
    plot_sweep(sweep_y,
               os.path.join(OUTPUT_DIR, "sweep_cutoff_y.png"),
               xlabel="BeamPositionCutoffY (mm)",
               title="CutoffY Sweep — Tumour Coverage vs OAR Dose")

    best_y_result = max(sweep_y,
                        key=lambda r: r["metrics"].get("tumour", {}).get("D95", 0))
    best_cutoff_y = best_y_result["param_value"]
    print(f"  Best CutoffY: {best_cutoff_y:.0f} mm "
          f"(GTVp D95 = {best_y_result['metrics']['tumour']['D95']:.4e})")

    # =================================================================
    # Part 3d: Energy spread patient test (best 1-2 values from water)
    # =================================================================
    if RUN_ENERGY_SPREAD_TUNING and spread_results:
        competitive = [r for r in spread_results
                       if r["flatness"] <= best_flatness * 1.2]
        es_to_test = sorted(set(r["spread"] for r in competitive))

        if len(es_to_test) > 1:
            print(f"\n--- Part 3d: Energy spread patient test ---")
            print(f"  Testing: {[f'{s*100:.1f}%' for s in es_to_test]}")

            sweep_es = run_parameter_sweep(
                selected, weights, scoring_masks, SWEEP_HISTORIES,
                sweep_values=es_to_test,
                sweep_param="energy_spread",
                fixed_cutoff_x=best_cutoff_x,
                fixed_cutoff_y=best_cutoff_y,
            )
            print_sweep_table(sweep_es, title="ENERGY SPREAD — DVH METRICS")

            best_es_result = max(sweep_es,
                                 key=lambda r: r["metrics"].get("tumour", {}).get("D95", 0))
            best_spread = best_es_result["param_value"]
            print(f"  Best energy spread: {best_spread*100:.1f}%")
        else:
            print(f"\n  Energy spread: {best_spread*100:.1f}% (only competitive value)")
    else:
        print(f"\n--- Part 3d: Energy spread patient test SKIPPED ---")
        print(f"  Using energy spread: {best_spread*100:.1f}%")

    # =================================================================
    # Part 3e: Final high-stat production run
    # =================================================================
    print(f"\n--- Part 3e: Final patient run ---")
    print(f"  CutoffX = {best_cutoff_x:.0f} mm")
    print(f"  CutoffY = {best_cutoff_y:.0f} mm")
    print(f"  Energy spread = {best_spread*100:.1f}%")
    print(f"  Histories = {PATIENT_HISTORIES}")

    # --- DICOM output (for VICTORIA) ---
    patient_basename_dcm = "A2_6/output/dose_sobp_patient"
    patient_file_dcm = generate_patient_sobp(
        selected, weights, patient_basename_dcm, PATIENT_HISTORIES,
        cutoff_x=best_cutoff_x, cutoff_y=best_cutoff_y,
        energy_spread=best_spread, output_type="dicom",
    )
    final_patient = os.path.join(OUTPUT_DIR, "patient_sobp.txt")
    shutil.copy2(patient_file_dcm, final_patient)
    print(f"Saved: {final_patient}")

    print(f"\nRunning patient SOBP simulation — DICOM ({PATIENT_HISTORIES} histories)...")
    run_topas(final_patient)
    print(f"DICOM dose: {os.path.join(OUTPUT_DIR, 'dose_sobp_patient.dcm')}")

    # --- CSV output (for dose overlay plot + DVH scoring) ---
    patient_basename_csv = "A2_6/output/dose_sobp_patient_csv"
    patient_file_csv = generate_patient_sobp(
        selected, weights, patient_basename_csv, PATIENT_HISTORIES,
        cutoff_x=best_cutoff_x, cutoff_y=best_cutoff_y,
        energy_spread=best_spread, output_type="csv",
    )
    final_patient_csv = os.path.join(OUTPUT_DIR, "patient_sobp_csv.txt")
    shutil.copy2(patient_file_csv, final_patient_csv)
    print(f"\nRunning patient SOBP simulation — CSV ({PATIENT_HISTORIES} histories)...")
    run_topas(final_patient_csv)

    # --- Final dose overlay on CT ---
    final_csv_path = os.path.join(PROJECT_ROOT, patient_basename_csv + ".csv")
    if os.path.isfile(final_csv_path):
        final_dose = read_dose_csv(final_csv_path)
        plot_dose_on_ct(
            final_dose, geom, slice_iz, rt_masks_all,
            os.path.join(OUTPUT_DIR, "final_sobp_patient.png"),
            title=f"Final SOBP in patient "
                  f"(CutoffX={best_cutoff_x:.0f}, CutoffY={best_cutoff_y:.0f} mm, "
                  f"spread={best_spread*100:.1f}%)",
        )
    else:
        print(f"  Warning: CSV dose file not found at {final_csv_path}")

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SOBP SUMMARY")
    print("=" * 70)
    print(f"  WET proximal      : {wet_proximal:.1f} mm")
    print(f"  WET distal        : {wet_distal:.1f} mm")
    print(f"  Tumour WET extent : {tumour_wet_extent:.1f} mm")
    print(f"  Distal energy     : {distal_energy:.1f} MeV")
    print(f"  Target region     : {target_prox:.1f} – {target_dist:.1f} mm depth in water")
    print(f"  Flatness (SD/mean): {flatness:.4f}")
    print(f"  Best CutoffX      : {best_cutoff_x:.0f} mm")
    print(f"  Best CutoffY      : {best_cutoff_y:.0f} mm")
    print(f"  Best energy spread: {best_spread*100:.1f}%")
    print(f"  Active beams      : {len(active_beams)}")
    active_str = ", ".join(f"{e:.1f} MeV ({w:.3f})" for e, w in active_beams)
    print(f"  Energies (weight) : {active_str}")
    print(f"\n  Patient TOPAS file: {final_patient}")
    print(f"  Run from project root:  topas A2_6/output/patient_sobp.txt")
    print(f"  Then export DVH from VICTORIA for comparison with photon/single-proton.")


if __name__ == "__main__":
    main()
