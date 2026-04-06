"""
Section 2.8: Pencil Beam Scanning (PBS) Proton Treatment

Compares PBS (modern clinical standard) against passive scattering
SOBP (Section 2.6) on the same patient CT.

Pipeline:
  Part 0 — Load CT geometry, compute WET, get tumour bounds
  Part 1 — Load/run pristine Bragg peaks in water (cache from 2.6 if available)
  Part 2 — Select energy layers and generate spot grid
  Part 3 — Build analytical dose influence matrix (pencil beam model)
  Part 4 — Optimise spot weights (NNLS)
  Part 5 — Generate TOPAS PBS file for Monte Carlo verification
  Part 6 — Run TOPAS + score DVH
  Part 7 — Comparison plots vs passive scattering

Usage:
    cd PHY4004_A2 && python3 A2_8/pbs_proton.py

Outputs (all in A2_8/output/):
    spot_grid.png               spot positions coloured by weight per layer
    dvh_comparison.png          PBS vs passive DVH overlay
    dose_map_comparison.png     2D dose side-by-side on axial slice
    lateral_profiles.png        lateral dose profiles comparison
    summary.csv                 DVH metrics comparison table
    patient_pbs.txt             TOPAS parameter file (PBS)
"""

import os
import sys
import csv
import time
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
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
except Exception as exc:
    raise RuntimeError(f"Matplotlib is required: {exc}")

try:
    from scipy.optimize import nnls as scipy_nnls
    from scipy import sparse
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

# Tumour fallback coordinates (from RTStruct GTV centroid)
TUMOUR_X_FALLBACK = -48.9   # mm
TUMOUR_Y_FALLBACK = 43.1    # mm
TUMOUR_Z_FALLBACK = 0.0     # mm

# PBS spot parameters
SPOT_SIGMA_0 = 4.0          # mm — initial Gaussian spot sigma at nozzle
SPOT_SPACING = 5.0           # mm — spot grid spacing
SPOT_MARGIN = 3.0            # mm — margin beyond GTV for spot placement
LATERAL_CUTOFF = 15.0        # mm — ignore dose beyond this radius from spot axis
MCS_FACTOR = 0.023           # empirical MCS broadening factor (mm^-1 of WET)

# Energy layer spacing (coarser than passive SOBP)
ENERGY_LAYER_STEP = 2.0      # MeV between layers

# Beam geometry (same as Section 2.6)
BEAM_TRANS_X = -46.0    # mm
BEAM_TRANS_Y = 140.0    # mm
BEAM_TRANS_Z = 0.0      # mm
BEAM_ROT_X = -90.0      # deg

# Histories
WATER_HISTORIES = 50000
PBS_HISTORIES = 500000        # total for PBS verification run
ENERGY_SPREAD = 0.01          # 1% fractional spread

# Skip TOPAS if True — produce analytical-only results
SKIP_TOPAS = False

# Passive scattering results directory (for comparison)
PASSIVE_DIR = os.path.join(PROJECT_ROOT, "A2_6", "output")

# RTStruct structure aliases
STRUCTURE_ALIASES = {
    "tumour": ["GTVp", "GTV", "Tumour", "Tumor"],
    "ptv":    ["PTV"],
    "lung_r": ["Lung_R", "Right Lung", "LungR"],
    "heart":  ["Heart"],
    "cord":   ["SpinalCord", "Spinal Cord", "Cord"],
    "body":   ["Body", "External", "BODY", "EXTERNAL"],
}

# Water phantom geometry (must match 2.6 for cached pristine peaks)
WATER_HLX = 50.0
WATER_HLZ = 100.0
WATER_Z_BINS = 800

# Shared colours and display names — consistent across DVH and dose map
STRUCTURE_COLOURS = {
    "tumour": "red", "ptv": "magenta", "lung_r": "deepskyblue",
    "heart": "orange", "cord": "limegreen", "body": "grey",
}
STRUCTURE_DISPLAY = {
    "tumour": "GTVp", "ptv": "PTV", "lung_r": "Right Lung",
    "heart": "Heart", "cord": "Spinal Cord", "body": "Body",
}
# Mapping from RTStruct raw names to internal keys (for contour colouring)
RTSTRUCT_TO_KEY = {
    "GTVp": "tumour", "GTV": "tumour", "Tumour": "tumour", "Tumor": "tumour",
    "PTV": "ptv",
    "Lung_R": "lung_r", "Right Lung": "lung_r", "LungR": "lung_r",
    "Heart": "heart",
    "SpinalCord": "cord", "Spinal Cord": "cord", "Cord": "cord",
    "Body": "body", "BODY": "body", "External": "body", "EXTERNAL": "body",
    "Lung_L": "lung_l",
}
STRUCTURE_COLOURS["lung_l"] = "cyan"
STRUCTURE_DISPLAY["lung_l"] = "Left Lung"


# ==================================================================
# PART 0 — CT GEOMETRY, WET, TUMOUR BOUNDS (reused from 2.6)
# ==================================================================

def load_ct_geometry():
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
        "dx": float(ds0.PixelSpacing[1]),
        "dy": float(ds0.PixelSpacing[0]),
        "dz": abs(zs[1] - zs[0]) if len(zs) > 1 else 1.0,
        "x0": float(ds0.ImagePositionPatient[0]),
        "y0": float(ds0.ImagePositionPatient[1]),
        "z0": zs[0],
        "slice_zs": zs,
    }


def load_gtv_bounds():
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
    roi_map = {int(roi.ROINumber): str(roi.ROIName)
               for roi in ds.StructureSetROISequence}
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
    iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - z_mm)))
    target_z = geom["slice_zs"][iz]
    ct_files = sorted(f for f in os.listdir(CT_DIR) if f.lower().endswith(".dcm"))
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
    ix = int(round((x_mm - geom["x0"]) / geom["dx"]))
    ix = max(0, min(ix, geom["cols"] - 1))
    slope = float(getattr(target_ds, "RescaleSlope", 1.0))
    intercept = float(getattr(target_ds, "RescaleIntercept", 0.0))
    pixel_column = target_ds.pixel_array[:, ix].astype(float)
    hu_column = pixel_column * slope + intercept
    y_centres = geom["y0"] + (np.arange(geom["rows"]) + 0.5) * geom["dy"]
    return y_centres, hu_column


def parse_schneider_params():
    params = {}
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
    hu = np.asarray(hu_array, dtype=float)
    hu_clamped = np.clip(hu, -1000.0, 2995.0)
    sections = schneider["hu_sections"]
    offsets = schneider["offsets"]
    factors = schneider["factors"]
    foffsets = schneider["factor_offsets"]
    dcorr = np.array(schneider["density_corr"])
    base_density = np.zeros_like(hu, dtype=float)
    for i in range(len(offsets)):
        lo = sections[i]
        hi = sections[i + 1]
        mask = (hu_clamped >= lo) & (hu_clamped < hi)
        base_density[mask] = offsets[i] + factors[i] * (foffsets[i] + hu_clamped[mask])
    corr_idx = np.clip(np.round(hu_clamped).astype(int) + 1000, 0, len(dcorr) - 1)
    correction = dcorr[corr_idx]
    return base_density * correction


def compute_wet_along_beam(geom, x_mm, z_mm, schneider):
    y_centres, hu_values = load_ct_hu_column(geom, x_mm, z_mm)
    rsp = hu_to_rsp(hu_values, schneider)
    order = np.argsort(-y_centres)
    y_ordered = y_centres[order]
    rsp_ordered = rsp[order]
    wet_increments = rsp_ordered * geom["dy"]
    cum_wet = np.cumsum(wet_increments)
    cum_wet = np.insert(cum_wet, 0, 0.0)[:-1]
    return y_ordered, cum_wet


def find_tumour_wet_boundaries(y_centres, cum_wet, gtv_bounds):
    proximal_y = gtv_bounds["y_max"]
    distal_y = gtv_bounds["y_min"]
    y_asc = y_centres[::-1]
    wet_asc = cum_wet[::-1]
    wet_proximal = float(np.interp(proximal_y, y_asc, wet_asc))
    wet_distal = float(np.interp(distal_y, y_asc, wet_asc))
    return wet_proximal, wet_distal, proximal_y, distal_y


# ==================================================================
# PART 0b — RTStruct masks (reused from 2.6)
# ==================================================================

def voxel_centres_xy(geom):
    # ImagePositionPatient already gives the centre of pixel (0,0),
    # so pixel (i) centre = x0 + i*dx, not x0 + (i+0.5)*dx.
    xs = geom["x0"] + np.arange(geom["cols"]) * geom["dx"]
    ys = geom["y0"] + np.arange(geom["rows"]) * geom["dy"]
    return xs, ys


def closest_slice_index(z_value, slice_zs):
    return int(np.argmin(np.abs(np.asarray(slice_zs) - z_value)))


def load_ct_slice_hu(geom, z_mm):
    """Load full HU image for the CT slice nearest the requested Z (mm)."""
    iz = closest_slice_index(z_mm, geom["slice_zs"])
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


def contour_to_mask_slice(geom, iz, contour_xy):
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
    rt_masks = build_rtstruct_masks(geom)
    if not rt_masks:
        raise RuntimeError("RTStruct masks required for DVH scoring")
    masks = {}
    for key, aliases in STRUCTURE_ALIASES.items():
        for alias in aliases:
            match = next((n for n in rt_masks if n.lower() == alias.lower()), None)
            if match:
                masks[key] = rt_masks[match]
                break
    print(f"  Scoring masks: {list(masks.keys())}")
    if "tumour" not in masks:
        raise RuntimeError("GTVp mask not found in RTStruct")
    return masks


def read_dose_csv(csv_path):
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


def run_topas(param_file):
    rel = os.path.relpath(param_file, PROJECT_ROOT)
    result = subprocess.run(
        [TOPAS_EXE, rel], cwd=PROJECT_ROOT,
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=3600,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-800:]
        raise RuntimeError(f"TOPAS failed for {rel}:\n{tail}")


# ==================================================================
# PART 1 — PRISTINE PEAKS (load cached or run)
# ==================================================================

def read_water_csv(csv_path):
    # Read first so we can auto-detect the actual number of Y bins in
    # this CSV (may come from 2.6 with a different bin count than our
    # local WATER_Z_BINS). Without this, depths are scaled wrong and
    # every downstream step — layer selection, NNLS, DVH — is invalid.
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
                iy = int(parts[1])
                dose = float(parts[3])
                dose_by_iy[iy] = dose
            except (ValueError, IndexError):
                continue
    if not dose_by_iy:
        return np.array([]), np.array([])

    # Detect bin count from the highest iy index in the file.
    n_bins_detected = max(dose_by_iy.keys()) + 1
    # Snap to the expected value if it's close (handles zero-padded
    # tails); otherwise trust the data.
    if abs(n_bins_detected - WATER_Z_BINS) <= 1:
        n_bins = WATER_Z_BINS
    else:
        n_bins = n_bins_detected
    dy = 2.0 * WATER_HLZ / n_bins

    depths, doses = [], []
    for iy in range(n_bins):
        y_centre = -WATER_HLZ + (iy + 0.5) * dy
        depth = WATER_HLZ - y_centre
        depths.append(depth)
        doses.append(dose_by_iy.get(iy, 0.0))
    order = np.argsort(depths)
    return np.array(depths)[order], np.array(doses)[order]


def generate_water_topas(energy_mev, output_basename, n_histories,
                         energy_spread=None):
    if energy_spread is None:
        energy_spread = ENERGY_SPREAD
    lines = [
        f"# Pristine Bragg peak: {energy_mev:.1f} MeV proton in water",
        "",
        's:Ge/World/Type     = "TsBox"',
        's:Ge/World/Material = "Vacuum"',
        "d:Ge/World/HLX      = 300.0 mm",
        "d:Ge/World/HLY      = 300.0 mm",
        "d:Ge/World/HLZ      = 300.0 mm",
        "",
        's:Ge/WaterPhantom/Type     = "TsBox"',
        's:Ge/WaterPhantom/Parent   = "World"',
        's:Ge/WaterPhantom/Material = "G4_WATER"',
        f"d:Ge/WaterPhantom/HLX      = {WATER_HLX:.1f} mm",
        f"d:Ge/WaterPhantom/HLY      = {WATER_HLZ:.1f} mm",
        f"d:Ge/WaterPhantom/HLZ      = {WATER_HLX:.1f} mm",
        "d:Ge/WaterPhantom/TransX    = 0.0 mm",
        "d:Ge/WaterPhantom/TransY    = 0.0 mm",
        "d:Ge/WaterPhantom/TransZ    = 0.0 mm",
        "i:Ge/WaterPhantom/XBins     = 1",
        f"i:Ge/WaterPhantom/YBins     = {WATER_Z_BINS}",
        "i:Ge/WaterPhantom/ZBins     = 1",
        "",
        f"i:Ts/ShowHistoryCountAtInterval = {max(1000, n_histories // 10)}",
        "i:Ts/NumberOfThreads            = 0",
        'b:Ts/PauseBeforeQuit            = "False"',
        "",
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
        f"u:So/Beam/BeamEnergySpread         = {energy_spread}",
        f"i:So/Beam/NumberOfHistoriesInRun   = {n_histories}",
        's:So/Beam/BeamPositionDistribution = "Flat"',
        's:So/Beam/BeamPositionCutoffShape  = "Ellipse"',
        "d:So/Beam/BeamPositionCutoffX      = 12.0 mm",
        "d:So/Beam/BeamPositionCutoffY      = 12.0 mm",
        's:So/Beam/BeamAngularDistribution  = "Gaussian"',
        "d:So/Beam/BeamAngularCutoffX       = 5.0 deg",
        "d:So/Beam/BeamAngularCutoffY       = 5.0 deg",
        "d:So/Beam/BeamAngularSpreadX       = 0.5 deg",
        "d:So/Beam/BeamAngularSpreadY       = 0.5 deg",
        "",
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


def find_bragg_peak_depth(depths, doses):
    return float(depths[np.argmax(doses)])


def load_or_run_pristine_peaks(energies):
    """Try to load cached pristine peaks from 2.6; run any missing."""
    profiles = {}
    missing = []
    for e in energies:
        cached = os.path.join(PROJECT_ROOT, "A2_6", "output", "_water",
                              f"pristine_{e:.1f}MeV.csv")
        if os.path.isfile(cached):
            depths, doses = read_water_csv(cached)
            profiles[e] = (depths, doses)
        else:
            missing.append(e)

    if profiles:
        print(f"  Loaded {len(profiles)} cached pristine peaks from 2.6")

    if missing:
        print(f"  Running {len(missing)} missing pristine peaks in water ...")
        for i, energy in enumerate(missing, 1):
            basename = os.path.join("A2_8", "output", "_water",
                                    f"pristine_{energy:.1f}MeV")
            print(f"    [{i}/{len(missing)}] {energy:.1f} MeV ... ", end="", flush=True)
            t0 = time.time()
            param_file = generate_water_topas(energy, basename, WATER_HISTORIES)
            run_topas(param_file)
            csv_path = os.path.join(PROJECT_ROOT, basename + ".csv")
            depths, doses = read_water_csv(csv_path)
            profiles[energy] = (depths, doses)
            try:
                os.remove(param_file)
            except OSError:
                pass
            print(f"peak at {depths[np.argmax(doses)]:.1f} mm  ({time.time()-t0:.1f}s)")

    return profiles


# ==================================================================
# PART 2 — ENERGY LAYER SELECTION AND SPOT GRID
# ==================================================================

def select_energy_layers(profiles, wet_proximal, wet_distal):
    """Select energies whose Bragg peaks span the target WET range."""
    bp_depths = {e: find_bragg_peak_depth(*profiles[e]) for e in profiles}
    # Keep energies whose peaks fall within [proximal, distal+small margin]
    margin = 2.0  # mm
    selected = sorted(
        e for e, d in bp_depths.items()
        if wet_proximal - margin <= d <= wet_distal + margin
    )
    if not selected:
        # Fallback: pick closest energies
        by_dist = sorted(profiles.keys(),
                         key=lambda e: abs(bp_depths[e] - (wet_proximal + wet_distal) / 2))
        selected = sorted(by_dist[:5])
    print(f"  Selected {len(selected)} energy layers: "
          f"{selected[0]:.1f} - {selected[-1]:.1f} MeV")
    return selected


def generate_spot_grid(gtv_bounds, energies):
    """
    Generate a spot grid covering the GTV + margin in the XZ plane.
    Each spot is (x, z, energy, layer_index).
    The beam fires along -Y, so spots are positioned in the XZ plane.
    """
    x_lo = gtv_bounds["x_min"] - SPOT_MARGIN
    x_hi = gtv_bounds["x_max"] + SPOT_MARGIN
    z_lo = gtv_bounds["z_min"] - SPOT_MARGIN
    z_hi = gtv_bounds["z_max"] + SPOT_MARGIN

    # Generate grid positions
    xs = np.arange(x_lo, x_hi + 0.5 * SPOT_SPACING, SPOT_SPACING)
    zs = np.arange(z_lo, z_hi + 0.5 * SPOT_SPACING, SPOT_SPACING)

    spots = []
    for li, energy in enumerate(energies):
        for x in xs:
            for z in zs:
                spots.append((float(x), float(z), float(energy), li))

    n_per_layer = len(xs) * len(zs)
    print(f"  Spot grid: {len(xs)}×{len(zs)} = {n_per_layer} spots/layer × "
          f"{len(energies)} layers = {len(spots)} total spots")
    print(f"  X range: [{x_lo:.1f}, {x_hi:.1f}] mm, Z range: [{z_lo:.1f}, {z_hi:.1f}] mm")

    return spots


# ==================================================================
# PART 3 — ANALYTICAL DOSE INFLUENCE MATRIX
# ==================================================================

def build_dose_influence_matrix(spots, profiles, geom, schneider, gtv_bounds):
    """
    Build the dose influence matrix D where D[i,j] = dose to voxel i
    from unit-weight spot j.

    Uses a pencil beam model:
      D = depth_dose(WET) × lateral_gaussian(r)

    where sigma broadens with depth via multiple Coulomb scattering.
    """
    print("  Building dose influence matrix (analytical pencil beam model) ...")
    t0 = time.time()

    xs, ys = voxel_centres_xy(geom)
    n_voxels = geom["cols"] * geom["rows"] * geom["n_slices"]
    n_spots = len(spots)

    # Precompute WET map: for each (ix, iz) column, compute cumulative WET
    # along the beam axis (-Y direction)
    wet_map = {}  # (ix, iz) -> (y_ordered, cum_wet)
    # Only compute for columns near the spot grid
    spot_xs = set(s[0] for s in spots)
    spot_zs = set(s[1] for s in spots)
    x_range = (min(spot_xs) - LATERAL_CUTOFF, max(spot_xs) + LATERAL_CUTOFF)
    z_range = (min(spot_zs) - LATERAL_CUTOFF, max(spot_zs) + LATERAL_CUTOFF)

    ix_range = np.where((xs >= x_range[0]) & (xs <= x_range[1]))[0]
    iz_range = range(geom["n_slices"])

    # For efficiency, compute WET on the central column and reuse
    # (the CT has only 4 slices, so lateral WET variation is small)
    centre_x = gtv_bounds["centre_x"]
    centre_z = gtv_bounds["centre_z"]
    y_central, wet_central = compute_wet_along_beam(geom, centre_x, centre_z, schneider)

    # Build interpolator: Y position -> cumulative WET
    y_asc = y_central[::-1]
    wet_asc = wet_central[::-1]

    # Precompute depth-dose lookup for each energy (interpolated to fine WET grid)
    max_wet = float(np.max(wet_central))
    if max_wet <= 0:
        max_wet = 200.0  # fallback (mm)
    wet_fine = np.linspace(0, max_wet, 2000)
    dd_lookup = {}  # energy -> dose(wet) interpolation array
    for e in set(s[2] for s in spots):
        depths, doses = profiles[e]
        dd_lookup[e] = np.interp(wet_fine, depths, doses)

    # Build sparse matrix using COO format
    rows, cols, vals = [], [], []

    for j, (sx, sz, energy, _) in enumerate(spots):
        dd = dd_lookup[energy]

        # For each voxel within lateral cutoff
        for iz in iz_range:
            vz = geom["slice_zs"][iz]
            dz = abs(vz - sz)
            if dz > LATERAL_CUTOFF:
                continue

            for ix in ix_range:
                vx = xs[ix]
                dx_lat = abs(vx - sx)
                if dx_lat > LATERAL_CUTOFF:
                    continue

                r_sq = (vx - sx)**2 + (vz - sz)**2
                if r_sq > LATERAL_CUTOFF**2:
                    continue

                for iy in range(geom["rows"]):
                    vy = ys[iy]
                    # Get WET at this voxel's Y position
                    wet_here = float(np.interp(vy, y_asc, wet_asc))

                    # Depth-dose component
                    if wet_here < 0:
                        continue
                    wet_idx = int(wet_here / max_wet * (len(wet_fine) - 1))
                    wet_idx = max(0, min(wet_idx, len(wet_fine) - 1))
                    depth_dose = dd[wet_idx]
                    if depth_dose < 1e-15:
                        continue

                    # Lateral Gaussian with depth-dependent sigma (MCS broadening)
                    sigma = np.sqrt(SPOT_SIGMA_0**2 + (MCS_FACTOR * wet_here)**2)
                    lateral = np.exp(-r_sq / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)

                    dose_ij = depth_dose * lateral
                    if dose_ij < 1e-18:
                        continue

                    # Linear voxel index: iz * (rows * cols) + iy * cols + ix
                    vi = iz * (geom["rows"] * geom["cols"]) + iy * geom["cols"] + ix
                    rows.append(vi)
                    cols.append(j)
                    vals.append(dose_ij)

    if HAS_SCIPY:
        D = sparse.csr_matrix((vals, (rows, cols)),
                              shape=(n_voxels, n_spots))
    else:
        # Fallback: dense matrix (may use a lot of memory)
        D = np.zeros((n_voxels, n_spots))
        for r, c, v in zip(rows, cols, vals):
            D[r, c] = v

    elapsed = time.time() - t0
    nnz = len(vals)
    sparsity = 1.0 - nnz / (n_voxels * n_spots) if n_voxels * n_spots > 0 else 0
    print(f"  Matrix: {n_voxels} voxels × {n_spots} spots, "
          f"{nnz} non-zero entries ({sparsity*100:.1f}% sparse), "
          f"{elapsed:.1f}s")

    return D


# ==================================================================
# PART 4 — SPOT WEIGHT OPTIMIZATION
# ==================================================================

def voxel_index(ix, iy, iz, geom):
    return iz * (geom["rows"] * geom["cols"]) + iy * geom["cols"] + ix


def optimize_spot_weights(D, spots, masks, geom):
    """
    Optimise spot weights using NNLS.

    Objective:
      - Target (GTV): uniform prescription dose
      - OAR penalty: minimise dose to heart, cord, lung
      - Smoothness: penalise large weight differences within each layer
      - Distal penalty: suppress hot-spots from highest-energy layers
    """
    print("  Optimising spot weights ...")
    t0 = time.time()

    n_spots = len(spots)

    # Get voxel indices for each structure
    target_idx = sorted(voxel_index(ix, iy, iz, geom)
                        for ix, iy, iz in masks["tumour"])
    oar_indices = {}
    oar_weights = {"heart": 0.15, "cord": 0.25, "lung_r": 0.10, "body": 0.02}
    for name, weight in oar_weights.items():
        if name in masks:
            idx = sorted(voxel_index(ix, iy, iz, geom)
                         for ix, iy, iz in masks[name])
            oar_indices[name] = (idx, weight)

    # Extract dose matrix rows for target
    if HAS_SCIPY and sparse.issparse(D):
        D_target = D[target_idx, :].toarray()
    else:
        D_target = D[target_idx, :]

    # Prescription: uniform dose equal to the mean of maximum dose per spot
    # in the target region
    max_per_spot = np.max(D_target, axis=0)
    prescription = np.mean(max_per_spot[max_per_spot > 0]) if np.any(max_per_spot > 0) else 1.0
    b_target = np.full(len(target_idx), prescription)

    # Give GTV rows a higher weighting so target coverage isn't sacrificed
    # for OAR sparing
    target_boost = 2.0
    A_parts = [target_boost * D_target]
    b_parts = [target_boost * b_target]

    # OAR rows: want zero dose, scaled by penalty weight
    for name, (idx, weight) in oar_indices.items():
        if HAS_SCIPY and sparse.issparse(D):
            D_oar = weight * D[idx, :].toarray()
        else:
            D_oar = weight * D[idx, :]
        A_parts.append(D_oar)
        b_parts.append(np.zeros(len(idx)))

    # Smoothness regularisation within each energy layer
    layer_indices = {}
    for j, (_, _, _, li) in enumerate(spots):
        layer_indices.setdefault(li, []).append(j)

    smooth_weight = 0.05 * prescription
    L_rows = []
    for li in sorted(layer_indices.keys()):
        idx_in_layer = layer_indices[li]
        for k in range(len(idx_in_layer) - 1):
            row = np.zeros(n_spots)
            row[idx_in_layer[k]] = -smooth_weight
            row[idx_in_layer[k + 1]] = smooth_weight
            L_rows.append(row)
    if L_rows:
        A_parts.append(np.array(L_rows))
        b_parts.append(np.zeros(len(L_rows)))

    # Distal fall-off penalty: suppress over-weighting of the highest-energy
    # spots, which tend to create hot-spots beyond the target
    energies_all = sorted(set(s[2] for s in spots))
    if len(energies_all) >= 3:
        e_max = energies_all[-1]
        e_range = e_max - energies_all[0]
        distal_weight = 0.15 * prescription
        for j, (_, _, energy, _) in enumerate(spots):
            frac = (energy - energies_all[0]) / e_range if e_range > 0 else 0
            if frac > 0.7:
                row = np.zeros(n_spots)
                row[j] = distal_weight * frac
                L_rows_d = row
                A_parts.append(L_rows_d.reshape(1, -1))
                b_parts.append(np.zeros(1))

    A = np.vstack(A_parts)
    b = np.concatenate(b_parts)

    # Solve with NNLS
    if HAS_SCIPY:
        w, residual = scipy_nnls(A, b)
    else:
        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        w = np.maximum(w, 0.0)

    # Normalise weights to sum to 1
    if w.sum() > 0:
        w = w / w.sum()

    n_active = np.sum(w > 1e-6)
    elapsed = time.time() - t0
    print(f"  {n_active}/{n_spots} spots active, optimised in {elapsed:.1f}s")

    return w


# ==================================================================
# PART 5 — TOPAS PBS FILE GENERATION
# ==================================================================

def generate_pbs_topas(spots, weights, output_basename, n_histories,
                       output_type="csv"):
    """Generate TOPAS parameter file with one beam source per active spot."""
    active = [(s, w) for s, w in zip(spots, weights) if w > 1e-6]
    if not active:
        raise ValueError("No active spots after optimisation.")

    w_arr = np.array([w for _, w in active])
    w_arr = w_arr / w_arr.sum()
    raw_counts = w_arr * n_histories
    counts = np.floor(raw_counts).astype(int)
    # Ensure at least 1 history per active spot
    counts = np.maximum(counts, 1)
    remainder = int(n_histories - counts.sum())
    if remainder > 0:
        fracs = raw_counts - counts
        for idx in np.argsort(-fracs)[:remainder]:
            counts[idx] += 1

    cutoff = 3.0 * SPOT_SIGMA_0  # lateral cutoff for Gaussian distribution

    lines = [
        "# PBS proton plan — auto-generated by A2_8/pbs_proton.py",
        f"# {len(active)} active spots, {n_histories} total histories",
        "",
        "includeFile = ct_geometry.txt",
        "",
        f"i:Ts/ShowHistoryCountAtInterval = {max(1000, n_histories // 20)}",
        "i:Ts/NumberOfThreads            = 0",
        'b:Ts/PauseBeforeQuit            = "False"',
        "",
    ]

    for i, (((sx, sz, energy, _), weight), nh) in enumerate(
            zip(active, counts), start=1):
        name = f"Spot{i}"
        lines += [
            f"# {name}: E={energy:.1f} MeV, pos=({sx:.1f},{sz:.1f}) mm, "
            f"w={weight:.6f}, N={nh}",
            f's:Ge/{name}/Type   = "Group"',
            f's:Ge/{name}/Parent = "World"',
            f"d:Ge/{name}/TransX = {sx:.1f} mm",
            f"d:Ge/{name}/TransY = {BEAM_TRANS_Y:.1f} mm",
            f"d:Ge/{name}/TransZ = {sz:.1f} mm",
            f"d:Ge/{name}/RotX   = {BEAM_ROT_X:.1f} deg",
            "d:Ge/{}/RotY   = 0.0 deg".format(name),
            "d:Ge/{}/RotZ   = 0.0 deg".format(name),
            "",
            f's:So/{name}/Type                     = "Beam"',
            f's:So/{name}/Component                = "{name}"',
            f's:So/{name}/BeamParticle             = "proton"',
            f"d:So/{name}/BeamEnergy               = {energy:.1f} MeV",
            f"u:So/{name}/BeamEnergySpread         = {ENERGY_SPREAD * 100.0}",
            f"i:So/{name}/NumberOfHistoriesInRun   = {nh}",
            f's:So/{name}/BeamPositionDistribution = "Gaussian"',
            f"d:So/{name}/BeamPositionSpreadX      = {SPOT_SIGMA_0:.1f} mm",
            f"d:So/{name}/BeamPositionSpreadY      = {SPOT_SIGMA_0:.1f} mm",
            f's:So/{name}/BeamPositionCutoffShape  = "Ellipse"',
            f"d:So/{name}/BeamPositionCutoffX      = {cutoff:.1f} mm",
            f"d:So/{name}/BeamPositionCutoffY      = {cutoff:.1f} mm",
            's:So/{}/BeamAngularDistribution  = "None"'.format(name),
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
    print(f"  Written TOPAS PBS file: {filepath} ({len(active)} spots)")
    return filepath


# ==================================================================
# PART 6 — DVH SCORING
# ==================================================================

def compute_analytical_dose(D, weights, geom):
    """Compute dose distribution from the analytical model."""
    if HAS_SCIPY and sparse.issparse(D):
        dose_flat = D.dot(weights)
    else:
        dose_flat = D @ weights

    dose_map = {}
    n_cols = geom["cols"]
    n_rows = geom["rows"]
    for iz in range(geom["n_slices"]):
        for iy in range(n_rows):
            for ix in range(n_cols):
                vi = iz * (n_rows * n_cols) + iy * n_cols + ix
                if dose_flat[vi] > 0:
                    dose_map[(ix, iy, iz)] = float(dose_flat[vi])
    return dose_map


# ==================================================================
# PART 7 — COMPARISON PLOTS
# ==================================================================

def compute_dvh_curve(dose_map, voxels, n_bins=200):
    """Compute cumulative DVH curve: (dose_values, volume_percent)."""
    doses = np.array([dose_map.get(v, 0.0) for v in voxels])
    if len(doses) == 0 or np.max(doses) == 0:
        return np.array([0]), np.array([100])
    d_max = np.max(doses)
    bins = np.linspace(0, d_max * 1.05, n_bins)
    vol_pct = np.array([100.0 * np.sum(doses >= d) / len(doses) for d in bins])
    return bins, vol_pct


def plot_dvh_comparison(pbs_dose, passive_dose, masks, filepath):
    """DVH for PBS plan. Dose axis in % of plan maximum."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Find dose maximum for normalisation
    all_doses = list(pbs_dose.values())
    dose_max = max(all_doses) if all_doses else 1.0
    if dose_max <= 0:
        dose_max = 1.0

    for name in ["tumour", "ptv", "heart", "cord", "lung_r"]:
        if name not in masks:
            continue
        voxels = sorted(masks[name])
        color = STRUCTURE_COLOURS.get(name, "black")
        label = STRUCTURE_DISPLAY.get(name, name)

        d_pbs, v_pbs = compute_dvh_curve(pbs_dose, voxels)
        ax.plot(d_pbs / dose_max * 100.0, v_pbs, color=color,
                linewidth=2.0, linestyle="-", label=label)

    ax.set_xlabel("Dose (% of plan max)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title("PBS Dose-Volume Histogram", fontsize=14)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10,
              frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_spot_grid(spots, weights, gtv_bounds, filepath):
    """Visualise spot positions coloured by weight for a few energy layers."""
    from matplotlib.gridspec import GridSpec

    energies = sorted(set(s[2] for s in spots))

    # Pick 3 layers that have the highest total weight.
    layer_total_w = {}
    for i, s in enumerate(spots):
        layer_total_w.setdefault(s[2], 0.0)
        layer_total_w[s[2]] += weights[i]
    ranked = sorted(layer_total_w.keys(), key=lambda e: -layer_total_w[e])
    distal = energies[-1]
    pick = []
    for e in ranked:
        if len(pick) >= 2:
            break
        if e != distal:
            pick.append(e)
    pick.append(distal)
    pick = sorted(pick)
    if len(pick) < 3 and len(energies) >= 3:
        pick = [energies[0], energies[len(energies) // 2], energies[-1]]

    n = len(pick)
    # GridSpec with a narrow column for the colorbar
    fig = plt.figure(figsize=(5.2 * n + 0.6, 5.5))
    gs = GridSpec(1, n + 1, figure=fig, width_ratios=[1] * n + [0.05],
                  wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[0, n])

    w_max = max(weights) if max(weights) > 0 else 1
    norm = Normalize(vmin=0, vmax=w_max)
    cmap = cm.YlOrRd

    for ax, energy in zip(axes, pick):
        layer_spots = [(s, weights[i]) for i, s in enumerate(spots) if s[2] == energy]
        xs = [s[0] for s, _ in layer_spots]
        zs = [s[1] for s, _ in layer_spots]
        ws = [w for _, w in layer_spots]

        ax.scatter(xs, zs, c=ws, cmap=cmap, norm=norm, s=80,
                   edgecolors="black", linewidth=0.5, zorder=3)
        ax.add_patch(plt.Rectangle(
            (gtv_bounds["x_min"], gtv_bounds["z_min"]),
            gtv_bounds["x_max"] - gtv_bounds["x_min"],
            gtv_bounds["z_max"] - gtv_bounds["z_min"],
            fill=False, edgecolor="red", linewidth=2, linestyle="--",
            label="GTV"))
        layer_w = layer_total_w.get(energy, 0.0)
        ax.set_xlabel("X (mm)", fontsize=11)
        ax.set_ylabel("Z (mm)", fontsize=11)
        ax.set_title(f"E = {energy:.1f} MeV  (Σw = {layer_w:.3f})", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label="Spot weight")
    fig.suptitle("PBS Spot Grid — Weight Distribution", fontsize=14)
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_dose_map_comparison(pbs_dose, passive_dose, geom, gtv_bounds,
                             filepath, rtstruct_masks=None):
    """Side-by-side 2D dose maps on CT with RTStruct contours."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    iz = closest_slice_index(gtv_bounds["centre_z"], geom["slice_zs"])
    target_z = geom["slice_zs"][iz]

    hu_image, _ = load_ct_slice_hu(geom, target_z)

    # ImagePositionPatient gives the CENTRE of pixel (0,0).
    # imshow extent maps to image EDGES, so shift by half a pixel.
    dx, dy = geom["dx"], geom["dy"]
    x_min = geom["x0"] - dx / 2
    x_max = geom["x0"] + (geom["cols"] - 0.5) * dx
    y_min = geom["y0"] - dy / 2
    y_max = geom["y0"] + (geom["rows"] - 0.5) * dy
    extent = [x_min, x_max, y_min, y_max]

    # Voxel-centre coordinate arrays for contour.
    cont_xs = geom["x0"] + np.arange(geom["cols"]) * dx
    cont_ys = geom["y0"] + np.arange(geom["rows"]) * dy

    def dose_to_2d(dose_map, iz):
        arr = np.zeros((geom["rows"], geom["cols"]))
        for iy in range(geom["rows"]):
            for ix in range(geom["cols"]):
                arr[iy, ix] = dose_map.get((ix, iy, iz), 0.0)
        return arr

    pbs_2d = dose_to_2d(pbs_dose, iz)

    n_panels = 2 if passive_dose else 1
    fig = plt.figure(figsize=(7.5 * n_panels + 0.8, 7.5))
    gs = GridSpec(1, n_panels + 1, figure=fig,
                  width_ratios=[1] * n_panels + [0.04], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cax = fig.add_subplot(gs[0, n_panels])

    vmax = float(np.max(pbs_2d))
    pas_2d = None
    if passive_dose:
        pas_2d = dose_to_2d(passive_dose, iz)
        vmax = max(vmax, float(np.max(pas_2d)))
    if vmax <= 0:
        vmax = 1.0

    # Map RTStruct names to the shared colour palette
    contour_colours = {}
    contour_display = {}
    for raw_name, key in RTSTRUCT_TO_KEY.items():
        contour_colours[raw_name] = STRUCTURE_COLOURS.get(key, "white")
        contour_display[raw_name] = STRUCTURE_DISPLAY.get(key, raw_name)

    drawn_structures = []

    def draw_panel(ax, dose_2d, title):
        ax.imshow(hu_image, cmap="gray", extent=extent, origin="lower",
                  vmin=-400, vmax=400, aspect="equal")
        dose_masked = np.ma.masked_where(dose_2d < 0.05 * vmax, dose_2d)
        im = ax.imshow(dose_masked, cmap="jet", extent=extent, origin="lower",
                       alpha=0.55, vmin=0, vmax=vmax, aspect="equal")
        if rtstruct_masks:
            for name, voxels in rtstruct_masks.items():
                mask = np.zeros((geom["rows"], geom["cols"]), dtype=bool)
                for (ix2, iy2, iz2) in voxels:
                    if iz2 == iz and 0 <= ix2 < geom["cols"] and 0 <= iy2 < geom["rows"]:
                        mask[iy2, ix2] = True
                if not mask.any():
                    continue
                colour = contour_colours.get(name, "white")
                ax.contour(cont_xs, cont_ys, mask.astype(int),
                           levels=[0.5], colors=[colour], linewidths=1.3)
                if name not in [s for s, _ in drawn_structures]:
                    drawn_structures.append((name, colour))
        ax.invert_yaxis()
        ax.set_xlabel("X (mm)", fontsize=12)
        ax.set_ylabel("Y (mm)", fontsize=12)
        ax.set_title(title, fontsize=13)
        return im

    im = draw_panel(axes[0], pbs_2d, "PBS")
    if passive_dose and pas_2d is not None:
        draw_panel(axes[1], pas_2d, "Passive Scattering")

    fig.colorbar(im, cax=cax, label="Dose (Gy)")

    # Structure legend
    if drawn_structures:
        legend_order = ["GTVp", "PTV", "Lung_R", "Heart",
                        "SpinalCord", "Lung_L", "Body", "BODY"]
        ordered = sorted(drawn_structures,
                         key=lambda x: legend_order.index(x[0])
                         if x[0] in legend_order else 99)
        patches = [Patch(facecolor="none", edgecolor=c, linewidth=1.5,
                         label=contour_display.get(n, n))
                   for n, c in ordered]
        axes[0].legend(handles=patches, loc="upper right", fontsize=8,
                       frameon=True, framealpha=0.8)

    fig.suptitle(f"Dose Distribution — Axial Slice Z={target_z:.1f} mm",
                 fontsize=14)
    fig.savefig(filepath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_lateral_profiles(pbs_dose, passive_dose, geom, gtv_bounds, filepath):
    """Lateral dose profiles (along X) at tumour centre depth."""
    xs, ys = voxel_centres_xy(geom)
    iz = closest_slice_index(gtv_bounds["centre_z"], geom["slice_zs"])
    # Find iy closest to tumour centre Y
    iy = int(np.argmin(np.abs(ys - gtv_bounds["centre_y"])))

    pbs_profile = np.array([pbs_dose.get((ix, iy, iz), 0.0) for ix in range(geom["cols"])])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, pbs_profile, "b-", linewidth=2, label="PBS")

    if passive_dose:
        pas_profile = np.array([passive_dose.get((ix, iy, iz), 0.0)
                                for ix in range(geom["cols"])])
        ax.plot(xs, pas_profile, "r--", linewidth=1.5, label="Passive")

    # GTV extent
    ax.axvline(gtv_bounds["x_min"], color="grey", linestyle=":", alpha=0.7)
    ax.axvline(gtv_bounds["x_max"], color="grey", linestyle=":", alpha=0.7,
               label="GTV edge")

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Dose (arb.)", fontsize=12)
    ax.set_title("Lateral Dose Profile at Tumour Centre", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=400)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def write_summary_csv(pbs_metrics, passive_metrics, filepath):
    """Write DVH metrics comparison table."""
    structures = sorted(set(list(pbs_metrics.keys()) +
                            list(passive_metrics.keys() if passive_metrics else [])))
    rows = []
    for name in structures:
        row = {"Structure": name}
        if name in pbs_metrics:
            for k, v in pbs_metrics[name].items():
                row[f"PBS_{k}"] = f"{v:.6e}"
        if passive_metrics and name in passive_metrics:
            for k, v in passive_metrics[name].items():
                row[f"Passive_{k}"] = f"{v:.6e}"
        rows.append(row)

    if rows:
        keys = list(rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Saved: {filepath}")


def print_metrics_table(pbs_metrics, passive_metrics):
    """Print side-by-side DVH metrics."""
    print("\n" + "=" * 90)
    print("DVH METRICS COMPARISON: PBS vs PASSIVE SCATTERING")
    print("=" * 90)
    header = f"{'Structure':<12s}  {'D95':>10s}  {'D50':>10s}  {'Mean':>10s}  {'Max':>10s}"
    for technique, metrics in [("PBS", pbs_metrics), ("Passive", passive_metrics)]:
        if not metrics:
            continue
        print(f"\n--- {technique} ---")
        print(header)
        print("-" * 60)
        for name in sorted(metrics.keys()):
            m = metrics[name]
            print(f"{name:<12s}  {m['D95']:10.4e}  {m['D50']:10.4e}  "
                  f"{m['mean']:10.4e}  {m['max']:10.4e}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("SECTION 2.8: PENCIL BEAM SCANNING (PBS) PROTON TREATMENT")
    print("=" * 70)

    # ---- Part 0: Load CT, WET, tumour bounds ----
    print("\nPart 0: Loading CT geometry and computing WET ...")
    geom = load_ct_geometry()
    print(f"  CT: {geom['cols']}×{geom['rows']}×{geom['n_slices']} voxels, "
          f"spacing ({geom['dx']:.2f}, {geom['dy']:.2f}, {geom['dz']:.2f}) mm")

    gtv = load_gtv_bounds()
    print(f"  GTV centre: ({gtv['centre_x']:.1f}, {gtv['centre_y']:.1f}, "
          f"{gtv['centre_z']:.1f}) mm")
    print(f"  GTV bounds: X[{gtv['x_min']:.1f}, {gtv['x_max']:.1f}], "
          f"Y[{gtv['y_min']:.1f}, {gtv['y_max']:.1f}], "
          f"Z[{gtv['z_min']:.1f}, {gtv['z_max']:.1f}]")

    schneider = parse_schneider_params()
    y_centres, cum_wet = compute_wet_along_beam(
        geom, gtv["centre_x"], gtv["centre_z"], schneider)
    wet_prox, wet_dist, y_prox, y_dist = find_tumour_wet_boundaries(
        y_centres, cum_wet, gtv)
    print(f"  WET boundaries: proximal={wet_prox:.1f} mm, distal={wet_dist:.1f} mm")

    masks = build_scoring_masks(geom)
    rt_masks = build_rtstruct_masks(geom)   # raw names for contour plots

    # ---- Part 1: Pristine peaks ----
    print("\nPart 1: Loading/running pristine Bragg peaks ...")
    energy_range = np.arange(60.0, 120.0, ENERGY_LAYER_STEP)
    profiles = load_or_run_pristine_peaks(sorted(energy_range.tolist()))

    # ---- Part 2: Energy layers + spot grid ----
    print("\nPart 2: Selecting energy layers and generating spot grid ...")
    energies = select_energy_layers(profiles, wet_prox, wet_dist)
    spots = generate_spot_grid(gtv, energies)

    # ---- Part 3: Dose influence matrix ----
    print("\nPart 3: Building analytical dose influence matrix ...")
    D = build_dose_influence_matrix(spots, profiles, geom, schneider, gtv)

    # ---- Part 4: Optimise spot weights ----
    print("\nPart 4: Optimising spot weights ...")
    weights = optimize_spot_weights(D, spots, masks, geom)

    # ---- Part 5: Generate TOPAS file ----
    print("\nPart 5: Generating TOPAS PBS parameter file ...")
    csv_basename = os.path.join("A2_8", "output", "dose_pbs_patient")
    pbs_topas = generate_pbs_topas(spots, weights, csv_basename,
                                   PBS_HISTORIES, output_type="csv")

    # Also generate a DICOM version for VICTORIA
    dicom_basename = os.path.join("A2_8", "output", "dose_pbs_patient_dicom")
    generate_pbs_topas(spots, weights, dicom_basename,
                       PBS_HISTORIES, output_type="dicom")

    # ---- Part 6: Run TOPAS + score ----
    pbs_dose = None
    pbs_metrics = None

    if not SKIP_TOPAS:
        print("\nPart 6: Running TOPAS PBS simulation ...")
        try:
            run_topas(pbs_topas)
            csv_path = os.path.join(PROJECT_ROOT, csv_basename + ".csv")
            pbs_dose = read_dose_csv(csv_path)
            pbs_metrics = compute_dvh_metrics(pbs_dose, masks)
            print("  TOPAS PBS simulation complete.")
        except Exception as exc:
            print(f"  TOPAS failed: {exc}")
            print("  Falling back to analytical dose model.")

    # If TOPAS didn't run or failed, use analytical model
    if pbs_dose is None:
        print("\nPart 6 (analytical): Computing dose from pencil beam model ...")
        pbs_dose = compute_analytical_dose(D, weights, geom)

    # Normalise analytical dose to 60 Gy prescription at GTV mean
    PRESCRIPTION_GY = 60.0
    gtv_voxels = sorted(masks.get("tumour", []))
    if gtv_voxels:
        gtv_doses = np.array([pbs_dose.get(v, 0.0) for v in gtv_voxels])
        gtv_mean = np.mean(gtv_doses)
        if gtv_mean > 0:
            scale = PRESCRIPTION_GY / gtv_mean
            pbs_dose = {k: v * scale for k, v in pbs_dose.items()}
            print(f"  Normalised dose: GTV mean → {PRESCRIPTION_GY:.0f} Gy "
                  f"(scale factor = {scale:.2f})")

    pbs_metrics = compute_dvh_metrics(pbs_dose, masks)

    # Passive scattering comparison is handled in Section 2.6; this
    # section focuses on the PBS plan alone.
    passive_dose = None
    passive_metrics = None

    # ---- Part 7: Plots and summary ----
    print("\nPart 7: Generating comparison plots ...")

    plot_spot_grid(spots, weights, gtv,
                   os.path.join(OUTPUT_DIR, "spot_grid.png"))

    plot_dvh_comparison(pbs_dose, passive_dose, masks,
                        os.path.join(OUTPUT_DIR, "dvh_comparison.png"))

    plot_dose_map_comparison(pbs_dose, passive_dose, geom, gtv,
                            os.path.join(OUTPUT_DIR, "dose_map_comparison.png"),
                            rtstruct_masks=rt_masks)

    plot_lateral_profiles(pbs_dose, passive_dose, geom, gtv,
                          os.path.join(OUTPUT_DIR, "lateral_profiles.png"))

    write_summary_csv(pbs_metrics, passive_metrics,
                      os.path.join(OUTPUT_DIR, "summary.csv"))

    print_metrics_table(pbs_metrics, passive_metrics)

    print("\n" + "=" * 70)
    print("DONE. Outputs in A2_8/output/")
    print("=" * 70)
    n_active = int(np.sum(weights > 1e-6))
    print(f"  Active spots: {n_active}/{len(spots)}")
    print(f"  TOPAS PBS file (CSV):   {csv_basename}.txt")
    print(f"  TOPAS PBS file (DICOM): {dicom_basename}.txt")
    if SKIP_TOPAS:
        print("  Note: TOPAS was skipped. Results are from the analytical model only.")
        print("  Set SKIP_TOPAS = False and run the .txt files with TOPAS for MC verification.")


if __name__ == "__main__":
    main()
