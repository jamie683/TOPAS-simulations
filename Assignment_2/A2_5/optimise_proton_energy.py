"""
Section 2.5: Proton Beam Energy Optimisation

Finds the proton beam energy that best places the Bragg peak at the
tumour by sweeping over energies, running TOPAS for each, and scoring
the resulting dose distributions.

Workflow:
    1. Coarse energy sweep  (60--230 MeV, 10 MeV steps)
    2. Local refinement     (±8 MeV around the best, 2 MeV steps)
    3. Select the optimal energy and write the final TOPAS file

Optimisation metric (simple, physics-based):
    score = tumour_D95
    D95 peaks naturally when the Bragg peak is at the tumour —
    too low an energy gives falloff/zero dose, too high gives only
    plateau dose. No directional overshoot penalty needed.

Usage:
    cd PHY4004_A2 && python3 A2_5/optimise_proton_energy.py

Outputs (all in A2_5/output/):
    energy_sweep_summary.csv        all tested energies + metrics
    proton_energy_vs_metrics.png    tumour metrics vs energy
    proton_depth_dose.png           depth-dose profiles at key energies
    optimised_proton_<E>MeV.txt     final TOPAS file at optimal energy
"""

import os
import sys
import csv
import time
import subprocess
from collections import defaultdict

import numpy as np

try:
    import pydicom
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydicom"])
    import pydicom

try:
    from matplotlib.path import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(f"Matplotlib is required: {exc}")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CT_DIR = os.path.join(PROJECT_ROOT, "CTData")
TOPAS_EXE = "/home/jamie/shellScripts/topas"

# Energy sweep ranges (MeV)
ENERGY_COARSE_MIN = 60.0
ENERGY_COARSE_MAX = 200.0
ENERGY_COARSE_STEP = 10.0
ENERGY_REFINE_HALF = 10.0   # ±10 MeV around best coarse energy
ENERGY_REFINE_STEP = 1.0

# TOPAS histories
SWEEP_HISTORIES = 5000
REFINE_HISTORIES = 30000
PROD_HISTORIES = 100000

# Fixed beam geometry from dose_scoring_proton.txt
BEAM_TRANS_X = -46.0    # mm
BEAM_TRANS_Y = 140.0    # mm
BEAM_TRANS_Z = 0.0      # mm
BEAM_ROT_X = -90.0      # deg
BEAM_ROT_Y = 0.0        # deg
BEAM_ROT_Z = 0.0        # deg
BEAM_POS_CUTOFF_X = 20.0  # mm (default; overridden by lateral sweep)
BEAM_POS_CUTOFF_Y = 20.0  # mm

# Lateral beam width sweep (CutoffX values to test, mm)
LATERAL_SWEEP_VALUES = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
LATERAL_SWEEP_HISTORIES = 30000
BEAM_ANG_CUTOFF = 5.0     # deg
BEAM_ANG_SPREAD = 0.5     # deg

# Tumour location (from verified beam alignment)
TUMOUR_X = -46.0   # mm  (matches TransX in the reference file)
TUMOUR_Y = 43.0    # mm  (from RTStruct GTV centroid)
TUMOUR_Z = 0.0     # mm
TUMOUR_RADIUS = 15.0  # mm

# Overshoot penalty weight in the combined score
OVERSHOOT_WEIGHT = 1.2

# RTStruct structure name aliases
STRUCTURE_ALIASES = {
    "tumour": ["GTVp", "GTV", "Tumour", "Tumor"],
    "lung_r": ["Lung_R", "Right Lung", "LungR"],
    "heart":  ["Heart"],
    "cord":   ["SpinalCord", "Spinal Cord", "Cord"],
    "body":   ["Body", "External", "BODY", "EXTERNAL"],
}

# Fallback box masks if RTStruct parsing fails
FALLBACK_BOXES = {
    "lung_r": {"x": (-20.0, 150.0), "y": (-80.0, 40.0), "z": (-30.0, 30.0)},
    "heart":  {"x": (-55.0, 20.0),  "y": (-15.0, 55.0), "z": (-30.0, 30.0)},
    "cord":   {"x": (-12.0, 12.0),  "y": (-95.0, -55.0), "z": (-25.0, 25.0)},
}


# ------------------------------------------------------------------
# CT Geometry Loading
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
        "dx": float(ds0.PixelSpacing[1]),
        "dy": float(ds0.PixelSpacing[0]),
        "dz": abs(zs[1] - zs[0]) if len(zs) > 1 else 1.0,
        "x0": float(ds0.ImagePositionPatient[0]),
        "y0": float(ds0.ImagePositionPatient[1]),
        "z0": zs[0],
        "slice_zs": zs,
    }


def voxel_centres_xy(geom):
    xs = geom["x0"] + (np.arange(geom["cols"]) + 0.5) * geom["dx"]
    ys = geom["y0"] + (np.arange(geom["rows"]) + 0.5) * geom["dy"]
    return xs, ys


# ------------------------------------------------------------------
# Structure Mask Building
# ------------------------------------------------------------------
def build_sphere_mask(geom, cx, cy, cz, radius):
    """Build a set of (ix, iy, iz) voxel indices inside a sphere."""
    voxels = set()
    for ix in range(geom["cols"]):
        x = geom["x0"] + (ix + 0.5) * geom["dx"]
        if abs(x - cx) > radius + geom["dx"]:
            continue
        for iy in range(geom["rows"]):
            y = geom["y0"] + (iy + 0.5) * geom["dy"]
            if abs(y - cy) > radius + geom["dy"]:
                continue
            for iz in range(geom["n_slices"]):
                z = geom["z0"] + (iz + 0.5) * geom["dz"]
                if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2:
                    voxels.add((ix, iy, iz))
    return voxels


def build_box_mask(geom, x_range, y_range, z_range, exclude=None):
    """Build a set of voxel indices inside a rectangular box."""
    voxels = set()
    exclude = exclude or set()
    for ix in range(geom["cols"]):
        x = geom["x0"] + (ix + 0.5) * geom["dx"]
        if not (x_range[0] <= x <= x_range[1]):
            continue
        for iy in range(geom["rows"]):
            y = geom["y0"] + (iy + 0.5) * geom["dy"]
            if not (y_range[0] <= y <= y_range[1]):
                continue
            for iz in range(geom["n_slices"]):
                z = geom["z0"] + (iz + 0.5) * geom["dz"]
                if not (z_range[0] <= z <= z_range[1]):
                    continue
                key = (ix, iy, iz)
                if key not in exclude:
                    voxels.add(key)
    return voxels


def build_full_grid_mask(geom):
    voxels = set()
    for ix in range(geom["cols"]):
        for iy in range(geom["rows"]):
            for iz in range(geom["n_slices"]):
                voxels.add((ix, iy, iz))
    return voxels


def find_rtstruct_file():
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(CT_DIR, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") == "RTSTRUCT":
            return fpath
    return None


def rtstruct_roi_name_map(ds):
    out = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            out[int(roi.ROINumber)] = str(roi.ROIName)
    return out


def closest_slice_index(z_value, slice_zs):
    return int(np.argmin(np.abs(np.asarray(slice_zs) - z_value)))


def contour_to_mask_slice(geom, iz, contour_xy):
    xs, ys = voxel_centres_xy(geom)
    poly = np.asarray(contour_xy, dtype=float)
    path = Path(poly)

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


def build_rtstruct_masks(geom, rtstruct_path):
    ds = pydicom.dcmread(rtstruct_path)
    roi_map = rtstruct_roi_name_map(ds)
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


def build_distal_mask(geom, tumour_mask):
    """
    Distal region: voxels beyond the tumour in the beam direction.

    The beam travels from Y=+200 toward the tumour at Y~+43, so it
    propagates in the -Y direction. "Distal" means further along -Y,
    i.e. Y < tumour's lowest Y boundary.

    We restrict the lateral extent to the tumour's X/Z bounding box
    so we only capture dose that would have hit the tumour if the
    Bragg peak were correctly placed.
    """
    # Find the tumour bounding box in physical coordinates
    tumour_xs, tumour_ys, tumour_zs = [], [], []
    for ix, iy, iz in tumour_mask:
        tumour_xs.append(geom["x0"] + (ix + 0.5) * geom["dx"])
        tumour_ys.append(geom["y0"] + (iy + 0.5) * geom["dy"])
        tumour_zs.append(geom["z0"] + (iz + 0.5) * geom["dz"])

    x_min, x_max = min(tumour_xs), max(tumour_xs)
    z_min, z_max = min(tumour_zs), max(tumour_zs)
    y_distal_limit = min(tumour_ys) - geom["dy"]  # just beyond the tumour

    distal = set()
    for ix in range(geom["cols"]):
        x = geom["x0"] + (ix + 0.5) * geom["dx"]
        if not (x_min <= x <= x_max):
            continue
        for iy in range(geom["rows"]):
            y = geom["y0"] + (iy + 0.5) * geom["dy"]
            if y > y_distal_limit:
                continue
            for iz in range(geom["n_slices"]):
                z = geom["z0"] + (iz + 0.5) * geom["dz"]
                if not (z_min <= z <= z_max):
                    continue
                key = (ix, iy, iz)
                if key not in tumour_mask:
                    distal.add(key)
    return distal


def build_structure_masks(geom):
    """Build voxel masks for tumour, OARs, distal region, and body."""
    masks = {}
    info = {"source": "fallback"}

    # Try RTStruct for all structures including tumour
    rtstruct_path = find_rtstruct_file()
    rt_masks = build_rtstruct_masks(geom, rtstruct_path) if rtstruct_path else {}

    if rt_masks:
        info["source"] = f"RTSTRUCT ({os.path.basename(rtstruct_path)})"
        info["structures"] = sorted(rt_masks.keys())
        for key, aliases in STRUCTURE_ALIASES.items():
            for alias in aliases:
                match = next((n for n in rt_masks if n.lower() == alias.lower()), None)
                if match:
                    masks[key] = set(rt_masks[match])
                    break

    # Tumour: prefer RTStruct contour, fallback to sphere
    if "tumour" not in masks or not masks["tumour"]:
        print("  WARNING: GTVp not found in RTStruct — using fallback sphere mask")
        tumour = build_sphere_mask(geom, TUMOUR_X, TUMOUR_Y, TUMOUR_Z, TUMOUR_RADIUS)
        if not tumour:
            raise RuntimeError("Tumour mask is empty — check coordinates vs CT geometry.")
        masks["tumour"] = tumour
    else:
        print(f"  Tumour mask from RTStruct: {len(masks['tumour'])} voxels")

    # Fallback box masks for missing structures
    for key in ("lung_r", "heart", "cord"):
        if key not in masks or not masks[key]:
            box = FALLBACK_BOXES[key]
            masks[key] = build_box_mask(geom, box["x"], box["y"], box["z"],
                                        exclude=masks["tumour"])

    if "body" not in masks or not masks["body"]:
        masks["body"] = build_full_grid_mask(geom)

    # Distal region for overshoot scoring
    masks["distal"] = build_distal_mask(geom, masks["tumour"])

    print(f"  Mask source: {info['source']}")
    print(f"  Tumour voxels: {len(masks['tumour'])}")
    print(f"  Distal voxels: {len(masks['distal'])}")

    return masks


# ------------------------------------------------------------------
# TOPAS File Generation
# ------------------------------------------------------------------
def generate_topas_file(energy_mev, output_basename, n_histories,
                        output_type="csv", cutoff_x=None):
    """
    Write a TOPAS parameter file matching dose_scoring_proton.txt
    but with the given energy, output type, and output path.
    """
    cx = cutoff_x if cutoff_x is not None else BEAM_POS_CUTOFF_X
    lines = [
        f"# AUTO-GENERATED: proton beam at {energy_mev:.1f} MeV",
        f"# Based on A2_5/dose_scoring_proton.txt (geometry unchanged)",
        "",
        "# Geometry",
        "includeFile = ct_geometry.txt",
        "",
        "# Run control",
        f"i:So/Beam/NumberOfHistoriesInRun   = {n_histories}",
        f"i:Ts/ShowHistoryCountAtInterval    = {max(1000, n_histories // 10)}",
        "i:Ts/NumberOfThreads               = 0",
        'b:Ts/PauseBeforeQuit               = "False"',
        "",
        "# Beam position (fixed — from verified geometry)",
        's:Ge/BeamPosition/Type   = "Group"',
        's:Ge/BeamPosition/Parent = "World"',
        f"d:Ge/BeamPosition/TransX = {BEAM_TRANS_X:.1f} mm",
        f"d:Ge/BeamPosition/TransY = {BEAM_TRANS_Y:.1f} mm",
        f"d:Ge/BeamPosition/TransZ = {BEAM_TRANS_Z:.1f} mm",
        f"d:Ge/BeamPosition/RotX   = {BEAM_ROT_X:.1f} deg",
        f"d:Ge/BeamPosition/RotY   = {BEAM_ROT_Y:.1f} deg",
        f"d:Ge/BeamPosition/RotZ   = {BEAM_ROT_Z:.1f} deg",
        "",
        f"# Source — {energy_mev:.1f} MeV proton beam",
        's:So/Beam/Type                     = "Beam"',
        's:So/Beam/Component                = "BeamPosition"',
        's:So/Beam/BeamParticle             = "proton"',
        f"d:So/Beam/BeamEnergy               = {energy_mev:.1f} MeV",
        "u:So/Beam/BeamEnergySpread         = 0.0",
        "",
        "# Spatial extent (fixed)",
        's:So/Beam/BeamPositionDistribution = "Flat"',
        's:So/Beam/BeamPositionCutoffShape  = "Ellipse"',
        f"d:So/Beam/BeamPositionCutoffX      = {cx:.1f} mm",
        f"d:So/Beam/BeamPositionCutoffY      = {BEAM_POS_CUTOFF_Y:.1f} mm",
        "",
        "# Angular spread (fixed)",
        's:So/Beam/BeamAngularDistribution  = "Gaussian"',
        f"d:So/Beam/BeamAngularCutoffX       = {BEAM_ANG_CUTOFF:.1f} deg",
        f"d:So/Beam/BeamAngularCutoffY       = {BEAM_ANG_CUTOFF:.1f} deg",
        f"d:So/Beam/BeamAngularSpreadX       = {BEAM_ANG_SPREAD:.1f} deg",
        f"d:So/Beam/BeamAngularSpreadY       = {BEAM_ANG_SPREAD:.1f} deg",
        "",
        "# Dose scoring",
        's:Sc/PatientDose/Quantity                  = "DoseToMedium"',
        's:Sc/PatientDose/Component                 = "Patient"',
        f's:Sc/PatientDose/OutputType                = "{output_type}"',
        f's:Sc/PatientDose/OutputFile                = "{output_basename}"',
        's:Sc/PatientDose/IfOutputFileAlreadyExists = "Overwrite"',
        'b:Sc/PatientDose/Active                    = "True"',
    ]

    filepath = os.path.join(PROJECT_ROOT, output_basename + "_run.txt")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    return filepath


# ------------------------------------------------------------------
# TOPAS Execution
# ------------------------------------------------------------------
def run_topas(param_file):
    """Run TOPAS from the project root directory."""
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
# Dose Scoring from CSV
# ------------------------------------------------------------------
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


def compute_metrics(dose_map, masks):
    """
    Compute tumour coverage and distal overshoot metrics from a dose map.

    Returns a dict with:
        tumour_mean, tumour_D95, tumour_D50, tumour_D02, tumour_max
        distal_mean   (mean dose in the distal/overshoot region)
        score         (combined optimisation metric)
    """
    # Collect dose values for each region
    tumour_doses = [dose_map.get(v, 0.0) for v in masks["tumour"]]
    distal_doses = [dose_map.get(v, 0.0) for v in masks["distal"]]

    tumour_arr = np.array(tumour_doses)
    distal_arr = np.array(distal_doses)

    tumour_mean = float(np.mean(tumour_arr)) if len(tumour_arr) else 0.0
    tumour_max = float(np.max(tumour_arr)) if len(tumour_arr) else 0.0

    # Dn means "dose received by at least n% of the volume"
    # D95 = 5th percentile (95% of volume gets at least this dose)
    tumour_D95 = float(np.percentile(tumour_arr, 5)) if len(tumour_arr) else 0.0
    tumour_D50 = float(np.percentile(tumour_arr, 50)) if len(tumour_arr) else 0.0
    tumour_D02 = float(np.percentile(tumour_arr, 98)) if len(tumour_arr) else 0.0

    distal_mean = float(np.mean(distal_arr)) if len(distal_arr) else 0.0

    # Score: maximise tumour mean dose while penalising distal overshoot.
    # tumour_mean rewards placing the Bragg peak *inside* the tumour
    # (not just the plateau region). The distal penalty discourages
    # energies that overshoot.
    score = tumour_mean - OVERSHOOT_WEIGHT * distal_mean

    return {
        "tumour_mean": tumour_mean,
        "tumour_D95": tumour_D95,
        "tumour_D50": tumour_D50,
        "tumour_D02": tumour_D02,
        "tumour_max": tumour_max,
        "distal_mean": distal_mean,
        "score": score,
    }


def extract_depth_dose(dose_map, geom, masks):
    """
    Extract dose vs depth along the beam central axis.

    The beam fires in -Y from Y=+200.  Depth is measured as distance
    from the beam entrance (highest Y in the CT grid), so depth
    increases from left to right on the plot.

    Returns (depth_mm, doses) where depth_mm=0 is the beam entrance
    side of the patient.
    """
    tumour_voxels = list(masks["tumour"])
    ix_c = int(round(np.mean([v[0] for v in tumour_voxels])))
    iz_c = int(round(np.mean([v[2] for v in tumour_voxels])))

    dy = geom["dy"]
    y_max = geom["y0"] + geom["rows"] * dy  # beam entrance side (+Y)

    doses = []
    depths = []
    for iy in range(geom["rows"]):
        y_phys = geom["y0"] + (iy + 0.5) * dy
        depth = y_max - y_phys  # distance from beam entrance
        depths.append(depth)
        doses.append(dose_map.get((ix_c, iy, iz_c), 0.0))

    # Sort by increasing depth (beam entrance first)
    order = np.argsort(depths)
    return np.array(depths)[order], np.array(doses)[order]


# ------------------------------------------------------------------
# Energy Sweep
# ------------------------------------------------------------------
def run_energy_sweep(energies, masks, geom, label, n_histories):
    """
    Run TOPAS at each energy and return a list of result dicts.

    Each result contains: energy, metrics, dose_map, csv_path, param_file.
    """
    results = []
    n_total = len(energies)

    for i, energy in enumerate(energies, 1):
        tag = f"{label}_{energy:.0f}MeV"
        basename = os.path.join("A2_5", "output", "_sweep", tag)
        print(f"  [{i}/{n_total}] {energy:.0f} MeV ... ", end="", flush=True)

        t0 = time.time()
        param_file = generate_topas_file(energy, basename, n_histories)
        run_topas(param_file)
        csv_path = os.path.join(PROJECT_ROOT, basename + ".csv")
        dose_map = read_dose_csv(csv_path)
        metrics = compute_metrics(dose_map, masks)
        elapsed = time.time() - t0

        print(f"score={metrics['score']:.4e}  "
              f"D95={metrics['tumour_D95']:.4e}  "
              f"distal={metrics['distal_mean']:.4e}  "
              f"({elapsed:.1f}s)")

        results.append({
            "energy": energy,
            "metrics": metrics,
            "dose_map": dose_map,
            "csv_path": csv_path,
            "param_file": param_file,
        })

    return results


# ------------------------------------------------------------------
# Output: Summary CSV
# ------------------------------------------------------------------
def write_summary_csv(all_results, filepath):
    """Write all tested energies and metrics to a CSV file."""
    rows = []
    for r in all_results:
        row = {"energy_MeV": r["energy"]}
        row.update(r["metrics"])
        rows.append(row)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Output: Plots
# ------------------------------------------------------------------
def plot_energy_metrics(all_results, filepath):
    """Plot tumour D95, tumour mean, and distal mean vs proton energy."""
    energies = [r["energy"] for r in all_results]
    d95 = [r["metrics"]["tumour_D95"] for r in all_results]
    t_mean = [r["metrics"]["tumour_mean"] for r in all_results]
    distal = [r["metrics"]["distal_mean"] for r in all_results]
    scores = [r["metrics"]["score"] for r in all_results]

    # Sort by energy for clean line plots
    order = np.argsort(energies)
    energies = [energies[i] for i in order]
    d95 = [d95[i] for i in order]
    t_mean = [t_mean[i] for i in order]
    distal = [distal[i] for i in order]
    scores = [scores[i] for i in order]

    # Normalise to the maximum of each series for clearer comparison
    def norm(arr):
        mx = max(arr) if max(arr) > 0 else 1.0
        return [v / mx for v in arr]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Top panel: dose metrics
    ax1.plot(energies, norm(d95), "o-", color="tab:red", label="Tumour D95")
    ax1.plot(energies, norm(t_mean), "s-", color="tab:blue", label="Tumour mean")
    ax1.plot(energies, norm(distal), "^-", color="tab:orange", label="Distal mean (overshoot)")
    ax1.set_ylabel("Normalised dose metric")
    ax1.set_title("Proton Energy Optimisation — Tumour Coverage vs Overshoot")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: combined score
    ax2.plot(energies, scores, "D-", color="tab:green", linewidth=2)
    best_idx = int(np.argmax(scores))
    ax2.axvline(energies[best_idx], color="red", linestyle="--", alpha=0.7,
                label=f"Best: {energies[best_idx]:.0f} MeV")
    ax2.set_xlabel("Proton beam energy (MeV)")
    ax2.set_ylabel("Score (tumour mean − overshoot)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_depth_dose(all_results, geom, masks, filepath):
    """
    Plot depth-dose profiles along the beam axis for the best energy
    and a few comparison energies (±20 MeV from best).

    Both the dose curves and the tumour shading are derived from the
    same voxel-index coordinate system (iy * dy), guaranteeing they
    are mutually consistent regardless of beam direction convention.
    """
    # Find best result
    best = max(all_results, key=lambda r: r["metrics"]["score"])
    best_e = best["energy"]

    # Select comparison energies
    comparison_energies = sorted(set([
        max(ENERGY_COARSE_MIN, best_e - 20),
        best_e,
        min(ENERGY_COARSE_MAX, best_e + 20),
    ]))

    fig, ax = plt.subplots(figsize=(9, 5))

    for r in all_results:
        if r["energy"] not in comparison_energies:
            continue
        positions, doses = extract_depth_dose(r["dose_map"], geom, masks)
        style = "-" if r["energy"] == best_e else "--"
        lw = 2.5 if r["energy"] == best_e else 1.5
        ax.plot(positions, doses, style, linewidth=lw,
                label=f'{r["energy"]:.0f} MeV')

    # Mark tumour region in depth coordinates
    dy = geom["dy"]
    y_max = geom["y0"] + geom["rows"] * dy
    tumour_ys = [geom["y0"] + (v[1] + 0.5) * dy for v in masks["tumour"]]
    tumour_depth_lo = y_max - max(tumour_ys)  # proximal edge (beam hits first)
    tumour_depth_hi = y_max - min(tumour_ys)  # distal edge
    ax.axvspan(tumour_depth_lo, tumour_depth_hi, alpha=0.15, color="red",
               label="Tumour")

    ax.set_xlabel("Depth from beam entrance (mm)")
    ax.set_ylabel("Dose (Gy)")
    ax.set_title("Depth-Dose Profiles — Bragg Peak vs Tumour Position")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Lateral Beam Width Sweep
# ------------------------------------------------------------------
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
    """Compute DVH metrics for each structure mask."""
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


def run_lateral_sweep(best_energy, masks, geom):
    """
    Sweep BeamPositionCutoffX at the best energy to find the
    tightest lateral beam width that maintains tumour coverage.

    Selection: smallest cutoff_x where tumour D95 >= 80% of the
    maximum tumour D95 across all tested widths.
    """
    print("\n" + "=" * 70)
    print("LATERAL BEAM WIDTH OPTIMISATION")
    print("=" * 70)
    print(f"  Energy: {best_energy:.1f} MeV")
    print(f"  Testing CutoffX values: {LATERAL_SWEEP_VALUES} mm")
    print(f"  Histories per run: {LATERAL_SWEEP_HISTORIES}")

    results = []
    n = len(LATERAL_SWEEP_VALUES)

    for i, cx in enumerate(LATERAL_SWEEP_VALUES, 1):
        tag = f"lateral_{cx:.0f}mm"
        basename = os.path.join("A2_5", "output", "_sweep", tag)
        print(f"  [{i}/{n}] CutoffX = {cx:.1f} mm ... ", end="", flush=True)

        t0 = time.time()
        param_file = generate_topas_file(
            energy_mev=best_energy,
            output_basename=basename,
            n_histories=LATERAL_SWEEP_HISTORIES,
            output_type="csv",
            cutoff_x=cx,
        )
        run_topas(param_file)

        csv_path = os.path.join(PROJECT_ROOT, basename + ".csv")
        dose_map = read_dose_csv(csv_path)
        metrics = compute_dvh_metrics(dose_map, masks)

        elapsed = time.time() - t0
        t_d95 = metrics.get("tumour", {}).get("D95", 0)
        b_mean = metrics.get("body", {}).get("mean", 0)
        print(f"GTVp D95={t_d95:.4e}  Body mean={b_mean:.4e}  ({elapsed:.1f}s)")

        try:
            os.remove(param_file)
        except OSError:
            pass

        results.append({
            "cutoff_x": cx,
            "metrics": metrics,
            "csv_path": csv_path,
        })

    # Print summary table
    print(f"\n  {'CutoffX (mm)':>14s} {'GTVp D95':>12s} {'GTVp mean':>12s} "
          f"{'Lung_R mean':>12s} {'Body mean':>12s}")
    print(f"  {'-' * 64}")
    for r in results:
        m = r["metrics"]
        t = m.get("tumour", {})
        l = m.get("lung_r", {})
        b = m.get("body", {})
        print(f"  {r['cutoff_x']:14.1f} {t.get('D95',0):12.4e} {t.get('mean',0):12.4e} "
              f"{l.get('mean',0):12.4e} {b.get('mean',0):12.4e}")

    # Select: smallest cutoff_x with tumour mean >= 80% of maximum mean
    max_mean = max(r["metrics"].get("tumour", {}).get("mean", 0) for r in results)
    threshold = 0.80 * max_mean
    eligible = [r for r in results
                if r["metrics"].get("tumour", {}).get("mean", 0) >= threshold]

    if eligible:
        best_lateral = min(eligible, key=lambda r: r["cutoff_x"])
    else:
        best_lateral = max(results,
                           key=lambda r: r["metrics"].get("tumour", {}).get("mean", 0))

    best_cx = best_lateral["cutoff_x"]
    print(f"\n  Tumour mean threshold (80% of max): {threshold:.4e}")
    print(f"  OPTIMAL CutoffX: {best_cx:.1f} mm")

    # Plot
    plot_lateral_sweep(results, os.path.join(OUTPUT_DIR, "proton_lateral_sweep.png"))

    return best_cx


def plot_lateral_sweep(results, filepath):
    """Plot DVH metrics vs lateral beam width."""
    cx_vals = [r["cutoff_x"] for r in results]
    t_d95 = [r["metrics"].get("tumour", {}).get("D95", 0) for r in results]
    t_mean = [r["metrics"].get("tumour", {}).get("mean", 0) for r in results]
    b_mean = [r["metrics"].get("body", {}).get("mean", 0) for r in results]
    l_mean = [r["metrics"].get("lung_r", {}).get("mean", 0) for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(cx_vals, t_d95, "o-", color="red", label="GTVp D95")
    ax1.plot(cx_vals, t_mean, "s-", color="blue", label="GTVp mean")
    ax1.set_ylabel("Dose (Gy)")
    ax1.set_title("Lateral Beam Width Optimisation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(cx_vals, b_mean, "^-", color="grey", label="Body mean")
    ax2.plot(cx_vals, l_mean, "d-", color="orange", label="Lung_R mean")
    ax2.set_xlabel("BeamPositionCutoffX (mm)")
    ax2.set_ylabel("Dose (Gy)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ------------------------------------------------------------------
# Output: Final TOPAS File
# ------------------------------------------------------------------
def write_final_topas(best_energy, filepath, cutoff_x=None):
    """
    Write the production TOPAS file at the optimal energy.
    Uses DICOM output and full histories for VICTORIA import.
    """
    basename = f"A2_5/output/dose_proton_{best_energy:.0f}MeV"
    param_file = generate_topas_file(
        energy_mev=best_energy,
        output_basename=basename,
        n_histories=PROD_HISTORIES,
        output_type="dicom",
        cutoff_x=cutoff_x,
    )

    # Move to the final location
    import shutil
    shutil.move(param_file, filepath)
    print(f"Saved: {filepath}")


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
def cleanup_sweep_files(results):
    """Remove intermediate TOPAS parameter files from the sweep."""
    for r in results:
        try:
            os.remove(r["param_file"])
        except OSError:
            pass


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "_sweep"), exist_ok=True)

    print("=" * 70)
    print("SECTION 2.5: PROTON BEAM ENERGY OPTIMISATION")
    print("=" * 70)
    print(f"Beam geometry: fixed (TransX={BEAM_TRANS_X}, TransY={BEAM_TRANS_Y}, "
          f"RotX={BEAM_ROT_X})")
    print(f"Tumour: ({TUMOUR_X}, {TUMOUR_Y}, {TUMOUR_Z}) mm, R={TUMOUR_RADIUS} mm")
    print(f"Metric: score = tumour_mean - {OVERSHOOT_WEIGHT}*distal_mean")

    # --- Step 1: Load geometry and build masks ---
    print("\nLoading CT geometry and building structure masks...")
    geom = load_ct_geometry()
    masks = build_structure_masks(geom)

    # --- Step 2: Coarse energy sweep ---
    coarse_energies = np.arange(
        ENERGY_COARSE_MIN,
        ENERGY_COARSE_MAX + 0.5 * ENERGY_COARSE_STEP,
        ENERGY_COARSE_STEP,
    )
    print(f"\nCoarse sweep: {coarse_energies[0]:.0f} to {coarse_energies[-1]:.0f} MeV "
          f"({len(coarse_energies)} steps, {SWEEP_HISTORIES} histories each)")

    coarse_results = run_energy_sweep(
        coarse_energies, masks, geom, label="coarse", n_histories=SWEEP_HISTORIES
    )

    best_coarse = max(coarse_results, key=lambda r: r["metrics"]["score"])
    print(f"\n  Best coarse energy: {best_coarse['energy']:.0f} MeV "
          f"(score={best_coarse['metrics']['score']:.4e})")

    # --- Step 3: Refined sweep around the best coarse energy ---
    refine_min = max(ENERGY_COARSE_MIN, best_coarse["energy"] - ENERGY_REFINE_HALF)
    refine_max = min(ENERGY_COARSE_MAX, best_coarse["energy"] + ENERGY_REFINE_HALF)
    refine_energies = np.arange(refine_min, refine_max + 0.5 * ENERGY_REFINE_STEP,
                                ENERGY_REFINE_STEP)
    # Skip energies already tested in the coarse sweep
    coarse_set = set(float(e) for e in coarse_energies)
    refine_energies = [e for e in refine_energies if e not in coarse_set]

    print(f"\nRefinement: {refine_min:.0f} to {refine_max:.0f} MeV "
          f"({len(refine_energies)} new steps)")

    refine_results = run_energy_sweep(
        refine_energies, masks, geom, label="refine", n_histories=REFINE_HISTORIES
    )

    # --- Step 4: Find the overall best ---
    all_results = coarse_results + refine_results

    # Bragg peak placement check: find the energy whose peak sits inside
    # the tumour (ideally at ~75% depth through it), not beyond the
    # distal edge.  This prevents the score metric from favouring
    # energies that overshoot but still deliver uniform plateau dose.
    dy = geom["dy"]
    y_max = geom["y0"] + geom["rows"] * dy
    tumour_ys = [geom["y0"] + (v[1] + 0.5) * dy for v in masks["tumour"]]
    tumour_depth_prox = y_max - max(tumour_ys)
    tumour_depth_dist = y_max - min(tumour_ys)
    # Target: Bragg peak at 75% through the tumour (proximal→distal)
    target_peak_depth = tumour_depth_prox + 0.75 * (tumour_depth_dist - tumour_depth_prox)

    print(f"\n  Tumour depth range: {tumour_depth_prox:.1f} – {tumour_depth_dist:.1f} mm")
    print(f"  Target Bragg peak depth: {target_peak_depth:.1f} mm (75% through tumour)")

    # For each result, find the Bragg peak depth
    for r in all_results:
        depths, doses = extract_depth_dose(r["dose_map"], geom, masks)
        r["bragg_peak_depth"] = float(depths[np.argmax(doses)])

    # Select best: among energies whose peak is inside the tumour
    # (within tumour depth range), pick the one with highest score.
    # If none have peaks inside, pick the one closest to the target depth.
    inside = [r for r in all_results
              if tumour_depth_prox <= r["bragg_peak_depth"] <= tumour_depth_dist]

    if inside:
        best = max(inside, key=lambda r: r["metrics"]["score"])
        print(f"  {len(inside)} energies have Bragg peak inside tumour")
    else:
        print("  No energy places peak inside tumour — selecting closest to target depth")
        best = min(all_results,
                   key=lambda r: abs(r["bragg_peak_depth"] - target_peak_depth))

    best_energy = best["energy"]
    m = best["metrics"]

    print(f"  Selected peak depth: {best['bragg_peak_depth']:.1f} mm")
    print("\n" + "=" * 70)
    print(f"OPTIMAL ENERGY: {best_energy:.1f} MeV")
    print("=" * 70)
    print(f"  Tumour mean : {m['tumour_mean']:.6e} Gy")
    print(f"  Tumour D95  : {m['tumour_D95']:.6e} Gy")
    print(f"  Tumour D50  : {m['tumour_D50']:.6e} Gy")
    print(f"  Tumour D02  : {m['tumour_D02']:.6e} Gy")
    print(f"  Tumour max  : {m['tumour_max']:.6e} Gy")
    print(f"  Distal mean : {m['distal_mean']:.6e} Gy")
    print(f"  Score       : {m['score']:.6e}")

    # --- Step 5: Lateral beam width optimisation ---
    best_cutoff_x = run_lateral_sweep(best_energy, masks, geom)

    # --- Step 6: Save outputs ---
    print("\nSaving outputs...")

    csv_path = os.path.join(OUTPUT_DIR, "energy_sweep_summary.csv")
    write_summary_csv(all_results, csv_path)

    plot_energy_metrics(all_results, os.path.join(OUTPUT_DIR, "proton_energy_vs_metrics.png"))
    plot_depth_dose(all_results, geom, masks, os.path.join(OUTPUT_DIR, "proton_depth_dose.png"))

    final_topas = os.path.join(OUTPUT_DIR, f"optimised_proton_{best_energy:.0f}MeV.txt")
    write_final_topas(best_energy, final_topas, cutoff_x=best_cutoff_x)

    # --- Step 7: Run the production file to generate the DICOM for VICTORIA ---
    print(f"\nRunning production TOPAS ({PROD_HISTORIES} histories) "
          f"to generate DICOM dose file...")
    run_topas(final_topas)
    dicom_path = os.path.join(OUTPUT_DIR, f"dose_proton_{best_energy:.0f}MeV.dcm")
    print(f"DICOM dose file for VICTORIA: {dicom_path}")

    cleanup_sweep_files(all_results)

    # --- Summary table ---
    print("\n" + "-" * 70)
    print(f"{'Energy (MeV)':>14s} {'Tumour D95':>14s} {'Tumour mean':>14s} "
          f"{'Distal mean':>14s} {'Score':>14s}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["energy"]):
        m = r["metrics"]
        marker = " <-- BEST" if r["energy"] == best_energy else ""
        print(f"{r['energy']:14.1f} {m['tumour_D95']:14.4e} {m['tumour_mean']:14.4e} "
              f"{m['distal_mean']:14.4e} {m['score']:14.4e}{marker}")

    print(f"\nDone. Run the optimised file from the project root:")
    print(f"  topas A2_5/output/optimised_proton_{best_energy:.0f}MeV.txt")


if __name__ == "__main__":
    main()
