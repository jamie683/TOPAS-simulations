import os
import sys
import csv
import time
import shutil
import subprocess
from collections import defaultdict

import numpy as np

try:
    import pydicom
except ImportError:
    print('pydicom not found. Installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pydicom'])
    import pydicom

try:
    from matplotlib.path import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(f'Matplotlib is required for this script: {exc}')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
CT_DIR = os.path.join(PROJECT_ROOT, 'CTData')
TOPAS_EXE = '/home/jamie/shellScripts/topas'

# ------------------------------------------------------------------
# VERIFIED GEOMETRY ANCHOR FROM dose_scoring.txt
# ------------------------------------------------------------------
# Truth we keep fixed:
#   - tumour centre at (-46, +43, 0) mm
#   - posterior source plane at +Y
#   - RotX = -90 deg, RotZ = 0 deg
#   - RotY is the empirically verified incidence-angle control
TUMOUR_X = -46.0
TUMOUR_Y = 43.0
TUMOUR_Z = 0.0
TUMOUR_RADIUS = 15.0
TRANS_Y = 200.0  # mm, hardcoded as requested for now
TRANS_Z = 0.0
ROT_X = -90.0
ROT_Z = 0.0

# Conservative patient envelope, only used for sanity checks.
PATIENT_X_MIN = -220.0
PATIENT_X_MAX = 220.0
PATIENT_Y_MIN = -150.0
PATIENT_Y_MAX = 150.0

# Beam model tied to the manually verified setup.
BEAM_ENERGY = 1.0
BEAM_POS_X = 5.0
BEAM_POS_Y = 5.0
BEAM_ANG_CUTOFF = 5.0
BEAM_ANG_SPREAD = 0.5

# Search settings.
SWEEP_HISTORIES = 5000
SWEEP_SEED = 42
PROD_HISTORIES = 10000000
SEARCH_BEAM_COUNTS = (2, 3, 4)

# Geometry search ranges requested by user.
X_MIN = -150.0
X_MAX = 80.0
DELTA_X_MIN = 20.0
DELTA_X_MAX = min(TUMOUR_X - X_MIN, X_MAX - TUMOUR_X)
THETA_MIN = 0.0
THETA_MAX = 50.0
X_COARSE_STEP = 20.0
THETA_COARSE_STEP = 10.0
X_REFINE_STEP = 10.0
THETA_REFINE_STEP = 5.0
REFINE_HALF_WIDTH_X = 20.0
REFINE_HALF_WIDTH_THETA = 10.0

# TransY (source distance) sweep for 3/4-beam plans — helps beams converge
# at tumour depth rather than behind it.
TRANS_Y_VALUES = [150.0, 200.0, 250.0, 300.0]
TRANS_Y_HISTORIES = 5000  # low histories since this is a coarse distance sweep

# Weight search.
ENABLE_WEIGHT_OPTIMISATION = True
WEIGHT_GRID_2BEAM = np.linspace(0.3, 0.7, 5)
WEIGHT_GRID_3BEAM = np.linspace(0.2, 0.35, 4)
WEIGHT_GRID_4BEAM_GROUP = np.linspace(0.35, 0.65, 7)

# RTStruct names expected in the assignment files.
STRUCTURE_ALIASES = {
    'tumour': ['GTVp', 'GTV', 'Tumour', 'Tumor'],
    'lung_r': ['Lung_R', 'Right Lung', 'LungR', 'Rt Lung'],
    'heart': ['Heart'],
    'cord': ['SpinalCord', 'Spinal Cord', 'Cord'],
    'body': ['Body', 'External', 'BODY', 'EXTERNAL'],
}

# Fallback masks if RTStruct parsing is unavailable.
FALLBACK_BOXES = {
    'lung_r': {'x': (-20.0, 150.0), 'y': (-80.0, 40.0), 'z': (-30.0, 30.0)},
    'heart': {'x': (-55.0, 20.0), 'y': (-15.0, 55.0), 'z': (-30.0, 30.0)},
    'cord': {'x': (-12.0, 12.0), 'y': (-95.0, -55.0), 'z': (-25.0, 25.0)},
}

OBJECTIVE_WEIGHTS = {
    'tumour_mean': 1.00,
    'tumour_D95': 1.35,
    'tumour_std': 0.25,
    'lung_r_mean': 0.00,
    'heart_mean': 0.30,
    'cord_D02': 0.70,
    'body_mean': 0.35,
}


# ------------------------------------------------------------------
# BEAM GEOMETRY
# ------------------------------------------------------------------
def beam_geom(trans_x: float, rot_y: float, trans_y: float = None):
    return {
        'TransX': float(trans_x),
        'TransY': float(trans_y if trans_y is not None else TRANS_Y),
        'TransZ': float(TRANS_Z),
        'RotX': float(ROT_X),
        'RotY': float(rot_y),
        'RotZ': float(ROT_Z),
    }


def convergence_angle(dx, trans_y=None):
    """Compute RotY that aims a beam offset by dx back at the tumour Y-depth.

    theta = arctan(dx / (source_Y - tumour_Y))

    This ensures beams converge at the tumour rather than crossing behind it.
    """
    ty = trans_y if trans_y is not None else TRANS_Y
    depth = ty - TUMOUR_Y
    if depth <= 0:
        return 0.0
    return float(np.degrees(np.arctan(dx / depth)))


def verify_source_outside_patient(geom):
    tx, ty = geom['TransX'], geom['TransY']
    inside_x = PATIENT_X_MIN <= tx <= PATIENT_X_MAX
    inside_y = PATIENT_Y_MIN <= ty <= PATIENT_Y_MAX
    if inside_x and inside_y:
        return False, f'SOURCE INSIDE PATIENT at ({tx:.1f}, {ty:.1f}) mm'
    return True, 'OK'


def symmetric_beam_params(n_beams: int, dx_primary: float, theta_primary: float,
                          dx_secondary: float | None = None, theta_secondary: float | None = None,
                          trans_y: float = None):
    """
    Hardcoded symmetric families, centred on the tumour x-position:
      2-beam:  (x_t+dx,+θ), (x_t-dx,-θ)
      3-beam:  (x_t+dx,+θ), (x_t,0), (x_t-dx,-θ)
      4-beam:  (x_t+dx1,+θ1), (x_t+dx2,+θ2), (x_t-dx2,-θ2), (x_t-dx1,-θ1)
    """
    x_t = TUMOUR_X
    if n_beams == 2:
        # theta locked to converge at tumour depth
        theta = convergence_angle(dx_primary, trans_y)
        return [beam_geom(x_t + dx_primary, +theta, trans_y),
                beam_geom(x_t - dx_primary, -theta, trans_y)]
    if n_beams == 3:
        theta = convergence_angle(dx_primary, trans_y)
        return [beam_geom(x_t + dx_primary, +theta, trans_y),
                beam_geom(x_t, 0.0, trans_y),
                beam_geom(x_t - dx_primary, -theta, trans_y)]
    if n_beams == 4:
        if dx_secondary is None or theta_secondary is None:
            raise ValueError('4-beam requires secondary dx/theta')
        return [
            beam_geom(x_t + dx_primary, +theta_primary, trans_y),
            beam_geom(x_t + dx_secondary, +theta_secondary, trans_y),
            beam_geom(x_t - dx_secondary, -theta_secondary, trans_y),
            beam_geom(x_t - dx_primary, -theta_primary, trans_y),
        ]
    raise ValueError(f'Unsupported n_beams={n_beams}')


def default_weights_for_n(n_beams: int):
    return [1.0 / n_beams] * n_beams


def candidate_weights(n_beams):
    if n_beams == 2:
        return [[float(w0), float(1.0 - w0)] for w0 in WEIGHT_GRID_2BEAM]

    if n_beams == 3:
        out = []
        for w_side in WEIGHT_GRID_3BEAM:
            w_mid = 1.0 - 2.0 * w_side
            if 0.1 <= w_mid <= 0.6:
                out.append([float(w_side), float(w_mid), float(w_side)])
        return out

    if n_beams == 4:
        out = []
        for outer_pair_total in WEIGHT_GRID_4BEAM_GROUP:
            inner_pair_total = 1.0 - outer_pair_total
            w_outer = 0.5 * outer_pair_total
            w_inner = 0.5 * inner_pair_total
            out.append([float(w_outer), float(w_inner), float(w_inner), float(w_outer)])
        return out

    return [default_weights_for_n(n_beams)]


def histories_from_weights(n_histories, weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    raw = weights * n_histories
    counts = np.floor(raw).astype(int)
    missing = int(n_histories - counts.sum())
    if missing > 0:
        frac_order = np.argsort(-(raw - counts))
        for idx in frac_order[:missing]:
            counts[idx] += 1
    return counts.tolist(), weights.tolist()


def generate_topas_file(beam_params, weights, config_label, output_basename,
                        n_histories, output_type='csv', seed=None):
    counts, weights = histories_from_weights(n_histories, weights)

    lines = [
        f'# AUTO-GENERATED: {config_label}',
        '# Verified basis: dose_scoring.txt posterior geometry.',
        '# Hardcoded for now: TransY=200 mm, RotX=-90 deg, RotZ=0 deg.',
        '# Only TransX and RotY are optimised here.',
        'includeFile = ct_geometry.txt',
        '',
        f'i:Ts/ShowHistoryCountAtInterval = {max(1000, n_histories // 10)}',
        'i:Ts/NumberOfThreads            = 0',
        'b:Ts/PauseBeforeQuit            = "False"',
    ]
    if seed is not None:
        lines.append(f'i:Ts/Seed                       = {seed}')
    lines.append('')

    for i, (g, nh, weight) in enumerate(zip(beam_params, counts, weights), start=1):
        ok, msg = verify_source_outside_patient(g)
        status = 'EXTERNAL' if ok else msg
        name = f'Beam{i}'
        lines += [
            f'# {name}: weight={weight:.4f}, histories={nh}',
            f'# Source = ({g["TransX"]:.1f}, {g["TransY"]:.1f}, {g["TransZ"]:.1f}) mm [{status}]',
            f'# Rotations = (RotX={g["RotX"]:.1f}, RotY={g["RotY"]:.1f}, RotZ={g["RotZ"]:.1f}) deg',
            f's:Ge/{name}/Type   = "Group"',
            f's:Ge/{name}/Parent = "World"',
            f'd:Ge/{name}/TransX = {g["TransX"]:.1f} mm',
            f'd:Ge/{name}/TransY = {g["TransY"]:.1f} mm',
            f'd:Ge/{name}/TransZ = {g["TransZ"]:.1f} mm',
            f'd:Ge/{name}/RotX   = {g["RotX"]:.1f} deg',
            f'd:Ge/{name}/RotY   = {g["RotY"]:.1f} deg',
            f'd:Ge/{name}/RotZ   = {g["RotZ"]:.1f} deg',
            '',
            f's:So/{name}/Type                     = "Beam"',
            f's:So/{name}/Component                = "{name}"',
            f's:So/{name}/BeamParticle             = "gamma"',
            f'd:So/{name}/BeamEnergy               = {BEAM_ENERGY:.1f} MeV',
            f'u:So/{name}/BeamEnergySpread         = 0.0',
            f'i:So/{name}/NumberOfHistoriesInRun   = {nh}',
            f's:So/{name}/BeamPositionDistribution = "Flat"',
            f's:So/{name}/BeamPositionCutoffShape  = "Ellipse"',
            f'd:So/{name}/BeamPositionCutoffX      = {BEAM_POS_X:.1f} mm',
            f'd:So/{name}/BeamPositionCutoffY      = {BEAM_POS_Y:.1f} mm',
            f's:So/{name}/BeamAngularDistribution  = "Gaussian"',
            f'd:So/{name}/BeamAngularCutoffX       = {BEAM_ANG_CUTOFF:.1f} deg',
            f'd:So/{name}/BeamAngularCutoffY       = {BEAM_ANG_CUTOFF:.1f} deg',
            f'd:So/{name}/BeamAngularSpreadX       = {BEAM_ANG_SPREAD:.1f} deg',
            f'd:So/{name}/BeamAngularSpreadY       = {BEAM_ANG_SPREAD:.1f} deg',
            '',
        ]

    lines += [
        's:Sc/PatientDose/Quantity                  = "DoseToMedium"',
        's:Sc/PatientDose/Component                 = "Patient"',
        f's:Sc/PatientDose/OutputType                = "{output_type}"',
        f's:Sc/PatientDose/OutputFile                = "{output_basename}"',
        's:Sc/PatientDose/IfOutputFileAlreadyExists = "Overwrite"',
        'b:Sc/PatientDose/Active                    = "True"',
    ]

    filepath = os.path.join(PROJECT_ROOT, output_basename + '_run.txt')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return filepath

# ------------------------------------------------------------------
# CT GEOMETRY + STRUCTURE MASKS
# ------------------------------------------------------------------
def load_ct_geometry(ct_dir):
    slices = []
    for fname in sorted(os.listdir(ct_dir)):
        if not fname.lower().endswith('.dcm'):
            continue
        fpath = os.path.join(ct_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if getattr(ds, 'Modality', '') == 'CT':
                slices.append(ds)
        except Exception:
            pass
    if not slices:
        raise FileNotFoundError(f'No CT slices found in {ct_dir}')

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    ds0 = slices[0]
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    dx = float(ds0.PixelSpacing[1])
    dy = float(ds0.PixelSpacing[0])
    x0 = float(ds0.ImagePositionPatient[0])
    y0 = float(ds0.ImagePositionPatient[1])
    zs = [float(s.ImagePositionPatient[2]) for s in slices]
    dz = abs(zs[1] - zs[0]) if len(zs) > 1 else 1.0

    return {
        'rows': rows,
        'cols': cols,
        'n_slices': len(zs),
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'x0': x0,
        'y0': y0,
        'z0': zs[0],
        'slice_zs': zs,
    }


def voxel_centres_xy(geom):
    xs = geom['x0'] + (np.arange(geom['cols']) + 0.5) * geom['dx']
    ys = geom['y0'] + (np.arange(geom['rows']) + 0.5) * geom['dy']
    return xs, ys


def build_full_grid_mask(geom):
    voxels = set()
    for ix in range(geom['cols']):
        for iy in range(geom['rows']):
            for iz in range(geom['n_slices']):
                voxels.add((ix, iy, iz))
    return voxels


def build_sphere_mask(geom, cx, cy, cz, radius):
    voxels = set()
    for ix in range(geom['cols']):
        x = geom['x0'] + (ix + 0.5) * geom['dx']
        if abs(x - cx) > radius + geom['dx']:
            continue
        for iy in range(geom['rows']):
            y = geom['y0'] + (iy + 0.5) * geom['dy']
            if abs(y - cy) > radius + geom['dy']:
                continue
            for iz in range(geom['n_slices']):
                z = geom['z0'] + (iz + 0.5) * geom['dz']
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2:
                    voxels.add((ix, iy, iz))
    return voxels


def build_box_mask(geom, x_range, y_range, z_range, exclude=None):
    voxels = set()
    exclude = exclude or set()
    for ix in range(geom['cols']):
        x = geom['x0'] + (ix + 0.5) * geom['dx']
        if not (x_range[0] <= x <= x_range[1]):
            continue
        for iy in range(geom['rows']):
            y = geom['y0'] + (iy + 0.5) * geom['dy']
            if not (y_range[0] <= y <= y_range[1]):
                continue
            for iz in range(geom['n_slices']):
                z = geom['z0'] + (iz + 0.5) * geom['dz']
                if not (z_range[0] <= z <= z_range[1]):
                    continue
                key = (ix, iy, iz)
                if key in exclude:
                    continue
                voxels.add(key)
    return voxels


def find_rtstruct_file(ct_dir):
    for fname in sorted(os.listdir(ct_dir)):
        if not fname.lower().endswith('.dcm'):
            continue
        fpath = os.path.join(ct_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, 'Modality', '') == 'RTSTRUCT':
            return fpath
    return None


def find_structure_name(struct_names, aliases):
    alias_lookup = {a.lower(): a for a in aliases}
    names_lower = {name.lower(): name for name in struct_names}
    for alias in aliases:
        if alias.lower() in names_lower:
            return names_lower[alias.lower()]
    return None


def rtstruct_roi_name_map(ds):
    out = {}
    if not hasattr(ds, 'StructureSetROISequence'):
        return out
    for roi in ds.StructureSetROISequence:
        out[int(roi.ROINumber)] = str(roi.ROIName)
    return out


def closest_slice_index(z_value, slice_zs):
    idx = int(np.argmin(np.abs(np.asarray(slice_zs) - z_value)))
    return idx


def contour_to_mask_slice(geom, iz, contour_xy):
    xs, ys = voxel_centres_xy(geom)
    poly = np.asarray(contour_xy, dtype=float)
    path = Path(poly)

    min_x = poly[:, 0].min() - geom['dx']
    max_x = poly[:, 0].max() + geom['dx']
    min_y = poly[:, 1].min() - geom['dy']
    max_y = poly[:, 1].max() + geom['dy']

    ixs = np.where((xs >= min_x) & (xs <= max_x))[0]
    iys = np.where((ys >= min_y) & (ys <= max_y))[0]
    if len(ixs) == 0 or len(iys) == 0:
        return set()

    xv, yv = np.meshgrid(xs[ixs], ys[iys], indexing='xy')
    points = np.column_stack([xv.ravel(), yv.ravel()])
    inside = path.contains_points(points, radius=1e-9)

    voxels = set()
    inside_grid = inside.reshape(len(iys), len(ixs))
    for j, iy in enumerate(iys):
        for i, ix in enumerate(ixs):
            if inside_grid[j, i]:
                voxels.add((int(ix), int(iy), int(iz)))
    return voxels


def build_rtstruct_masks(geom, rtstruct_path):
    ds = pydicom.dcmread(rtstruct_path)
    roi_map = rtstruct_roi_name_map(ds)
    if not hasattr(ds, 'ROIContourSequence'):
        return {}

    masks_by_name = defaultdict(set)

    for roi_contour in ds.ROIContourSequence:
        roi_number = int(getattr(roi_contour, 'ReferencedROINumber', -1))
        roi_name = roi_map.get(roi_number)
        if not roi_name:
            continue
        if not hasattr(roi_contour, 'ContourSequence'):
            continue

        for contour in roi_contour.ContourSequence:
            data = np.asarray(getattr(contour, 'ContourData', []), dtype=float)
            if data.size < 9:
                continue
            pts = data.reshape(-1, 3)
            z_mean = float(np.mean(pts[:, 2]))
            iz = closest_slice_index(z_mean, geom['slice_zs'])
            poly_xy = pts[:, :2]
            masks_by_name[roi_name].update(contour_to_mask_slice(geom, iz, poly_xy))

    return dict(masks_by_name)


def build_structure_masks(geom):
    info = {'source': 'fallback'}
    masks = {}

    tumour = build_sphere_mask(geom, TUMOUR_X, TUMOUR_Y, TUMOUR_Z, TUMOUR_RADIUS)
    if not tumour:
        raise RuntimeError('Tumour mask is empty. Check tumour coordinates against CT geometry.')
    masks['tumour'] = tumour

    rtstruct_path = find_rtstruct_file(CT_DIR)
    rt_masks = build_rtstruct_masks(geom, rtstruct_path) if rtstruct_path else {}

    if rt_masks:
        info['source'] = f'RTSTRUCT ({os.path.basename(rtstruct_path)})'
        info['available_structures'] = sorted(rt_masks.keys())
        for key, aliases in STRUCTURE_ALIASES.items():
            match = find_structure_name(rt_masks.keys(), aliases)
            if match:
                masks[key] = set(rt_masks[match])
                info[f'{key}_name'] = match

    # Fallbacks if RTStruct is missing or a requested structure was not found.
    exclude = masks['tumour']
    for key in ('lung_r', 'heart', 'cord'):
        if key not in masks or not masks[key]:
            box = FALLBACK_BOXES[key]
            masks[key] = build_box_mask(geom, box['x'], box['y'], box['z'], exclude=exclude)
            info[f'{key}_fallback'] = True

    if 'body' not in masks or not masks['body']:
        masks['body'] = build_full_grid_mask(geom)
        info['body_fallback'] = True

    # Keep body non-empty and derive non-tumour body.
    masks['body_minus_tumour'] = set(masks['body']) - set(masks['tumour'])

    return masks, info


# ------------------------------------------------------------------
# TOPAS EXECUTION + METRICS
# ------------------------------------------------------------------
def run_topas(param_file):
    rel = os.path.relpath(param_file, PROJECT_ROOT)
    result = subprocess.run(
        [TOPAS_EXE, rel],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=1800,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '')[-800:]
        raise RuntimeError(tail)


def percentile_from_values(values, q):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean_from_values(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def std_from_values(values):
    if len(values) == 0:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def score_dose(csv_path, masks):
    names = ['tumour', 'lung_r', 'heart', 'cord', 'body_minus_tumour']
    values = {name: [] for name in names}
    sums = {name: 0.0 for name in names}
    maxs = {name: 0.0 for name in names}
    total_dose = 0.0

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                key = (int(parts[0]), int(parts[1]), int(parts[2]))
                dose = float(parts[3])
            except Exception:
                continue

            total_dose += dose
            for name in names:
                if key in masks[name]:
                    values[name].append(dose)
                    sums[name] += dose
                    if dose > maxs[name]:
                        maxs[name] = dose

    metrics = {
        'total_dose': total_dose,
        'tumour_mean': mean_from_values(values['tumour']),
        'tumour_std': std_from_values(values['tumour']),
        'tumour_D95': percentile_from_values(values['tumour'], 5),
        'tumour_D98': percentile_from_values(values['tumour'], 2),
        'tumour_D50': percentile_from_values(values['tumour'], 50),
        'tumour_D02': percentile_from_values(values['tumour'], 98),
        'tumour_max': maxs['tumour'],
        'lung_r_mean': mean_from_values(values['lung_r']),
        'lung_r_D20': percentile_from_values(values['lung_r'], 80),
        'heart_mean': mean_from_values(values['heart']),
        'heart_D20': percentile_from_values(values['heart'], 80),
        'cord_mean': mean_from_values(values['cord']),
        'cord_D02': percentile_from_values(values['cord'], 98),
        'cord_max': maxs['cord'],
        'body_mean': mean_from_values(values['body_minus_tumour']),
        'body_D50': percentile_from_values(values['body_minus_tumour'], 50),
        'tumour_integral': sums['tumour'],
        'lung_r_integral': sums['lung_r'],
        'heart_integral': sums['heart'],
        'cord_integral': sums['cord'],
        'n_tumour_voxels': len(values['tumour']),
        'n_lung_r_voxels': len(values['lung_r']),
        'n_heart_voxels': len(values['heart']),
        'n_cord_voxels': len(values['cord']),
    }
    return metrics


def objective(metrics):
    w = OBJECTIVE_WEIGHTS
    return (
        + w['tumour_mean'] * metrics['tumour_mean']
        + w['tumour_D95'] * metrics['tumour_D95']
        - w['tumour_std'] * metrics['tumour_std']
        - w['lung_r_mean'] * metrics['lung_r_mean']
        - w['heart_mean'] * metrics['heart_mean']
        - w['cord_D02'] * metrics['cord_D02']
        - w['body_mean'] * metrics['body_mean']
    )



# ------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------
def beam_signature(beam_params):
    parts = []
    for b in beam_params:
        s = f'{b["TransX"]:.1f}/{b["RotY"]:.1f}'
        if b["TransY"] != TRANS_Y:
            s += f'/Y{b["TransY"]:.0f}'
        parts.append(s)
    return ';'.join(parts)


def evaluate_candidate(n_beams, beam_params, weights, label_suffix, masks, search_dir,
                       n_histories=SWEEP_HISTORIES, seed=SWEEP_SEED):
    stem = f'{label_suffix}_' + '_'.join(
        f'x{b["TransX"]:+06.1f}_r{b["RotY"]:+05.1f}_ty{b["TransY"]:.0f}' for b in beam_params
    ) + '_' + '_'.join(f'w{i+1}{100*x:05.1f}' for i, x in enumerate(weights))
    basename = os.path.join('A2_4', 'output', search_dir, stem.replace(' ', ''))
    param_file = generate_topas_file(
        beam_params=beam_params,
        weights=weights,
        config_label=f'{n_beams}-beam params={beam_signature(beam_params)} weights={weights}',
        output_basename=basename,
        n_histories=n_histories,
        output_type='csv',
        seed=seed,
    )
    run_topas(param_file)
    csv_path = os.path.join(PROJECT_ROOT, basename + '.csv')
    metrics = score_dose(csv_path, masks)
    score = objective(metrics)
    return {
        'n_beams': n_beams,
        'beam_params': beam_params,
        'weights': [float(x) for x in weights],
        'metrics': metrics,
        'score': float(score),
        'csv_path': csv_path,
        'param_file': param_file,
    }


def coarse_candidates_for_n(n_beams):
    xs = np.arange(DELTA_X_MIN, DELTA_X_MAX + 0.5 * X_COARSE_STEP, X_COARSE_STEP)
    thetas = np.arange(THETA_MIN, THETA_MAX + 0.5 * THETA_COARSE_STEP, THETA_COARSE_STEP)
    out = []
    if n_beams in (2, 3):
        # theta is computed from dx via convergence_angle(), so only sweep dx
        for x in xs:
            out.append(symmetric_beam_params(n_beams, x, 0.0))  # theta_primary ignored
        return out

    # 4-beam: use 2x coarser steps to keep the 4D grid manageable.
    step_4b = 2 * X_COARSE_STEP
    tstep_4b = 2 * THETA_COARSE_STEP
    xs_inner = np.arange(DELTA_X_MIN, min(60.0, DELTA_X_MAX) + 0.5 * step_4b, step_4b)
    thetas_inner = np.arange(0.0, min(30.0, THETA_MAX) + 0.5 * tstep_4b, tstep_4b)
    xs = np.arange(DELTA_X_MIN, DELTA_X_MAX + 0.5 * step_4b, step_4b)
    thetas = np.arange(THETA_MIN, THETA_MAX + 0.5 * tstep_4b, tstep_4b)
    for x1 in xs:
        for t1 in thetas:
            for x2 in xs_inner:
                for t2 in thetas_inner:
                    if abs(x2) > abs(x1):
                        continue
                    out.append(symmetric_beam_params(4, x1, t1, x2, t2))
    return out


def refine_candidates_for_best(n_beams, best_params):
    out = []
    if n_beams in (2, 3):
        x0 = abs(best_params[0]['TransX'] - TUMOUR_X)
        xs = np.arange(max(DELTA_X_MIN, x0 - REFINE_HALF_WIDTH_X),
                       min(DELTA_X_MAX, x0 + REFINE_HALF_WIDTH_X) + 0.5 * X_REFINE_STEP,
                       X_REFINE_STEP)
        for x in xs:
            out.append(symmetric_beam_params(n_beams, x, 0.0))
        return out

    x1 = abs(best_params[0]['TransX'] - TUMOUR_X)
    t1 = abs(best_params[0]['RotY'])
    x2 = abs(best_params[1]['TransX'] - TUMOUR_X)
    t2 = abs(best_params[1]['RotY'])
    xs1 = np.arange(max(DELTA_X_MIN, x1 - REFINE_HALF_WIDTH_X), min(DELTA_X_MAX, x1 + REFINE_HALF_WIDTH_X) + 0.5 * X_REFINE_STEP, X_REFINE_STEP)
    ts1 = np.arange(max(THETA_MIN, t1 - REFINE_HALF_WIDTH_THETA), min(THETA_MAX, t1 + REFINE_HALF_WIDTH_THETA) + 0.5 * THETA_REFINE_STEP, THETA_REFINE_STEP)
    xs2 = np.arange(max(DELTA_X_MIN, x2 - REFINE_HALF_WIDTH_X), min(DELTA_X_MAX, x2 + REFINE_HALF_WIDTH_X) + 0.5 * X_REFINE_STEP, X_REFINE_STEP)
    ts2 = np.arange(max(THETA_MIN, t2 - REFINE_HALF_WIDTH_THETA), min(THETA_MAX, t2 + REFINE_HALF_WIDTH_THETA) + 0.5 * THETA_REFINE_STEP, THETA_REFINE_STEP)
    for a in xs1:
        for b in ts1:
            for c in xs2:
                for d in ts2:
                    if abs(c) > abs(a):
                        continue
                    out.append(symmetric_beam_params(4, a, b, c, d))
    return out


def sweep_config(n_beams, masks):
    search_dir = f'_search_{n_beams}beam'
    os.makedirs(os.path.join(OUTPUT_DIR, search_dir), exist_ok=True)
    results = []
    best = None

    for beam_params in coarse_candidates_for_n(n_beams):
        candidate = evaluate_candidate(
            n_beams=n_beams,
            beam_params=beam_params,
            weights=default_weights_for_n(n_beams),
            label_suffix='coarse',
            masks=masks,
            search_dir=search_dir,
        )
        results.append(candidate)
        if best is None or candidate['score'] > best['score']:
            best = candidate

    for beam_params in refine_candidates_for_best(n_beams, best['beam_params']):
        candidate = evaluate_candidate(
            n_beams=n_beams,
            beam_params=beam_params,
            weights=default_weights_for_n(n_beams),
            label_suffix='refine',
            masks=masks,
            search_dir=search_dir,
        )
        results.append(candidate)
        if candidate['score'] > best['score']:
            best = candidate

    if ENABLE_WEIGHT_OPTIMISATION:
        for weights in candidate_weights(n_beams):
            candidate = evaluate_candidate(
                n_beams=n_beams,
                beam_params=best['beam_params'],
                weights=weights,
                label_suffix='weights',
                masks=masks,
                search_dir=search_dir,
            )
            results.append(candidate)
            if candidate['score'] > best['score']:
                best = candidate

    # TransY (source distance) sweep for multi-beam plans.
    # Angled beams may converge behind the tumour at the default distance;
    # sweeping TransY finds the distance where beams overlap at tumour depth.
    if n_beams >= 3 and len(TRANS_Y_VALUES) > 0:
        # Extract best geometry parameters to rebuild with different TransY
        bp = best['beam_params']
        dx1 = abs(bp[0]['TransX'] - TUMOUR_X)
        t1 = abs(bp[0]['RotY'])
        dx2 = abs(bp[1]['TransX'] - TUMOUR_X) if n_beams == 4 else None
        t2 = abs(bp[1]['RotY']) if n_beams == 4 else None

        for ty in TRANS_Y_VALUES:
            trial_params = symmetric_beam_params(
                n_beams, dx1, t1, dx2, t2, trans_y=ty)
            candidate = evaluate_candidate(
                n_beams=n_beams,
                beam_params=trial_params,
                weights=best['weights'],
                label_suffix='transY',
                masks=masks,
                search_dir=search_dir,
                n_histories=TRANS_Y_HISTORIES,
            )
            results.append(candidate)
            if candidate['score'] > best['score']:
                best = candidate

    return results, best


# ------------------------------------------------------------------
# REPORTING / OUTPUT
# ------------------------------------------------------------------
def write_results_csv(n_beams, results):
    outpath = os.path.join(OUTPUT_DIR, f'sweep_metrics_{n_beams}beam.csv')
    rows = []
    for r in results:
        row = {
            'n_beams': r['n_beams'],
            'beam_signature': beam_signature(r['beam_params']),
            'weights': ';'.join(f'{w:.4f}' for w in r['weights']),
            'score': r['score'],
            'csv_path': r['csv_path'],
        }
        row.update(r['metrics'])
        rows.append(row)
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return outpath


def plot_sweep_results(n_beams, results):
    best_by_sig = {}
    for r in results:
        sig = beam_signature(r['beam_params'])
        if sig not in best_by_sig or r['score'] > best_by_sig[sig]['score']:
            best_by_sig[sig] = r

    xs = list(range(len(best_by_sig)))
    bests = sorted(best_by_sig.values(), key=lambda rr: rr['score'], reverse=True)[:20]
    labels = [beam_signature(r['beam_params']) for r in bests]
    tumour_d95 = [r['metrics']['tumour_D95'] for r in bests]
    lung_mean = [r['metrics']['lung_r_mean'] for r in bests]
    score = [r['score'] for r in bests]

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    x = np.arange(len(bests))
    ax.plot(x, tumour_d95, 'o-', label='Tumour D95')
    ax.plot(x, lung_mean, 's-', label='Lung_R mean (reported)')
    ax.plot(x, score, 'd-', label='Objective score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=7)
    ax.set_xlabel('Best candidate geometries')
    ax.set_ylabel('Dose / score (arb.)')
    ax.set_title(f'{n_beams}-beam sweep (top 20 candidates)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f'sweep_{n_beams}beam_optimised.png')
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def generate_production_file(n_beams, best):
    basename = os.path.join('A2_4', 'output', f'dose_{n_beams}beam_optimised')
    tmp = generate_topas_file(
        beam_params=best['beam_params'],
        weights=best['weights'],
        config_label=f'Optimised {n_beams}-beam plan | params={beam_signature(best["beam_params"])} | weights={best["weights"]}',
        output_basename=basename,
        n_histories=PROD_HISTORIES,
        output_type='dicom',
        seed=None,
    )
    final = os.path.join(OUTPUT_DIR, f'optimised_{n_beams}beam_optimised.txt')
    shutil.move(tmp, final)
    return final


def cleanup_intermediate_txt(results):
    for r in results:
        try:
            os.remove(r['param_file'])
        except OSError:
            pass


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 78)
    print('POSTERIOR PHOTON FIELD SWEEP WITH INDEPENDENT TransX + RotY (WEIGHTS + 4-BEAM)')
    print('=' * 78)
    print('Verified geometry anchor: dose_scoring.txt posterior field.')
    print('RotX=-90 deg, RotZ=0 deg fixed. TransY swept for 3/4-beam plans.')
    print('Optimisation stages: coarse ΔX/RotY sweep -> refine -> weights -> TransY sweep (3/4-beam).')

    geom = load_ct_geometry(CT_DIR)
    masks, mask_info = build_structure_masks(geom)

    print('\nMask source:', mask_info.get('source', 'unknown'))
    if 'available_structures' in mask_info:
        print('Available RTSTRUCT structures:', ', '.join(mask_info['available_structures']))
    for key in ('tumour_name', 'lung_r_name', 'heart_name', 'cord_name', 'body_name'):
        if key in mask_info:
            print(f'  {key}: {mask_info[key]}')

    summary = []
    for n_beams in SEARCH_BEAM_COUNTS:
        t0 = time.time()
        results, best = sweep_config(n_beams, masks)
        elapsed = time.time() - t0

        csv_out = write_results_csv(n_beams, results)
        plot_out = plot_sweep_results(n_beams, results)
        prod_out = generate_production_file(n_beams, best)
        cleanup_intermediate_txt(results)

        m = best['metrics']
        print('\n' + '-' * 78)
        print(f'{n_beams}-BEAM BEST PLAN')
        print('-' * 78)
        print(f'beam params : {beam_signature(best["beam_params"])}')
        print(f'weights     : {[round(x, 4) for x in best["weights"]]}')
        print(f'score       : {best["score"]:.6e}')
        print(f'tumour mean : {m["tumour_mean"]:.6e}')
        print(f'tumour D95  : {m["tumour_D95"]:.6e}')
        print(f'tumour std  : {m["tumour_std"]:.6e}')
        print(f'lung_R mean : {m["lung_r_mean"]:.6e}  (reported only, not penalised)')
        print(f'heart mean  : {m["heart_mean"]:.6e}')
        print(f'cord D02    : {m["cord_D02"]:.6e}')
        print(f'body mean   : {m["body_mean"]:.6e}')
        print(f'metrics CSV : {csv_out}')
        print(f'plot        : {plot_out}')
        print(f'production  : {prod_out}')
        print(f'elapsed     : {elapsed:.1f} s')

        summary.append({'n_beams': n_beams, 'beam_signature': beam_signature(best['beam_params']), 'weights': best['weights'], 'score': best['score'], 'tumour_D95': m['tumour_D95'], 'lung_r_mean': m['lung_r_mean'], 'heart_mean': m['heart_mean'], 'cord_D02': m['cord_D02']})

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    for item in summary:
        print(f"{item['n_beams']}-beam | params={item['beam_signature']} | weights={[round(x, 4) for x in item['weights']]} | score={item['score']:.6e} | tumour D95={item['tumour_D95']:.6e} | lung_R mean={item['lung_r_mean']:.6e} (not penalised) | heart mean={item['heart_mean']:.6e} | cord D02={item['cord_D02']:.6e}")

    print('\nRun the production files from the project root, then inspect the DICOM dose in VICTORIA.')


if __name__ == '__main__':
    main()
