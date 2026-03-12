import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# ============================================================
# USER SETTINGS
# ============================================================
TOPAS_EXE = "/home/jamie/shellScripts/topas"
BASE_TXT = Path("A3_aluminium.txt")
OUTDIR = Path("radial_runs")
OUTDIR.mkdir(exist_ok=True)

# Test first: tiny set
INSERT_TRANSZ_CM = [-18, -12, -6, 0, 6, 12]
SEEDS = [101,202,303,404,505]

RADIAL_SCORER_BASENAME = "RadialProfile"

def set_param(text: str, key: str, value: str) -> str:
    pattern = rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*).*$"
    repl = rf"\g<prefix>{value}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        new_text = text.rstrip() + "\n" + f"{key} = {value}\n"
    return new_text

def run_topas(input_txt: Path) -> None:
    subprocess.run(
        [TOPAS_EXE, input_txt.name],
        check=True,
        cwd=input_txt.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def generate_and_run_radial_cases():
    for f in OUTDIR.glob("RadialProfile*.csv"):
        f.unlink()
        
    base_text = BASE_TXT.read_text()

    # -------------------------
    # Baseline: no plate
    # easiest trick = make plate water
    # -------------------------
    for seed in SEEDS:
        txt = base_text
        txt = set_param(txt, "i:Ts/Seed", str(seed))
        txt = set_param(txt, "s:Ge/Plate/Material", '"G4_WATER"')
        txt = set_param(txt, "s:Sc/Radial/OutputFile", f'"{RADIAL_SCORER_BASENAME}_water_s{seed}"')

        runfile = OUTDIR / f"run_radial_water_s{seed}.txt"
        runfile.write_text(txt)
        run_topas(runfile)

    # -------------------------
    # Insert runs: vary plate position
    # -------------------------
    for transz in INSERT_TRANSZ_CM:
        for seed in SEEDS:
            txt = base_text
            txt = set_param(txt, "i:Ts/Seed", str(seed))
            txt = set_param(txt, "s:Ge/Plate/Material", '"G4_Al"')
            txt = set_param(txt, "d:Ge/Plate/TransZ", f"{transz} cm")
            txt = set_param(txt, "s:Sc/Radial/OutputFile", f'"{RADIAL_SCORER_BASENAME}_ins_z{transz:+.2f}_s{seed}"')

            runfile = OUTDIR / f"run_radial_ins_z{transz:+.2f}_s{seed}.txt"
            runfile.write_text(txt)
            run_topas(runfile)

# Baseline (no plate / water-equivalent insert removed)
BASELINE_PATTERN = "RadialProfile_water_s*.csv"

# Insert runs
INSERT_PATTERN = "RadialProfile_ins_z*_s*.csv"

# Output width metric to plot
WIDTH_METRIC = "R80"   # choose from: "R50", "R80", "R90", "RMS"

# Selected insert positions to show as example radial-profile overlays
PROFILE_POSITIONS_TO_PLOT = [-18.0, -6.0, 0.0, 6.0]

# If you want x-axis as depth in water instead of relative TransZ:
# depth_in_water_cm = PHANTOM_FRONT_DEPTH_CM + insert_transz_relative_cm
PHANTOM_FRONT_DEPTH_CM = 0.0
PHANTOM_CENTRE_CM = 20.0   # used only if converting relative TransZ -> depth

# True if your insert positions in filenames are relative to phantom centre
TRANSZ_IS_RELATIVE_TO_PHANTOM_CENTRE = True


# ============================================================
# HEADER PARSING
# ============================================================
def read_header_lines(csv_path: Path):
    header = []
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                header.append(line.strip())
            else:
                break
    return header


def parse_radial_bin_info(csv_path: Path):
    """
    Parse TOPAS header lines like:
      # R in 50 bins of 0.3 cm
      # Phi in 1 bin of 360 deg
      # Z in 1 bin of 0.4 cm
    Returns:
      n_rbins, dr_cm
    """
    header = read_header_lines(csv_path)
    n_rbins = None
    dr_cm = None

    for line in header:
        m = re.search(r"#\s*R\s+in\s+(\d+)\s+bins?\s+of\s+([0-9.eE+-]+)\s*cm", line)
        if m:
            n_rbins = int(m.group(1))
            dr_cm = float(m.group(2))
            break

    if n_rbins is None or dr_cm is None:
        raise ValueError(f"Could not parse radial binning from header of {csv_path}")

    return n_rbins, dr_cm


# ============================================================
# CSV LOADING
# ============================================================
def load_radial_profile(csv_path: Path):
    """
    Load TOPAS radial scorer CSV.

    Expected TOPAS cylinder scorer style:
      col0 = iR
      col1 = iPhi
      col2 = iZ
      col3 = score (EnergyDeposit)

    Returns:
      r_cm          : radial bin centres
      e_r           : energy deposited per radial bin (summed over phi,z if needed)
    """
    n_rbins, dr_cm = parse_radial_bin_info(csv_path)

    data = np.loadtxt(csv_path, comments="#", delimiter=",")
    if data.size == 0:
        raise ValueError(f"{csv_path} has no numeric rows.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{csv_path} expected >=4 columns, got {data.shape[1]}")

    iR = data[:, 0].astype(int)
    score = data[:, 3].astype(float)

    # Sum all rows belonging to each radial bin
    e_r = np.zeros(n_rbins, dtype=float)
    for ir, val in zip(iR, score):
        if 0 <= ir < n_rbins:
            e_r[ir] += val

    r_cm = (np.arange(n_rbins) + 0.5) * dr_cm
    return r_cm, e_r


# ============================================================
# RADIAL METRICS
# ============================================================
def cumulative_fraction(values):
    total = np.sum(values)
    if total <= 0:
        return np.zeros_like(values)
    return np.cumsum(values) / total


def radius_at_fraction(r_cm, e_r, frac):
    """
    Radius containing a chosen fraction of deposited energy.
    Uses linear interpolation on cumulative distribution.
    """
    cdf = cumulative_fraction(e_r)

    if np.all(cdf == 0):
        return np.nan

    idx = np.searchsorted(cdf, frac)
    if idx == 0:
        return float(r_cm[0])
    if idx >= len(r_cm):
        return float(r_cm[-1])

    x0, x1 = r_cm[idx - 1], r_cm[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]

    if y1 == y0:
        return float(x1)

    return float(x0 + (frac - y0) * (x1 - x0) / (y1 - y0))


def rms_radius(r_cm, e_r):
    total = np.sum(e_r)
    if total <= 0:
        return np.nan
    return float(np.sqrt(np.sum(e_r * r_cm**2) / total))


def compute_metrics(r_cm, e_r):
    return {
        "R50": radius_at_fraction(r_cm, e_r, 0.50),
        "R80": radius_at_fraction(r_cm, e_r, 0.80),
        "R90": radius_at_fraction(r_cm, e_r, 0.90),
        "RMS": rms_radius(r_cm, e_r),
    }


# ============================================================
# FILE NAME PARSING
# ============================================================
def extract_seed_from_name(path: Path):
    m = re.search(r"_s(\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Could not parse seed from filename: {path.name}")
    return int(m.group(1))


def extract_transz_from_name(path: Path):
    """
    Expects names like:
      RadialProfile_ins_z-18.00_s101.csv
      RadialProfile_ins_z+6.00_s202.csv
    """
    m = re.search(r"_z([+-]?\d+(?:\.\d+)?)_s\d+\.csv$", path.name)
    if not m:
        raise ValueError(f"Could not parse insert position from filename: {path.name}")
    return float(m.group(1))


def transz_to_depth_cm(transz_cm):
    """
    Convert relative TransZ (with phantom centre at +20 cm and front face at 0 cm)
    to physical depth in water from beam entrance.
    """
    if TRANSZ_IS_RELATIVE_TO_PHANTOM_CENTRE:
        return PHANTOM_CENTRE_CM + transz_cm
    return transz_cm


# ============================================================
# DATA COLLECTION
# ============================================================
def load_baseline_runs():
    files = sorted(OUTDIR.glob(BASELINE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No baseline files found matching {BASELINE_PATTERN} in {OUTDIR}")

    runs = []
    for f in files:
        r_cm, e_r = load_radial_profile(f)
        metrics = compute_metrics(r_cm, e_r)
        runs.append({
            "file": f,
            "seed": extract_seed_from_name(f),
            "r_cm": r_cm,
            "e_r": e_r,
            "metrics": metrics,
        })
    return runs


def load_insert_runs():
    files = sorted(OUTDIR.glob(INSERT_PATTERN))
    if not files:
        raise FileNotFoundError(f"No insert files found matching {INSERT_PATTERN} in {OUTDIR}")

    grouped = {}
    for f in files:
        z = extract_transz_from_name(f)
        r_cm, e_r = load_radial_profile(f)
        metrics = compute_metrics(r_cm, e_r)

        grouped.setdefault(z, []).append({
            "file": f,
            "seed": extract_seed_from_name(f),
            "r_cm": r_cm,
            "e_r": e_r,
            "metrics": metrics,
        })

    return grouped


# ============================================================
# AVERAGING
# ============================================================
def mean_profile(runs):
    """
    Average radial profiles across seeds.
    Assumes same r grid for all runs.
    """
    r_ref = runs[0]["r_cm"]
    stack = np.array([run["e_r"] for run in runs], dtype=float)
    return r_ref, np.mean(stack, axis=0), np.std(stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(r_ref)


def metric_summary(runs, metric_name):
    vals = np.array([run["metrics"][metric_name] for run in runs], dtype=float)
    mean = np.mean(vals)
    std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    sem = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, std, sem

def rebin_radial_profile(r_cm, e_r, factor=2):
    """
    Rebin radial profile by combining adjacent bins.
    factor=2 means 2 old bins -> 1 new bin.

    Returns:
      r_new : new radial bin centres
      e_new : rebinned energy per new radial bin
    """
    r_cm = np.asarray(r_cm, float)
    e_r = np.asarray(e_r, float)

    n = len(e_r) // factor
    if n < 1:
        raise ValueError("Rebin factor too large for profile length.")

    r_new = r_cm[:n * factor].reshape(n, factor).mean(axis=1)
    e_new = e_r[:n * factor].reshape(n, factor).sum(axis=1)

    return r_new, e_new

# ============================================================
# PLOTTING
# ============================================================
def plot_baseline_profile(baseline_runs):
    r_cm, mean_e, std_e = mean_profile(baseline_runs)
    r_cm, mean_e = rebin_radial_profile(r_cm, mean_e, factor=2)   # or 3, 4
    norm = np.sum(mean_e)
    y = mean_e / norm if norm > 0 else mean_e
    plt.figure()
    plt.plot(r_cm, y, marker="o", ms=3)
    plt.xlabel("Radius (cm)", fontsize=14)
    plt.xlim(0, 8)
    plt.ylabel("Normalised energy deposit", fontsize=14)
    plt.title("Baseline radial energy-deposition profile (no inset plate)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "baseline_radial_profile.png", dpi=300)


def plot_selected_profiles(insert_grouped, baseline_runs):
    plt.figure()

    # baseline
    r0, e0, _ = mean_profile(baseline_runs)
    r0, e0 = rebin_radial_profile(r0, e0, factor=2)
    y0 = e0 / np.sum(e0) if np.sum(e0) > 0 else e0
    plt.plot(r0, y0, label="No plate", linewidth=2)

    for z in PROFILE_POSITIONS_TO_PLOT:
        if z not in insert_grouped:
            continue
        r_cm, mean_e, _ = mean_profile(insert_grouped[z])
        r_cm, mean_e = rebin_radial_profile(r_cm, mean_e, factor=2)
        y = mean_e / np.sum(mean_e) if np.sum(mean_e) > 0 else mean_e
        x_label = transz_to_depth_cm(z)
        plt.plot(r_cm, y, label=f"Insert depth = {x_label:.1f} cm")

    plt.xlabel("Radius (cm)", fontsize=14)
    plt.xlim(0, 8)
    plt.ylabel("Normalised energy deposit", fontsize=14)
    plt.title("Radial profiles at 25 cm depth", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "selected_radial_profiles.png", dpi=300)


def plot_width_vs_position(insert_grouped, baseline_runs, metric_name="R80"):
    baseline_mean, baseline_std, baseline_sem = metric_summary(baseline_runs, metric_name)

    z_vals = sorted(insert_grouped.keys())
    x_depth = np.array([transz_to_depth_cm(z) for z in z_vals], dtype=float)
    means = []
    sems = []

    for z in z_vals:
        mean, std, sem = metric_summary(insert_grouped[z], metric_name)
        means.append(mean)
        sems.append(sem)

    means = np.array(means)
    sems = np.array(sems)

    plt.figure()
    plt.errorbar(x_depth, means, yerr=sems, fmt="o", capsize=4, elinewidth=1, ms=5, label=f"With inset plate")
    plt.axhline(float(baseline_mean), linestyle="--", label=f"No plate baseline ({metric_name} = {baseline_mean:.3f} cm)")
    plt.xlabel("Insert depth in water (cm)", fontsize=14)
    plt.ylabel(f"{metric_name} beam width at 25 cm depth (cm)", fontsize=14)
    plt.title(f"{metric_name} vs insert position", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{metric_name}_vs_insert_position.png", dpi=300)

def plot_relative_broadening(insert_grouped, baseline_runs, metric_name="R80"):
    baseline_mean, baseline_std, baseline_sem = metric_summary(baseline_runs, metric_name)

    z_vals = sorted(insert_grouped.keys())
    x_depth = np.array([transz_to_depth_cm(z) for z in z_vals], dtype=float)

    deltas = []
    sems = []

    for z in z_vals:
        mean, std, sem = metric_summary(insert_grouped[z], metric_name)

        delta = mean - baseline_mean
        deltas.append(delta)

        # propagate uncertainty (baseline + measurement)
        total_sem = np.sqrt(sem**2 + baseline_sem**2)
        sems.append(total_sem)

    deltas = np.array(deltas)
    sems = np.array(sems)

    plt.figure()
    plt.errorbar(
        x_depth,
        deltas,
        yerr=sems,
        fmt="o",
        capsize=4,
        elinewidth=1,
        ms=5
    )

    plt.axhline(0, linestyle="--")
    plt.xlabel("Insert depth in water (cm)", fontsize=14)
    plt.ylabel(f"Δ{metric_name} beam width (cm)", fontsize=14)
    plt.title(f"Beam broadening relative to baseline ({metric_name})", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTDIR / f"delta_{metric_name}_vs_insert_position.png", dpi=300)

# ============================================================
# OUTPUT TABLES
# ============================================================
def save_summary_csv(insert_grouped, baseline_runs):
    metric_names = ["R50", "R80", "R90", "RMS"]

    lines = []
    lines.append("case,insert_transz_cm,insert_depth_cm,metric,mean_cm,std_cm,sem_cm")

    # baseline
    for m in metric_names:
        mean, std, sem = metric_summary(baseline_runs, m)
        lines.append(f"baseline,,,{m},{mean:.6f},{std:.6f},{sem:.6f}")

    # inserts
    for z in sorted(insert_grouped.keys()):
        depth = transz_to_depth_cm(z)
        runs = insert_grouped[z]
        for m in metric_names:
            mean, std, sem = metric_summary(runs, m)
            lines.append(f"insert,{z:.6f},{depth:.6f},{m},{mean:.6f},{std:.6f},{sem:.6f}")

    out_path = OUTDIR / "radial_width_summary.csv"
    out_path.write_text("\n".join(lines))


# ============================================================
# MAIN
# ============================================================
def main():
    generate_and_run_radial_cases()
    baseline_runs = load_baseline_runs()
    insert_grouped = load_insert_runs()

    print("=== BASELINE (no plate) ===")
    for m in ["R50", "R80", "R90", "RMS"]:
        mean, std, sem = metric_summary(baseline_runs, m)
        print(f"{m}: {mean:.4f} ± {sem:.4f} cm (SEM)")

    print("\n=== INSERT RUNS ===")
    for z in sorted(insert_grouped.keys()):
        depth = transz_to_depth_cm(z)
        mean, std, sem = metric_summary(insert_grouped[z], WIDTH_METRIC)
        print(f"Depth {depth:5.1f} cm (TransZ {z:+5.1f} cm): {WIDTH_METRIC} = {mean:.4f} ± {sem:.4f} cm")

    save_summary_csv(insert_grouped, baseline_runs)
    plot_baseline_profile(baseline_runs)
    plot_selected_profiles(insert_grouped, baseline_runs)
    plot_width_vs_position(insert_grouped, baseline_runs, metric_name=WIDTH_METRIC)
    plot_relative_broadening(insert_grouped, baseline_runs, metric_name=WIDTH_METRIC)

    print("\nSaved:")
    print(" - baseline_radial_profile.png")
    print(" - selected_radial_profiles.png")
    print(f" - {WIDTH_METRIC}_vs_insert_position.png")
    print(" - radial_width_summary.csv")


if __name__ == "__main__":
    main()