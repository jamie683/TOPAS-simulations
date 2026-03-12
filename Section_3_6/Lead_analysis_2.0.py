import re
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================
TOPAS_EXE = "/home/jamie/shellScripts/topas"
BASE_TXT = Path("A3_aluminium.txt")

# ---------- Phase 1: lead thickness matching ----------
MATCH_OUTDIR = Path("lead_match_runs")
MATCH_OUTDIR.mkdir(exist_ok=True)

SEEDS = [101, 202, 303]
FINAL_SEEDS = [101, 202, 303, 404, 505]

DOSE_SCORER_BASENAME = "DoseZ"

# Bisection settings for lead HL
HL_LOW0 = 0.26
HL_HIGH0 = 0.34
HL_STEP = 0.02

EPS_Z_CM = 0.02
EPS_HL_CM = 0.002
MAX_ITER = 8

HIST_FAR = 2000
HIST_MID = 5000
HIST_FINAL = 15000

FAR_THRESH_CM = 0.10

# ---------- Phase 2: radial comparison ----------
RADIAL_OUTDIR = Path("lead_radial_runs")
RADIAL_OUTDIR.mkdir(exist_ok=True)

RADIAL_SCORER_BASENAME = "RadialProfile"
WIDTH_METRIC = "R80"   # choose from: "R50", "R80", "R90", "RMS"

# Geometry assumptions
PHANTOM_CENTRE_CM = 20.0
TRANSZ_SURFACE_CM = -18.0   # centre of 4 cm thick plate at the surface
AL_HL_CM = 2.0            # 2 cm thick aluminium insert from 3.5
Z_BIN_WIDTH_CM = 0.05       # depth-dose scorer bin width

# ============================================================
# GENERAL HELPERS
# ============================================================
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

def safe_remove_glob(folder: Path, pattern: str):
    for f in folder.glob(pattern):
        f.unlink()

def rebin_radial_profile(r_cm, e_r, factor=2):
    r_cm = np.asarray(r_cm, float)
    e_r = np.asarray(e_r, float)

    n = len(e_r) // factor
    r_new = r_cm[:n * factor].reshape(n, factor).mean(axis=1)
    e_new = e_r[:n * factor].reshape(n, factor).sum(axis=1)

    return r_new, e_new

# ============================================================
# DEPTH-DOSE / BRAGG PEAK HELPERS
# ============================================================
def load_dose_csv(csv_path: Path):
    data = np.loadtxt(csv_path, comments="#", delimiter=",")
    if data.size == 0:
        raise ValueError(f"{csv_path} has no numeric rows.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{csv_path} expected >=4 columns, got {data.shape[1]}")

    iz = data[:, 2].astype(float)
    dose = data[:, 3].astype(float)
    z_cm = (iz + 0.5) * Z_BIN_WIDTH_CM
    return z_cm, dose


def bragg_peak_z_parabolic(z_cm: np.ndarray, dose: np.ndarray) -> float:
    i = int(np.argmax(dose))
    if i == 0 or i == len(dose) - 1:
        return float(z_cm[i])

    y0, y1, y2 = dose[i - 1], dose[i], dose[i + 1]
    x0, x1, x2 = z_cm[i - 1], z_cm[i], z_cm[i + 1]

    dx = x2 - x1
    denom = (y0 - 2 * y1 + y2)
    if dx == 0 or denom == 0:
        return float(x1)

    delta = (y0 - y2) / (2 * denom)
    return float(x1 + delta * dx)

# ============================================================
# PHASE 1: LEAD THICKNESS MATCHING (BISECTION)
# ============================================================
def run_depth_case(material: str, hl_cm: float, seed: int, histories: int, out_base: str) -> float:
    base_text = BASE_TXT.read_text()

    txt = base_text
    txt = set_param(txt, "i:Ts/Seed", str(seed))
    txt = set_param(txt, "i:So/Beam/NumberOfHistoriesInRun", str(histories))
    txt = set_param(txt, "s:Ge/Plate/Material", material)
    txt = set_param(txt, "d:Ge/Plate/HL", f"{hl_cm:.4f} cm")
    txt = set_param(txt, "d:Ge/Plate/TransZ", f"{TRANSZ_SURFACE_CM} cm")
    txt = set_param(txt, "s:Sc/Dose/OutputFile", f'"{out_base}"')

    runfile = MATCH_OUTDIR / f"run_{out_base}.txt"
    runfile.write_text(txt)
    run_topas(runfile)

    csv = MATCH_OUTDIR / f"{out_base}.csv"
    z, dose = load_dose_csv(csv)
    return bragg_peak_z_parabolic(z, dose)


def compute_al_reference():
    peaks = []
    for seed in FINAL_SEEDS:
        print(f"Running aluminium reference, seed {seed}")
        zpk = run_depth_case(
            material='"G4_Al"',
            hl_cm=AL_HL_CM,
            seed=seed,
            histories=HIST_FINAL,
            out_base=f"{DOSE_SCORER_BASENAME}_al_ref_s{seed}"
        )
        peaks.append(zpk)

    peaks = np.array(peaks, float)
    mean = float(np.mean(peaks))
    std = float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0
    sem = std / np.sqrt(len(peaks)) if len(peaks) > 1 else 0.0
    return peaks, (mean, std, sem)

_lead_cache = {}

def lead_peak_mean(hl_cm: float, histories: int, seeds=None):
    if seeds is None:
        seeds = SEEDS

    key = (round(hl_cm, 6), histories, tuple(seeds))
    if key in _lead_cache:
        return _lead_cache[key]

    peaks = []
    for seed in seeds:
        out_base = f"{DOSE_SCORER_BASENAME}_lead_hl{hl_cm:.4f}_h{histories}_s{seed}"
        print(f"Running lead HL={hl_cm:.4f} cm, hist={histories}, seed {seed}")
        zpk = run_depth_case(
            material='"G4_Pb"',
            hl_cm=hl_cm,
            seed=seed,
            histories=histories,
            out_base=out_base
        )
        peaks.append(zpk)

    peaks = np.array(peaks, float)
    mean = float(np.mean(peaks))
    std = float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0
    sem = std / np.sqrt(len(peaks)) if len(peaks) > 1 else 0.0

    result = {"peaks": peaks, "mean": mean, "std": std, "sem": sem}
    _lead_cache[key] = result
    return result


def find_matched_lead_hl_bisection(z_al_mean: float):
    hl_low = HL_LOW0
    hl_high = HL_HIGH0

    f_low = lead_peak_mean(hl_low, HIST_FAR)["mean"] - z_al_mean
    f_high = lead_peak_mean(hl_high, HIST_FAR)["mean"] - z_al_mean

    expand_count = 0
    while not (f_low > 0 and f_high < 0):
        expand_count += 1
        if expand_count > 10:
            raise RuntimeError(
                f"Failed to bracket solution. "
                f"hl_low={hl_low}, f_low={f_low:+.4f}; "
                f"hl_high={hl_high}, f_high={f_high:+.4f}"
            )

        if f_low < 0 and f_high < 0:
            hl_low -= HL_STEP
            hl_high -= HL_STEP
            if hl_low <= 0:
                raise RuntimeError("Bracket shifted below 0 cm HL.")
        elif f_low > 0 and f_high > 0:
            hl_low += HL_STEP
            hl_high += HL_STEP
        elif f_low < 0 and f_high > 0:
            hl_low, hl_high = hl_high, hl_low

        f_low = lead_peak_mean(hl_low, HIST_FAR)["mean"] - z_al_mean
        f_high = lead_peak_mean(hl_high, HIST_FAR)["mean"] - z_al_mean

    print("\nBracket found:")
    print(f"  HL_low={hl_low:.4f} cm  -> z_peak - target = {f_low:+.4f} cm")
    print(f"  HL_high={hl_high:.4f} cm -> z_peak - target = {f_high:+.4f} cm")

    history = []
    best_hl = 0.5 * (hl_low + hl_high)

    for i in range(1, MAX_ITER + 1):
        hl_mid = 0.5 * (hl_low + hl_high)

        mid_result = lead_peak_mean(hl_mid, HIST_FAR)
        z_mid = mid_result["mean"]
        delta = z_mid - z_al_mean
        hist_used = HIST_FAR

        if abs(delta) <= FAR_THRESH_CM:
            mid_result = lead_peak_mean(hl_mid, HIST_MID)
            z_mid = mid_result["mean"]
            delta = z_mid - z_al_mean
            hist_used = HIST_MID

        history.append((i, hl_low, hl_high, hl_mid, hist_used, z_mid, delta))

        print(
            f"iter {i:02d}: HL_mid={hl_mid:.4f} cm  "
            f"z_peak={z_mid:.4f} cm  "
            f"delta={delta:+.4f} cm  "
            f"(hist={hist_used})  "
            f"bracket=[{hl_low:.4f},{hl_high:.4f}]"
        )

        best_hl = hl_mid

        if abs(delta) < EPS_Z_CM or abs(hl_high - hl_low) < EPS_HL_CM:
            break

        if delta > 0:
            # lead too thin -> peak too deep -> increase HL
            hl_low = hl_mid
        else:
            # lead too thick -> peak too shallow -> decrease HL
            hl_high = hl_mid

    final_result = lead_peak_mean(best_hl, HIST_FINAL, seeds=FINAL_SEEDS)
    return best_hl, final_result, history


def save_lead_match_summary(al_stats, best_hl, lead_final_result, history):
    z_al_mean, z_al_std, z_al_sem = al_stats
    lines = []
    lines.append("case,hl_cm,peak_mean_cm,peak_std_cm,peak_sem_cm,delta_from_al_cm")
    lines.append(f"al_reference,{AL_HL_CM:.6f},{z_al_mean:.6f},{z_al_std:.6f},{z_al_sem:.6f},0.000000")

    delta = lead_final_result["mean"] - z_al_mean
    lines.append(
        f"lead_matched,{best_hl:.6f},{lead_final_result['mean']:.6f},"
        f"{lead_final_result['std']:.6f},{lead_final_result['sem']:.6f},{delta:.6f}"
    )

    out_path = MATCH_OUTDIR / "lead_match_summary.csv"
    out_path.write_text("\n".join(lines))

    hist_lines = ["iter,hl_low_cm,hl_high_cm,hl_mid_cm,histories_used,peak_mid_cm,delta_cm"]
    for row in history:
        hist_lines.append(",".join(str(x) for x in row))
    (MATCH_OUTDIR / "lead_bisection_history.csv").write_text("\n".join(hist_lines))


def plot_lead_match(al_stats, history):
    z_al_mean, _, _ = al_stats

    hls = np.array([r[3] for r in history], float)
    deltas = np.array([r[6] for r in history], float)

    plt.figure()
    plt.plot(hls, deltas, marker="o")
    plt.axhline(0.0, linestyle="--", label=f"Al reference = {z_al_mean:.3f} cm")
    plt.xlabel("Lead insert half-length, HL (cm)", fontsize=14)
    plt.ylabel("z_peak,Pb(HL) - z_peak,Al (cm)", fontsize=14)
    plt.title("Bisection convergence for lead thickness", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MATCH_OUTDIR / "lead_bragg_match.png", dpi=300)

# ============================================================
# RADIAL PROFILE HELPERS
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


def load_radial_profile(csv_path: Path):
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

    e_r = np.zeros(n_rbins, dtype=float)
    for ir, val in zip(iR, score):
        if 0 <= ir < n_rbins:
            e_r[ir] += val

    r_cm = (np.arange(n_rbins) + 0.5) * dr_cm
    return r_cm, e_r


def cumulative_fraction(values):
    total = np.sum(values)
    if total <= 0:
        return np.zeros_like(values)
    return np.cumsum(values) / total


def radius_at_fraction(r_cm, e_r, frac):
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
# PHASE 2: RADIAL COMPARISON
# ============================================================
def generate_and_run_lead_radial_cases(best_lead_hl_cm: float):
    safe_remove_glob(RADIAL_OUTDIR, "RadialProfile*.csv")
    safe_remove_glob(RADIAL_OUTDIR, "*.txt")

    base_text = BASE_TXT.read_text()

    cases = [
        ("water", '"G4_WATER"', AL_HL_CM),
        ("aluminium", '"G4_Al"', AL_HL_CM),
        ("lead", '"G4_Pb"', best_lead_hl_cm),
    ]

    for label, material, hl in cases:
        for seed in FINAL_SEEDS:
            print(f"Running radial case: {label}, HL={hl:.4f} cm, seed {seed}")
            txt = base_text
            txt = set_param(txt, "i:Ts/Seed", str(seed))
            txt = set_param(txt, "s:Ge/Plate/Material", material)
            txt = set_param(txt, "d:Ge/Plate/HL", f"{hl:.4f} cm")
            txt = set_param(txt, "d:Ge/Plate/TransZ", f"{TRANSZ_SURFACE_CM} cm")
            txt = set_param(txt, "s:Sc/Radial/OutputFile", f'"{RADIAL_SCORER_BASENAME}_{label}_s{seed}"')

            runfile = RADIAL_OUTDIR / f"run_radial_{label}_s{seed}.txt"
            runfile.write_text(txt)
            run_topas(runfile)


def extract_seed_from_name(path: Path):
    m = re.search(r"_s(\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Could not parse seed from filename: {path.name}")
    return int(m.group(1))


def extract_case_from_name(path: Path):
    m = re.search(r"RadialProfile_([A-Za-z0-9_]+)_s\d+\.csv$", path.name)
    if not m:
        raise ValueError(f"Could not parse case name from filename: {path.name}")
    return m.group(1)


def load_radial_case_runs():
    files = sorted(RADIAL_OUTDIR.glob("RadialProfile_*_s*.csv"))
    if not files:
        raise FileNotFoundError("No radial CSVs found.")

    grouped = {}
    for f in files:
        case = extract_case_from_name(f)
        r_cm, e_r = load_radial_profile(f)
        metrics = compute_metrics(r_cm, e_r)

        grouped.setdefault(case, []).append({
            "file": f,
            "seed": extract_seed_from_name(f),
            "r_cm": r_cm,
            "e_r": e_r,
            "metrics": metrics,
        })

    return grouped


def mean_profile(runs):
    r_ref = runs[0]["r_cm"]
    stack = np.array([run["e_r"] for run in runs], dtype=float)
    std = np.std(stack, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(r_ref)
    return r_ref, np.mean(stack, axis=0), std


def metric_summary(runs, metric_name):
    vals = np.array([run["metrics"][metric_name] for run in runs], dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    sem = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, std, sem


def save_radial_summary(case_runs):
    metric_names = ["R50", "R80", "R90", "RMS"]
    lines = []
    lines.append("case,metric,mean_cm,std_cm,sem_cm")

    for case in ["water", "aluminium", "lead"]:
        runs = case_runs.get(case, [])
        if not runs:
            continue
        for m in metric_names:
            mean, std, sem = metric_summary(runs, m)
            lines.append(f"{case},{m},{mean:.6f},{std:.6f},{sem:.6f}")

    out_path = RADIAL_OUTDIR / "lead_radial_summary.csv"
    out_path.write_text("\n".join(lines))


def plot_radial_profiles(case_runs):
    plt.figure()

    for case, label in [("water", "No plate"), ("aluminium", "Aluminium"), ("lead", "Lead")]:
        runs = case_runs.get(case, [])
        if not runs:
            continue
        r_cm, mean_e, _ = mean_profile(runs)
        r_cm, mean_e = rebin_radial_profile(r_cm, mean_e, factor=2)

        y = mean_e / np.sum(mean_e) if np.sum(mean_e) > 0 else mean_e
        plt.plot(r_cm, y, label=label, linewidth=2)

    plt.xlabel("Radius (cm)", fontsize=14)
    plt.xlim(0,8)
    plt.ylabel("Normalised energy deposit", fontsize=14)
    plt.title("Radial energy-deposition profiles at 25 cm depth", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RADIAL_OUTDIR / "lead_material_radial_profiles.png", dpi=300)


def plot_width_comparison(case_runs, metric_name="R80"):
    cases = []
    means = []
    sems = []

    for case in ["water", "aluminium", "lead"]:
        runs = case_runs.get(case, [])
        if not runs:
            continue
        mean, std, sem = metric_summary(runs, metric_name)
        cases.append(case.capitalize())
        means.append(mean)
        sems.append(sem)

    x = np.arange(len(cases))

    plt.figure()
    plt.errorbar(x, means, yerr=sems, fmt="o", capsize=4, elinewidth=1, ms=6)
    plt.xticks(x, cases)
    plt.ylabel(f"{metric_name} beam width at 25 cm depth (cm)", fontsize=14)
    plt.title(f"{metric_name} comparison for water, aluminium, and matched lead", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RADIAL_OUTDIR / f"lead_material_{metric_name}_comparison.png", dpi=300)


def plot_relative_broadening(case_runs, metric_name="R80"):
    water_runs = case_runs.get("water", [])
    aluminium_runs = case_runs.get("aluminium", [])
    lead_runs = case_runs.get("lead", [])

    water_mean, _, water_sem = metric_summary(water_runs, metric_name)

    labels = []
    deltas = []
    sems = []

    for case_name, runs in [("Aluminium", aluminium_runs), ("Lead", lead_runs)]:
        mean, _, sem = metric_summary(runs, metric_name)
        delta = mean - water_mean
        total_sem = np.sqrt(sem**2 + water_sem**2)
        labels.append(case_name)
        deltas.append(delta)
        sems.append(total_sem)

    x = np.arange(len(labels))

    plt.figure()
    plt.errorbar(x, deltas, yerr=sems, fmt="o", capsize=4, elinewidth=1, ms=6)
    plt.axhline(0, linestyle="--")
    plt.xticks(x, labels)
    plt.ylabel(f"Δ{metric_name} beam width relative to no plate (cm)", fontsize = 14)
    plt.title(f"Relative beam broadening for aluminium and matched lead", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RADIAL_OUTDIR / f"lead_material_delta_{metric_name}.png", dpi=300)

# ============================================================
# MAIN
# ============================================================
def main():
    safe_remove_glob(MATCH_OUTDIR, "DoseZ*.csv")
    safe_remove_glob(MATCH_OUTDIR, "*.txt")

    # -------------------------
    # Phase 1: aluminium reference + lead bisection
    # -------------------------
    aluminium_peaks, aluminium_stats = compute_al_reference()
    z_al_mean, z_al_std, z_al_sem = aluminium_stats

    best_hl, lead_final_result, history = find_matched_lead_hl_bisection(z_al_mean)

    save_lead_match_summary(aluminium_stats, best_hl, lead_final_result, history)
    plot_lead_match(aluminium_stats, history)

    print("\n=== LEAD THICKNESS MATCHING ===")
    print(f"Aluminium reference peak: {z_al_mean:.4f} ± {z_al_sem:.4f} cm (SEM)")
    print(f"Matched lead HL = {best_hl:.4f} cm")
    print(f"Matched lead physical thickness = {2 * best_hl:.4f} cm")
    print(
        f"Lead final peak: {lead_final_result['mean']:.4f} ± {lead_final_result['sem']:.4f} cm (SEM)  "
        f"delta={lead_final_result['mean'] - z_al_mean:+.4f} cm"
    )

    # -------------------------
    # Phase 2: radial comparison
    # -------------------------
    generate_and_run_lead_radial_cases(best_hl)

    case_runs = load_radial_case_runs()
    save_radial_summary(case_runs)
    plot_radial_profiles(case_runs)
    plot_width_comparison(case_runs, metric_name=WIDTH_METRIC)
    plot_relative_broadening(case_runs, metric_name=WIDTH_METRIC)

    print("\n=== RADIAL COMPARISON ===")
    for case in ["water", "aluminium", "lead"]:
        runs = case_runs.get(case, [])
        if not runs:
            continue
        mean, std, sem = metric_summary(runs, WIDTH_METRIC)
        print(f"{case.capitalize():>5s}: {WIDTH_METRIC} = {mean:.4f} ± {sem:.4f} cm (SEM)")

    print("\nSaved in lead_match_runs/:")
    print(" - lead_match_summary.csv")
    print(" - lead_bisection_history.csv")
    print(" - lead_bragg_match.png")

    print("\nSaved in lead_radial_runs/:")
    print(" - lead_radial_summary.csv")
    print(" - lead_material_radial_profiles.png")
    print(f" - lead_material_{WIDTH_METRIC}_comparison.png")
    print(f" - lead_material_delta_{WIDTH_METRIC}.png")


if __name__ == "__main__":
    main()