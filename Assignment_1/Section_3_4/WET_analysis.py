import re
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# User settings
# ----------------------------
TOPAS_EXE = "/home/jamie/shellScripts/topas"
BASE_TXT = Path("A3_proton.txt")
OUTDIR = Path("wet_runs")
OUTDIR.mkdir(exist_ok=True)

# Beam/score geometry assumptions
Z_BIN_WIDTH_CM = 0.05

# Phantom geometry
PHANTOM_CENTRE_CM = 26.0
PHANTOM_HALF_LENGTH_CM = 20.0
PHANTOM_FRONT_FACE_CM = PHANTOM_CENTRE_CM - PHANTOM_HALF_LENGTH_CM  # = 6.0 cm

SEEDS = [101, 202, 303]

SCORER_BASENAME = "DoseZ"

PLATE_MATERIAL = '"G4_Al"'

# Exclude insert positions where the distal edge is within this distance of the water Bragg peak
EXCLUSION_MARGIN_CM = 5.0

# ----------------------------
# Helpers
# ----------------------------
def sem(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(len(x)))

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

def load_dose_csv(csv_path: Path):
    d = np.loadtxt(csv_path, comments="#", delimiter=",")
    if d.size == 0:
        raise ValueError(f"{csv_path} has no numeric data. Check TOPAS run/scorer output.")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.shape[1] < 4:
        raise ValueError(f"{csv_path} has {d.shape[1]} columns; expected >=4.")
    iz = d[:, 2].astype(float)
    dose = d[:, 3].astype(float)
    z_cm = (iz + 0.5) * Z_BIN_WIDTH_CM
    return z_cm, dose

def bragg_peak_z_parabolic(z_cm: np.ndarray, dose: np.ndarray, window: int = 4) -> float:
    z_cm = np.asarray(z_cm, float)
    dose = np.asarray(dose, float)

    i = int(np.argmax(dose))

    lo = max(0, i - window)
    hi = min(len(z_cm), i + window + 1)

    z_fit = z_cm[lo:hi]
    d_fit = dose[lo:hi]

    # need at least 3 points
    if len(z_fit) < 3:
        return float(z_cm[i])

    a, b, c = np.polyfit(z_fit, d_fit, 2)

    # reject bad fits
    if a >= 0:
        return float(z_cm[i])

    z_peak = -b / (2 * a)

    # reject vertices outside the fit window
    if z_peak < z_fit.min() or z_peak > z_fit.max():
        return float(z_cm[i])

    return float(z_peak)

def transz_to_depth_cm(transz_cm: float) -> float:
    """
    Convert plate centre world z-position to depth from phantom entrance.
    """
    return 20.0 + transz_cm

def get_param_cm(text: str, key: str) -> float:
    pattern = rf"{re.escape(key)}\s*=\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*cm"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not parse {key} from TOPAS file.")
    return float(m.group(1))

def rebin_mean(x, y, factor):
    n = len(y) // factor
    x2 = x[:n*factor].reshape(n, factor).mean(axis=1)
    y2 = y[:n*factor].reshape(n, factor).mean(axis=1)
    return x2, y2

# ----------------------------
# Main
# ----------------------------
def main():
    base_text = BASE_TXT.read_text()
    PLATE_HALF_LENGTH_CM = get_param_cm(base_text, "d:Ge/Plate/HL")
    # Clean old files
    for f in OUTDIR.glob("DoseZ*.csv"):
        f.unlink()
    for f in OUTDIR.glob("run_*.txt"):
        f.unlink()
            
    # -------------------------
    # 1) Baseline: plate replaced by water
    # -------------------------
    baseline_peaks = []
    baseline_curves = []

    for seed in SEEDS:
        txt = base_text
        txt = set_param(txt, "i:Ts/Seed", str(seed))
        txt = set_param(txt, "s:Ge/Plate/Material", '"G4_WATER"')
        txt = set_param(txt, "s:Sc/Dose/OutputFile", f'"{SCORER_BASENAME}_water_s{seed}"')

        runfile = OUTDIR / f"run_water_s{seed}.txt"
        runfile.write_text(txt)
        run_topas(runfile)

        csv = OUTDIR / f"{SCORER_BASENAME}_water_s{seed}.csv"
        z, dose = load_dose_csv(csv)
        baseline_peaks.append(bragg_peak_z_parabolic(z, dose))
        baseline_curves.append(dose)

    baseline_curves = np.array(baseline_curves)
    dose_mean = baseline_curves.mean(axis=0)
    dose_norm = dose_mean / dose_mean.max()
    z_baseline = z
    baseline_peaks = np.array(baseline_peaks, float)
    zpk_water_mean = float(baseline_peaks.mean())
    zpk_water_sem = sem(baseline_peaks)

    # -------------------------
    # 2) Automatically design insert scan around Bragg peak
    # -------------------------

    # Convert peak depth to phantom-relative TransZ
    transz_peak = zpk_water_mean - 20.0

    # coarse sampling across phantom
    coarse = np.array([-18, -13, -8, -3, 2], dtype=float)

    # dense sampling around the Bragg peak
    dense_lo = transz_peak - 3.0
    dense_hi = transz_peak + 2.0
    dense = np.arange(dense_lo, dense_hi, 1.5)

    # remove coarse points that fall inside dense region
    coarse = coarse[(coarse < dense_lo) | (coarse > dense_hi)]

    # combine, remove exact duplicates, and sort
    all_pos = np.sort(np.unique(np.round(np.concatenate((coarse, dense)), 6)))

    # remove near-neighbours
    filtered = [all_pos[0]]
    for x in all_pos[1:]:
        if x - filtered[-1] >= 0.5:
            filtered.append(x)

    INSERT_TRANSZ_CM = np.array(filtered)

    INSERT_TRANSZ_CM = np.sort(INSERT_TRANSZ_CM)
    print("Detected Bragg peak depth:", zpk_water_mean)
    print("Peak TransZ coordinate:", transz_peak)
    print("Insert scan positions:", INSERT_TRANSZ_CM)

    # -------------------------
    # 2) Plate runs: vary TransZ and compute WET
    # -------------------------
    # rows: (transz, depth_cm, wet_mean, wet_sem, zpk_insert_mean, zpk_insert_sem, dist_to_peak_cm)
    results = []

    for transz in INSERT_TRANSZ_CM:
        peaks = []

        for seed in SEEDS:
            txt = base_text
            txt = set_param(txt, "i:Ts/Seed", str(seed))
            txt = set_param(txt, "s:Ge/Plate/Material", PLATE_MATERIAL)
            txt = set_param(txt, "d:Ge/Plate/TransZ", f"{transz} cm")
            txt = set_param(txt, "s:Sc/Dose/OutputFile", f'"{SCORER_BASENAME}_ins_z{transz:+.2f}_s{seed}"')

            runfile = OUTDIR / f"run_ins_z{transz:+.2f}_s{seed}.txt"
            runfile.write_text(txt)
            run_topas(runfile)

            csv = OUTDIR / f"{SCORER_BASENAME}_ins_z{transz:+.2f}_s{seed}.csv"
            z, dose = load_dose_csv(csv)
            peaks.append(bragg_peak_z_parabolic(z, dose))

        peaks = np.array(peaks, float)
        zpk_ins_mean = float(peaks.mean())
        zpk_ins_sem = sem(peaks)

        wet = zpk_water_mean - zpk_ins_mean
        wet_sem = float(np.sqrt(zpk_water_sem**2 + zpk_ins_sem**2))

        depth_cm = transz_to_depth_cm(transz)
        dist_to_peak_cm = zpk_water_mean - depth_cm

        results.append((transz, depth_cm, wet, wet_sem, zpk_ins_mean, zpk_ins_sem, dist_to_peak_cm))

    results = np.array(results, float)

    # -------------------------
    # 3) Save results
    # -------------------------
    depth_cm = results[:, 1]
    wet = results[:, 2]
    wet_sem = results[:, 3]
    dist_to_peak = results[:, 6]

    # sort by physical depth
    order = np.argsort(depth_cm)
    depth_cm = depth_cm[order]
    wet = wet[order]
    wet_sem = wet_sem[order]
    dist_to_peak = dist_to_peak[order]

    # Distal edge of insert relative to beam direction
    distal_edge_depth_cm = depth_cm + PLATE_HALF_LENGTH_CM

    # Distance from distal edge of insert to water Bragg peak
    distal_edge_to_peak_cm = zpk_water_mean - distal_edge_depth_cm

    # Valid WET points: insert must end sufficiently upstream of Bragg peak
    valid_mask = distal_edge_to_peak_cm > EXCLUSION_MARGIN_CM

    # Plateau WET summary (this is what you report)
    wet_valid = wet[valid_mask]

    if len(wet_valid) < 2:
        raise ValueError("Not enough valid WET points after Bragg-peak exclusion.")

    wet_mean_report = float(np.mean(wet_valid))
    wet_std_report = float(np.std(wet_valid, ddof=1))

    results_sorted = np.column_stack([
        depth_cm,
        wet,
        wet_sem,
        dist_to_peak,
        distal_edge_depth_cm,
        distal_edge_to_peak_cm,
        valid_mask.astype(int)
    ])

    np.savetxt(
        OUTDIR / "wet_vs_position.csv",
        results_sorted,
        delimiter=",",
        header="insert_depth_cm,wet_cm,wet_sem_cm,centre_to_peak_cm,distal_edge_depth_cm,distal_edge_to_peak_cm,used_in_final_average",
        comments=""
    )

    # -------------------------
    # 4) WET vs insert depth
    # -------------------------
    plt.figure()

    # included points
    plt.errorbar(
        depth_cm[valid_mask], wet[valid_mask], yerr=wet_sem[valid_mask],
        fmt="o", capsize=4, elinewidth=1, ms=5, label="Included in WET average"
    )

    # excluded points
    plt.errorbar(
        depth_cm[~valid_mask], wet[~valid_mask], yerr=wet_sem[~valid_mask],
        fmt="s", capsize=4, elinewidth=1, ms=5, alpha=0.7, label="Excluded near Bragg peak"
    )

    plt.axvline(zpk_water_mean, linestyle="--",
                label=f"Water Bragg peak = {zpk_water_mean:.2f} cm")

    plt.xlabel("Insert centre depth in water (cm)", fontsize=14)
    plt.ylabel("WET from Bragg peak shift (cm water)", fontsize=14)
    plt.title("WET vs insert depth", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "wet_vs_depth_global.png", dpi=300)

    # -------------------------
    # 5) WET vs insert depth (zoom around Bragg peak) with secondary axis showing Bragg curve
    # -------------------------

    # Rebin for smooth Bragg plot
    z_plot, dose_plot = rebin_mean(z_baseline, dose_mean, factor=4)
    dose_norm_plot = dose_plot / dose_plot.max()

    # zoom window around Bragg peak
    mask = (depth_cm > zpk_water_mean - 5) & (depth_cm < zpk_water_mean + 2)

    plt.figure()
    ax1 = plt.gca()

    zoom_included = mask & valid_mask
    zoom_excluded = mask & (~valid_mask)

    ax1.errorbar(depth_cm[zoom_included], wet[zoom_included], yerr=wet_sem[zoom_included],
                fmt="o", capsize=4, elinewidth=1, ms=4,
                label="Included in WET average")

    ax1.errorbar(depth_cm[zoom_excluded], wet[zoom_excluded], yerr=wet_sem[zoom_excluded],
                fmt="s", capsize=4, elinewidth=1, ms=4, alpha=0.7,
                label="Excluded near Bragg peak")

    ax1.set_xlabel("Insert centre depth in water (cm)", fontsize=14)
    ax1.set_ylabel("WET from Bragg peak shift (cm water)", fontsize=14)
    ax1.set_title("WET behaviour near the Bragg peak", fontsize=16)
    ax1.grid(True, alpha=0.3)

    # secondary axis: Bragg curve
    ax2 = ax1.twinx()
    ax2.plot(z_plot, dose_norm_plot, color="C1",
            linewidth=2, alpha=0.8,
            label="Normalized depth–dose")

    ax2.set_ylabel("Normalized dose", fontsize=14)

    ax1.axvline(zpk_water_mean, linestyle="--", color="black")

    # merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.savefig(OUTDIR / "wet_vs_depth_zoom.png", dpi=300)

    '''# -------------------------
    # 5) WET vs distance to Bragg peak
    # -------------------------
    plt.figure()
    plt.errorbar(dist_to_peak, wet, yerr=wet_sem, fmt="o", capsize=4, elinewidth=1, ms=5)
    plt.axvline(0.0, linestyle="--", label="Insert centre at water Bragg peak")
    plt.xlabel("Distance from insert centre to water Bragg peak (cm)")
    plt.ylabel("WET from Bragg peak shift (cm water)")
    plt.title("WET vs distance to water Bragg peak")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "wet_vs_distance_to_peak.png", dpi=300)'''

    # -------------------------
    # 6) Print summary
    # -------------------------
    print(f"Water Bragg peak: {zpk_water_mean:.4f} ± {zpk_water_sem:.4f} cm (SEM, N={len(SEEDS)})")
    print("Depth(cm)   TransZ(cm)   WET(cm)    SEM(cm)   Dist-to-peak(cm)")
    results_print = results[np.argsort(results[:, 1])]
    for row in results_print:
        t, d, w, u, _, _, dp = row
        print(f"{d:8.2f}   {t:+9.2f}   {w:8.4f}   {u:7.4f}   {dp:10.4f}")
    print()
    print("Representative", PLATE_MATERIAL, "WET (excluding points near Bragg peak):")
    print(f"WET = {wet_mean_report:.4f} ± {wet_std_report:.4f} cm water")
    print(f"Computed from {len(wet_valid)} valid insert positions")
    #print(f"Exclusion rule: distal edge at least {EXCLUSION_MARGIN_CM:.2f} cm upstream of water Bragg peak")
    print("Accepted insert depths (cm):", depth_cm[valid_mask])
    #print("Rejected insert depths (cm):", depth_cm[~valid_mask])

if __name__ == "__main__":
    main()
