import re
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

TOPAS_EXE = "/home/jamie/shellScripts/topas"

CARBON_TXT = Path("A2_carbon.txt")

# ---------------------------
# Timing helper
# ---------------------------
def run_topas(param_file: Path):
    t0 = time.time()
    subprocess.run([TOPAS_EXE, str(param_file)], check=True)
    print(f"TOPAS finished {param_file} in {time.time() - t0:.1f} s")

# ---------------------------
# Binning
# ---------------------------
Z_BIN_WIDTH_CM = 0.05  # 40 cm / 800 bins; keep consistent with scorer

# ---------------------------
# Text replacement helper
# ---------------------------
def replace_line(text: str, pattern: str, new_line: str) -> str:
    rgx = re.compile(pattern, re.MULTILINE)
    if not rgx.search(text):
        raise RuntimeError(f"Pattern not found: {pattern}")
    return rgx.sub(new_line, text, count=1)

# ---------------------------
# Distal range helpers
# ---------------------------
def distal_R(depth_cm: np.ndarray, dose: np.ndarray, level: float) -> float:
    y = dose / np.max(dose)
    i_peak = int(np.argmax(y))
    d = depth_cm[i_peak:]
    yy = y[i_peak:]

    idx = np.where((yy[:-1] >= level) & (yy[1:] < level))[0]
    if len(idx) == 0:
        return float("nan")

    i = idx[0]
    x0, x1 = d[i], d[i + 1]
    y0, y1 = yy[i], yy[i + 1]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))

def compute_Rs(depth_cm: np.ndarray, dose: np.ndarray):
    R80 = distal_R(depth_cm, dose, level=0.8)
    R90 = distal_R(depth_cm, dose, level=0.9)
    return R80, R90

# ---------------------------
# CSV loading
# ---------------------------
def load_depth_dose_from_csv(csv_path: Path):
    data = np.loadtxt(csv_path, comments="#", delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    iz = data[:, 2]
    dose = data[:, 3]
    depth_cm = (iz + 0.5) * Z_BIN_WIDTH_CM
    return depth_cm, dose

# ---------------------------
# Peak extraction: same idea as Section 3.1
# ---------------------------
def peak_depth_parabolic(depth_cm: np.ndarray, dose: np.ndarray) -> float:
    """
    Sub-bin Bragg peak estimate using a quadratic through (i-1, i, i+1)
    around the maximum dose bin.
    """
    i = int(np.argmax(dose))

    if i == 0 or i == len(dose) - 1:
        return float(depth_cm[i])

    x1, x2, x3 = depth_cm[i - 1], depth_cm[i], depth_cm[i + 1]
    y1, y2, y3 = dose[i - 1], dose[i], dose[i + 1]

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        return float(x2)

    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom

    if A == 0:
        return float(x2)

    x_vertex = -B / (2 * A)
    return float(x_vertex)

# ---------------------------
# One TOPAS run at one carbon energy
# ---------------------------
def run_one_energy(base_text: str, E_MeV: int, histories: int, out_base: str, seed: int = 1):
    run_txt = Path(f"run_{out_base}.txt")
    out_csv = Path(f"{out_base}.csv")

    # Delete old outputs
    for p in Path(".").glob(f"{out_base}.csv"):
        p.unlink()
    for p in Path(".").glob(f"{out_base}_*.csv"):
        p.unlink()

    txt = base_text
    txt = replace_line(txt, r"^d:So/Beam/BeamEnergy\s*=.*$", f"d:So/Beam/BeamEnergy = {E_MeV} MeV")
    txt = replace_line(txt, r"^i:So/Beam/NumberOfHistoriesInRun\s*=.*$", f"i:So/Beam/NumberOfHistoriesInRun = {histories}")
    txt = replace_line(txt, r"^i:Ts/Seed\s*=.*$", f"i:Ts/Seed = {seed}")
    txt = replace_line(txt, r'^s:Sc/Dose/OutputFile\s*=.*$', f's:Sc/Dose/OutputFile = "{out_base}"')

    run_txt.write_text(txt)
    run_topas(run_txt)

    if not out_csv.exists():
        raise RuntimeError(f"Expected output not created: {out_csv}")

    return load_depth_dose_from_csv(out_csv)

# ---------------------------
# CONFIG
# ---------------------------
PROTON_E_MEV = 220

# Proton target from Section 3.1
TARGET_ZPEAK_CM = 30.3736

# Bracketing start near previous R80-based result (previous approach gave ~5120 MeV, but that was not correct approach for assignment)
E_LOW0 = 5000
E_HIGH0 = 5250
BRACKET_STEP = 50  # MeV

# Convergence criteria
EPS_Z_CM = 0.05    # stop if |z_peak,C - z_peak,p| < 0.05 cm
EPS_E_MEV = 10     # or if bracket is narrower than 10 MeV

# Histories policy (adaptive)
HIST_FAR = 1000
HIST_MID = 3000
HIST_NEAR = 8000
HIST_FINAL = 20000

FAR_THRESH_CM = 0.5
NEAR_THRESH_CM = 0.1

# ---------------------------
# Carbon peak cache
# ---------------------------
_carbon_cache = {}  # (E, histories) -> dict of results

carbon_text = CARBON_TXT.read_text()

def carbon_metrics(E_MeV: int, histories: int) -> dict:
    """
    Run/load one carbon simulation and extract:
      - z_peak (parabolic)
      - R80
      - R90
    Cached by (E, histories).
    """
    key = (E_MeV, histories)
    if key in _carbon_cache:
        return _carbon_cache[key]

    out_base = f"DoseZ_C_{E_MeV}MeV"
    depth_c, dose_c = run_one_energy(
        carbon_text,
        E_MeV,
        histories,
        out_base=out_base,
        seed=E_MeV + 7
    )

    z_peak = peak_depth_parabolic(depth_c, dose_c)
    R80, R90 = compute_Rs(depth_c, dose_c)

    result = {
        "depth_cm": depth_c,
        "dose": dose_c,
        "z_peak": z_peak,
        "R80": R80,
        "R90": R90,
    }
    _carbon_cache[key] = result
    return result

def carbon_peak_depth(E_MeV: int, histories: int) -> float:
    return carbon_metrics(E_MeV, histories)["z_peak"]

# ---------------------------
# Matching function
# ---------------------------
def f(E: int, histories: int) -> float:
    z_peak = carbon_peak_depth(E, histories)
    return z_peak - TARGET_ZPEAK_CM

# ---------------------------
# 1.) Bracket the solution
# ---------------------------
E_low = int(E_LOW0)
E_high = int(E_HIGH0)

f_low = f(E_low, HIST_FAR)
f_high = f(E_high, HIST_FAR)

expand_count = 0
while not (f_low < 0 and f_high > 0):
    expand_count += 1
    if expand_count > 20:
        raise RuntimeError(
            f"Failed to bracket peak-match solution after expansions. "
            f"Current: E_low={E_low}, f_low={f_low:+.4f}; "
            f"E_high={E_high}, f_high={f_high:+.4f}"
        )

    # Both too shallow -> move energies up
    if f_low < 0 and f_high < 0:
        E_low += BRACKET_STEP
        E_high += BRACKET_STEP

    # Both too deep -> move energies down
    elif f_low > 0 and f_high > 0:
        E_low -= BRACKET_STEP
        E_high -= BRACKET_STEP
        if E_low <= 0:
            raise RuntimeError("Bracket shifted below 0 MeV. Check assumptions / geometry.")

    # Mixed signs but reversed order -> swap
    elif f_low > 0 and f_high < 0:
        E_low, E_high = E_high, E_low

    f_low = f(E_low, HIST_FAR)
    f_high = f(E_high, HIST_FAR)

print("\nBracket found:")
print(f"  E_low={E_low} MeV  -> z_peak - target = {f_low:+.4f} cm")
print(f"  E_high={E_high} MeV -> z_peak - target = {f_high:+.4f} cm")

# ---------------------------
# 2.) Bisection with adaptive histories
# ---------------------------
iter_count = 0
history_log = []

while True:
    iter_count += 1
    E_mid = int(round(0.5 * (E_low + E_high)))

    # Stage 1: cheap estimate
    z_mid = carbon_peak_depth(E_mid, HIST_FAR)
    deltaZ = z_mid - TARGET_ZPEAK_CM
    hist_used = HIST_FAR

    # Stage 2: if within 0.5 cm, upgrade
    if abs(deltaZ) <= FAR_THRESH_CM:
        z_mid = carbon_peak_depth(E_mid, HIST_MID)
        deltaZ = z_mid - TARGET_ZPEAK_CM
        hist_used = HIST_MID

    # Stage 3: if within 0.1 cm, upgrade again
    if abs(deltaZ) <= NEAR_THRESH_CM:
        z_mid = carbon_peak_depth(E_mid, HIST_NEAR)
        deltaZ = z_mid - TARGET_ZPEAK_CM
        hist_used = HIST_NEAR

    history_log.append((iter_count, E_low, E_high, E_mid, hist_used, z_mid, deltaZ))

    print(
        f"iter {iter_count:02d}: "
        f"E_mid={E_mid} MeV  "
        f"z_peak={z_mid:.4f} cm  "
        f"delta={deltaZ:+.4f} cm  "
        f"(hist={hist_used})  "
        f"bracket=[{E_low},{E_high}]"
    )

    # Stopping criteria
    if abs(deltaZ) < EPS_Z_CM or (E_high - E_low) <= EPS_E_MEV:
        E_match = float(E_mid)
        break

    # Bisection update
    if deltaZ < 0:
        # Carbon peak too shallow -> need more energy
        E_low = E_mid
    else:
        # Carbon peak too deep -> need less energy
        E_high = E_mid

print(f"\nBisection match (Bragg peak): {E_match:.1f} MeV total  ({E_match/12.0:.1f} MeV/u)")

# ---------------------------
# 3.) Final high-stat verification
# ---------------------------
final_E = int(round(E_match))
final_metrics = carbon_metrics(final_E, HIST_FINAL)

z_final = final_metrics["z_peak"]
R80_final = final_metrics["R80"]
R90_final = final_metrics["R90"]

print(f"\nFinal check @ {final_E} MeV with {HIST_FINAL} histories:")
print(f"  Carbon z_peak = {z_final:.4f} cm (target {TARGET_ZPEAK_CM:.4f})  delta={z_final - TARGET_ZPEAK_CM:+.4f} cm")
print(f"  Carbon R80    = {R80_final:.4f} cm")
print(f"  Carbon R90    = {R90_final:.4f} cm")

# ---------------------------
# 4.) Plot convergence history
# ---------------------------
Emids = [r[3] for r in history_log]
deltas = [r[6] for r in history_log]

plt.figure(figsize=(8, 5))
plt.plot(Emids, deltas, marker="o")
plt.axhline(0.0, linestyle="--")
plt.xlabel("E_mid (MeV total)")
plt.ylabel("z_peak,C(E) - z_peak,p (cm)")
plt.title("Bisection convergence on carbon energy (Bragg peak depth)")
plt.grid(True)
plt.tight_layout()
plt.savefig("section2_bisection_convergence_peakmatch.png", dpi=300)
plt.show()
