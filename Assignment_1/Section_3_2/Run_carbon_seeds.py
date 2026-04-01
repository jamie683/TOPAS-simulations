import re
import subprocess
from pathlib import Path
import time

TOPAS_EXE = "/home/jamie/shellScripts/topas"
CARBON_TXT = Path("A2_carbon.txt")
BASE_TEXT = CARBON_TXT.read_text()

# ----- MATCHED ENERGY -----
E_STAR = 5102  # MeV total
SEEDS = [1, 2, 3, 4, 5]

# Histories
HIST_ESTAR = 40000

# ----- OPTIONAL SLOPE RUNS -----
DO_SLOPE = True
DELTA_E = 25
HIST_SLOPE = 30000

# ----- Output naming -----
OUTDIR = Path(".")
OUTDIR.mkdir(exist_ok=True)

def replace_line(text: str, pattern: str, new_line: str) -> str:
    rgx = re.compile(pattern, re.MULTILINE)
    if not rgx.search(text):
        raise RuntimeError(f"Pattern not found: {pattern}")
    return rgx.sub(new_line, text, count=1)

def run_topas(param_file: Path) -> None:
    t0 = time.time()
    subprocess.run([TOPAS_EXE, str(param_file)], check=True)
    print(f"TOPAS finished {param_file.name} in {time.time()-t0:.1f} s")

def run_carbon(E_MeV: int, histories: int, seed: int) -> Path:
    out_base = f"DoseZ_C_{E_MeV}MeV_seed{seed:03d}"
    run_txt = OUTDIR / f"run_{out_base}.txt"
    out_csv = OUTDIR / f"{out_base}.csv"

    if out_csv.exists():
        out_csv.unlink()

    txt = BASE_TEXT
    txt = replace_line(txt, r"^d:So/Beam/BeamEnergy\s*=.*$", f"d:So/Beam/BeamEnergy = {E_MeV} MeV")
    txt = replace_line(txt, r"^i:So/Beam/NumberOfHistoriesInRun\s*=.*$", f"i:So/Beam/NumberOfHistoriesInRun = {histories}")
    txt = replace_line(txt, r"^i:Ts/Seed\s*=.*$", f"i:Ts/Seed = {seed}")
    txt = replace_line(txt, r'^s:Sc/Dose/OutputFile\s*=.*$', f's:Sc/Dose/OutputFile = "{out_base}"')

    run_txt.write_text(txt)
    run_topas(run_txt)

    if not out_csv.exists():
        raise RuntimeError(f"Expected output not created: {out_csv}")

    return out_csv

def main():
    print(f"Running carbon seeds at E*={E_STAR} MeV, histories={HIST_ESTAR}, seeds={SEEDS}")
    for s in SEEDS:
        print(f"  -> E={E_STAR} MeV, seed={s}")
        run_carbon(E_STAR, HIST_ESTAR, s)

    if DO_SLOPE:
        for E in (E_STAR - DELTA_E, E_STAR + DELTA_E):
            print(f"\nRunning slope points at E={E} MeV, histories={HIST_SLOPE}, seeds={SEEDS}")
            for s in SEEDS:
                print(f"  -> E={E} MeV, seed={s}")
                run_carbon(E, HIST_SLOPE, s)

    print("\nDone. Carbon CSVs produced with names like:")
    print(f"  DoseZ_C_{E_STAR}MeV_seed001.csv")

if __name__ == "__main__":
    main()
