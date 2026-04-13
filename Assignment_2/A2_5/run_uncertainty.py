"""
A2_5 — Monte Carlo uncertainty driver for the single monoenergetic proton beam.

Re-runs the optimised proton plan produced by ``optimise_proton_energy.py`` with
``N_REPLICATES`` independent RNG seeds, builds per-structure cumulative DVHs
directly from the TOPAS voxel-level dose CSVs, and aggregates replicates into
mean ± 1σ bands.

Template expected (auto-detected via glob):
    A2_5/output/optimised_proton_<E>MeV.txt

Outputs (under ``A2_5/output/uncertainty/``):
    replicate_{kk}.txt   — TOPAS input for seed k
    replicate_{kk}.csv   — TOPAS voxel dose output
    replicate_{kk}.npz   — per-replicate cached DVHs
    aggregated.npz       — ensemble mean / 1σ / min / max (+ D-value stats)

Re-runs are cached: replicates whose ``.npz`` is already present are loaded
from disk. Pass ``--force`` to re-run every replicate or ``--rescore`` to
recompute DVHs from existing CSVs without re-running TOPAS.

Usage:
    python3 A2_5/run_uncertainty.py
    python3 A2_5/run_uncertainty.py --force
    python3 A2_5/run_uncertainty.py --rescore
"""

import glob
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Shared DVH utilities + RTStruct mask builders live under A2_4/.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "A2_4"))
import dvh_utils  # noqa: E402
from sweep_beams import load_ct_geometry, build_structure_masks  # noqa: E402


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
N_REPLICATES = 10
HISTORIES_PER_REPLICATE = 1_000_000
SEEDS = list(range(1, N_REPLICATES + 1))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
UNCERT_DIR = os.path.join(OUTPUT_DIR, "uncertainty")
CT_DIR     = os.path.join(PROJECT_ROOT, "CTData")

STRUCTURE_KEYS = ("tumour", "ptv", "lung_r", "lung_l", "heart", "cord", "body")

FORCE_FLAG   = "--force"
RESCORE_FLAG = "--rescore"

STRUCT_DISPLAY = {
    "tumour": "GTVp",
    "ptv":    "PTV",
    "lung_r": "Right Lung",
    "lung_l": "Left Lung",
    "heart":  "Heart",
    "cord":   "Spinal Cord",
    "body":   "Body",
}


def find_template():
    """Auto-detect the optimised proton template via glob."""
    pattern = os.path.join(OUTPUT_DIR, "optimised_proton_*MeV.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]  # latest if multiple


def replicate_basename_rel(k):
    return os.path.join("A2_5", "output", "uncertainty", f"replicate_{k:02d}")


def replicate_paths(k):
    base_rel = replicate_basename_rel(k)
    return {
        "basename_rel": base_rel,
        "txt": os.path.join(PROJECT_ROOT, base_rel + ".txt"),
        "csv": os.path.join(PROJECT_ROOT, base_rel + ".csv"),
        "npz": os.path.join(PROJECT_ROOT, base_rel + ".npz"),
    }


def generate_replicate_inputs(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()
    for k, seed in enumerate(SEEDS):
        paths = replicate_paths(k)
        new_text = dvh_utils.rewrite_topas_for_replicate(
            template_text,
            seed=seed,
            histories=HISTORIES_PER_REPLICATE,
            output_basename=paths["basename_rel"],
        )
        os.makedirs(os.path.dirname(paths["txt"]), exist_ok=True)
        with open(paths["txt"], "w", encoding="utf-8") as f:
            f.write(new_text)


def build_masks():
    print("\nBuilding RTStruct masks ...")
    geom = load_ct_geometry(CT_DIR)
    masks, info = build_structure_masks(geom)
    print(f"Mask source: {info.get('source', 'unknown')}")
    for key in STRUCTURE_KEYS:
        print(f"  {key:<18s} voxels={len(masks.get(key, []))}")
    return masks


def main():
    force   = FORCE_FLAG   in sys.argv
    rescore = RESCORE_FLAG in sys.argv
    os.makedirs(UNCERT_DIR, exist_ok=True)

    template_path = find_template()

    print("=" * 78)
    print("A2_5 — PROTON SINGLE-BEAM MONTE CARLO UNCERTAINTY")
    print("=" * 78)
    print(f"Replicates          : {N_REPLICATES}")
    print(f"Histories/replicate : {HISTORIES_PER_REPLICATE:,}")
    print(f"Seeds               : {SEEDS}")
    print(f"Template            : {template_path or '(not found)'}")
    print(f"Output dir          : {UNCERT_DIR}")
    if force:
        print("Mode                : FORCE (re-run TOPAS + rescore)")
    elif rescore:
        print("Mode                : RESCORE (recompute DVHs from existing CSVs)")
    print("-" * 78)

    if not template_path or not os.path.isfile(template_path):
        print("Template not found. Run A2_5/optimise_proton_energy.py first.")
        return

    generate_replicate_inputs(template_path)
    masks = build_masks()
    centres, edges = dvh_utils.common_dose_grid()

    replicate_dvhs = []
    for k, seed in enumerate(SEEDS):
        paths = replicate_paths(k)

        if (not force) and (not rescore) and os.path.isfile(paths["npz"]):
            _, _, _, dvh = dvh_utils.load_replicate_npz(paths["npz"])
            replicate_dvhs.append(dvh)
            print(f"[{k + 1:02d}/{N_REPLICATES}] seed={seed}  (cached)")
            continue

        if rescore and os.path.isfile(paths["csv"]):
            print(f"[{k + 1:02d}/{N_REPLICATES}] seed={seed}  rescoring ... ",
                  end="", flush=True)
        else:
            t0 = time.time()
            print(f"[{k + 1:02d}/{N_REPLICATES}] seed={seed}  TOPAS ... ",
                  end="", flush=True)
            dvh_utils.run_topas(paths["txt"], project_root=PROJECT_ROOT)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.0f}s). Scoring DVH ... ", end="", flush=True)

        dose_dict = dvh_utils.load_topas_dose_csv(paths["csv"])
        plan_dose_max, dvh = dvh_utils.compute_replicate_dvhs(
            dose_dict, masks, STRUCTURE_KEYS, edges)
        dvh_utils.save_replicate_npz(
            paths["npz"], seed, plan_dose_max, centres, dvh)
        replicate_dvhs.append(dvh)
        print(f"plan_max={plan_dose_max:.3e} Gy")

    agg = dvh_utils.aggregate_replicates(replicate_dvhs, STRUCTURE_KEYS, centres)
    agg_path = os.path.join(UNCERT_DIR, "aggregated.npz")
    dvh_utils.save_aggregated_npz(agg_path, centres, agg)
    print(f"\nAggregated DVHs saved: {agg_path}")

    # Console summary
    print()
    print("--- Ensemble DVH summary (dose as % of replicate max) ---")
    print(f'{"Structure":<14s} {"D95":>14s} {"D50":>14s} {"D02":>14s} {"Dmax":>14s}')
    print("-" * 74)
    for key in STRUCTURE_KEYS:
        stats = agg[key]

        def fmt(dname):
            m = stats.get(f"{dname}_mean", float("nan"))
            s = stats.get(f"{dname}_std",  float("nan"))
            if not np.isfinite(m):
                return "      —       "
            return f"{m:6.2f} ± {s:4.2f}"
        print(f"{STRUCT_DISPLAY.get(key, key):<14s} {fmt('D95'):>14s} "
              f"{fmt('D50'):>14s} {fmt('D02'):>14s} {fmt('Dmax'):>14s}")

    print("\nDone.")
    print("Next: python3 A2_5/analyse_dvh_uncertainty.py")


if __name__ == "__main__":
    main()
