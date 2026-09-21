"""
Known-material validation of the screening framework (rules fixed in
docs/descriptors.md before any comparison was made).

Joins data/literature_benchmark.csv (B: literature) with
results/stability_windows.csv (A: computed) and assigns each literature row one
of:

    framework agrees
    framework agrees - kinetically stable beyond the thermodynamic bound
    framework disagrees
    insufficient evidence

Output: results/validation.csv, printed summary, Spearman rank correlation for
the rows with a numeric onset (stated as consistency, not significance).

Run: python adib_validate.py   (after adib_screening.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
LIT = ROOT / "data" / "literature_benchmark.csv"
WIN = ROOT / "results" / "stability_windows.csv"
OUT = ROOT / "results" / "validation.csv"

AGREE_TOL = 0.2        # V; allowance for product dissolution/activity and DFT error
KINETIC_MARGIN = 0.5   # V; observed stability this far beyond phi_ox is reported separately
SCORED = {"verbatim-full-text", "verbatim-abstract"}


def verdict(r):
    if r.verification not in SCORED:
        return "insufficient evidence", "source not verified"
    if not isinstance(r.mp_formula, str) or not r.mp_formula:
        return "insufficient evidence", "no single MP representation"
    if pd.isna(r.phi_ox):
        return "insufficient evidence", "no computed window"
    if r.mp_formula == "Al":
        return ("framework agrees - kinetically stable beyond the thermodynamic bound",
                "Al is the reservoir metal (phi_ox = 0 by definition); the 1.0 V onset "
                "reflects its native oxide")
    t = r.potential_type
    if t == "alloying_observed":
        ok = r.phi_red > 0
        return ("framework agrees" if ok else "framework disagrees",
                f"phi_red = {r.phi_red:.2f} V ({'>' if ok else '<='} 0: "
                f"{'predicts' if ok else 'does not predict'} alloying during plating; "
                f"products {r.red_products})")
    if t not in ("onset", "corrodes_at", "stable_up_to"):
        return "insufficient evidence", "qualitative observation, no numeric potential"
    e = float(r.exp_potential_V_vs_Al)
    if t in ("onset", "corrodes_at") and e < r.phi_ox - AGREE_TOL:
        return ("framework disagrees",
                f"observed corrosion at {e:.2f} V, below the bound {r.phi_ox:.2f} V - "
                "a reaction is missing from the model or the reference differs")
    if e > r.phi_ox + KINETIC_MARGIN:
        return ("framework agrees - kinetically stable beyond the thermodynamic bound",
                f"observed {t} {e:.2f} V vs bound {r.phi_ox:.2f} V: stability the bulk "
                "descriptor cannot explain (passivation/kinetics)")
    return "framework agrees", f"observed {t} {e:.2f} V vs bound {r.phi_ox:.2f} V"


def main():
    lit = pd.read_csv(LIT)
    # literature onsets measure chlorination/oxidation of the substrate: phi_chlor
    win = pd.read_csv(WIN)[["formula", "phi_chlor", "ox_products", "phi_red", "red_products",
                            "r0", "kind0"]].rename(columns={"phi_chlor": "phi_ox"})
    df = lit.merge(win, left_on="mp_formula", right_on="formula", how="left").drop(columns="formula")
    df[["verdict", "explanation"]] = [verdict(r) for r in df.itertuples()]
    df.to_csv(OUT, index=False)

    show = df[["substrate", "electrode_side", "potential_type", "exp_potential_V_vs_Al",
               "phi_ox", "phi_red", "verdict"]]
    print(show.to_string(index=False))
    print("\n" + df["verdict"].value_counts().to_string())

    num = df[df["potential_type"].isin(["onset", "corrodes_at"])
             & df["verification"].isin(SCORED) & df["phi_ox"].notna()
             & (df["mp_formula"] != "Al")].drop_duplicates("mp_formula")
    if len(num) >= 3:
        rho, _ = spearmanr(num["exp_potential_V_vs_Al"].astype(float), num["phi_ox"])
        print(f"\nSpearman rank correlation, measured onset vs phi_ox: rho = {rho:.2f} "
              f"(n = {len(num)}: {', '.join(num['substrate'])}). With n this small this "
              "shows consistency or inconsistency only, not significance.")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
