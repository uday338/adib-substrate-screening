"""
The core of the project in one short script (~60 lines of actual work).

It does exactly three things, for a handful of materials:

    1. F1 + F2   keep only materials that are stable and metallic
    2. window    compute the potential above which the melt attacks them,
                 and whether they react with the aluminium plated on them
    3. compare   put those numbers next to the measured values from Wang et al. 2018

Everything reads from the cached Materials Project data in data/, so it needs
no API key and no network. The full pipeline (adib_*.py) does the same thing
for 330 compositions, with sensitivity and uncertainty checks on top.

Run:  python simple_screen.py
"""

import warnings

import pandas as pd

import adib_data as ad
import adib_windows as aw

warnings.filterwarnings("ignore")

# Materials to look at: three literature references and three candidates.
MATERIALS = {
    "Ti":   "Al-Cl-Ti",        # measured onset 1.1 V (Wang 2018)
    "TiN":  "Al-Cl-N-Ti",      # measured stable to >= 2.5 V
    "W":    "Al-Cl-W",         # used as a collector in practice
    "TiB2": "Al-B-Cl-Ti",      # our candidate
    "VB2":  "Al-B-Cl-V",       # our candidate
    "Mg":   "Al-Cl-Mg",        # expected to fail
}
MEASURED = {"Ti": "onset 1.1 V", "TiN": "stable to >= 2.5 V", "W": "stable (no number)"}

# F1: how far above the convex hull a material may sit and still be considered
# (Sun et al. 2016: 90th percentile of the metastability of real crystals).
EHULL_MAX = {"nitride": 0.195, "default": 0.067}


def main():
    summary = ad.fetch_summary()          # cached table of Materials Project entries
    rows = []

    for formula, chemsys in MATERIALS.items():
        # --- pick the entry for this material -------------------------------
        entries = summary[summary["formula"] == formula].sort_values("energy_above_hull")
        entry = entries.iloc[0]
        family = entry["family"]
        limit = EHULL_MAX.get(family, EHULL_MAX["default"])

        # --- the two hard filters -------------------------------------------
        passes_f1 = entry["energy_above_hull"] <= limit      # stable enough?
        passes_f2 = bool(entry["is_metal"])                  # conducts electrons?

        # --- the physics: what the melt does to it at each potential ---------
        # phi_ox  : below this, the melt cannot attack it (a lower bound)
        # r0      : energy released if it reacts with plated Al at 0 V (0 = inert)
        result = aw.scan(ad.load_entries(chemsys), [formula])[formula]

        rows.append({
            "material": formula,
            "family": family,
            "E_hull (meV/atom)": round(1000 * entry["energy_above_hull"]),
            "stable (F1)": passes_f1,
            "metallic (F2)": passes_f2,
            "phi_ox (V)": result["phi_ox"],
            "attacked by": result["ox_products"],
            "r0 (eV/atom)": round(result["r0"], 2),
            "reacts with Al at 0 V": result["kind0"] != "stable",
            "measured": MEASURED.get(formula, "not tested in the literature"),
        })

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nphi_ox is a LOWER BOUND: corrosion below it is impossible, but a")
    print("material can survive above it if a passive film protects it, which is")
    print("why TiN measures far better than its computed bound.")


if __name__ == "__main__":
    main()
