"""
Shortlist of candidates for future validation, in report form.

Reads results/screening_candidates.csv (A computed + C inferred) and
data/literature_benchmark.csv (B literature). Writes results/shortlist.csv.

Selection (all pre-registered in docs/descriptors.md):
  * tiers come from the Pareto ranking (layers 1-2 high, 3-4 medium);
  * the table applies the lowest pre-registered anodic scenario, phi_ox >= 0.5 V.
    Without it the first Pareto layer includes Al-rich aluminides whose only
    merit is r0 = 0 while Al is extracted from them just above 0 V (see report);
  * rejected examples are listed with the filter that removed them.

Run: python adib_shortlist.py   (after adib_screening.py and adib_validate.py)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CAND = ROOT / "results" / "screening_candidates.csv"
LIT = ROOT / "data" / "literature_benchmark.csv"
OUT = ROOT / "results" / "shortlist.csv"
SCENARIO_V = 0.5


def lit_status(formula, lit):
    rows = lit[lit["mp_formula"] == formula]
    if rows.empty:
        return "no chloroaluminate experiment found (B: none)"
    bits = []
    for _, r in rows.iterrows():
        tag = {"verbatim-full-text": "verified", "verbatim-abstract": "verified (abstract)",
               "search-summary": "unverified summary", "none": "unsourced label"}[r["verification"]]
        e = (f" {r['exp_potential_V_vs_Al']} V" if pd.notna(r["exp_potential_V_vs_Al"])
             and str(r["exp_potential_V_vs_Al"]) not in ("", "nan") else "")
        bits.append(f"{r['corrosion_stability']}{e} [{r['electrode_side']}; {tag}; "
                    f"{r['citation'].split(',')[0]} {r['doi']}]")
    return " | ".join(bits)


def row(r, lit, priority):
    al = ("does not react with Al at 0 V" if r["kind0"] == "stable"
          else f"reacts with Al at 0 V -> {r['products0']} ({r['r0']:.2f} eV/atom)")
    return {
        "priority": priority,
        "material": r["formula"],
        "family": r["family"],
        "mp_id": r["material_id"],
        "stability (A)": f"E_hull {1000 * r['energy_above_hull']:.0f} meV/atom; "
                         + ("ICSD-known" if not r["theoretical"] else "hypothetical in MP"),
        "conductivity proxy (A)": "metallic (PBE)" if r["is_metal"] else
                                  f"band gap {r['band_gap']:.2f} eV (PBE)",
        "anodic limit phi_ox (A)": f"{r['phi_ox']:.2f} V ({r['ox_mechanism']}: {r['ox_products']}); "
                                   f"alt. route {r['phi_ox_alt']:.2f} V",
        "Al interaction at plating (A)": al,
        "Al nucleation affinity": r["al_nucleation_affinity"],
        "reason for selection (C)": r["reason"] if isinstance(r["reason"], str) else "",
        "confidence (C)": r["confidence"],
        "validation status (B)": lit_status(r["formula"], lit),
    }


def main():
    df = pd.read_csv(CAND)
    lit = pd.read_csv(LIT)
    ok = df["status"] == "candidate"
    rows = []
    for tier in ("high priority", "medium priority"):
        sel = df[ok & (df["tier"] == tier) & (df["phi_ox"] >= SCENARIO_V)]
        rows += [row(r, lit, f"{tier} candidate for future validation") for _, r in sel.iterrows()]

    # rejected examples, each with the filter responsible
    rej = []
    for status, n in (("rejected: F2 not metallic (PBE)", 10),
                      ("rejected: F1 E_hull above metastability window", 6)):
        for _, r in df[df["status"] == status].sort_values("energy_above_hull").head(n).iterrows():
            rej.append({"priority": "rejected", "material": r["formula"], "family": r["family"],
                        "mp_id": r["material_id"], "reason for selection (C)": status,
                        "stability (A)": f"E_hull {1000 * r['energy_above_hull']:.0f} meV/atom",
                        "conductivity proxy (A)": f"band gap {r['band_gap']:.2f} eV (PBE)",
                        "validation status (B)": lit_status(r["formula"], lit)})
    low = df[ok & (df["phi_ox"] < SCENARIO_V)]
    for f in ["Al6Fe", "Al10V", "Ti", "Mg", "Cr2N", "TiN", "Mo"]:
        s = low[low["formula"] == f]
        if s.empty:
            s = df[(df["formula"] == f) & ok & (df["phi_ox"] < SCENARIO_V)]
        if s.empty:
            continue
        r = s.iloc[0]
        rej.append({"priority": "rejected", "material": f, "family": r["family"],
                    "mp_id": r["material_id"],
                    "reason for selection (C)": f"anodic limit {r['phi_ox']:.2f} V < {SCENARIO_V} V "
                                                f"scenario ({r['ox_mechanism']}: {r['ox_products']})",
                    "anodic limit phi_ox (A)": f"{r['phi_ox']:.2f} V",
                    "validation status (B)": lit_status(f, lit)})
    out = pd.DataFrame(rows + rej)
    out.to_csv(OUT, index=False)
    print(out["priority"].value_counts().to_string())
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
