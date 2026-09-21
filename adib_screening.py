"""
ADIB anode-free substrate screening - physics-informed screening pipeline.

Stages (all thresholds pre-registered in docs/descriptors.md):
    0  candidate space with a stated rationale per family       (adib_data.py)
    1  F1 thermodynamic stability: E_hull <= 90th pct metastability (Sun 2016)
    2  F2 metallic at PBE level (conduction proxy)
    3  electrolyte stability window from a grand-potential phase diagram open
       to Al and Cl (adib_windows.py): phi_ox, phi_red, r0
    4  surface descriptors where MP has them (elements only)
    5  Pareto ranking, no weights; sensitivity over the anodic cut-off

Every column is tagged by knowledge type in results/screening_columns.csv:
A = computed (MP or derived from MP energies), C = inferred (screening/ML).
Literature (B) lives in data/literature_benchmark.csv and is joined only in
adib_validate.py.

Run:  python adib_screening.py            (needs data/ from adib_data.py)
      python adib_screening.py --workers 4
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import adib_data as ad

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WINDOWS_CSV = RESULTS / "stability_windows.csv"
CURVES_NPZ = RESULTS / "stability_curves.npz"

EHULL_MAX = {"default": 0.067, "nitride": 0.195}     # eV/atom, Sun et al. 2016
ANODIC_CUTOFFS = [0.5, 1.0, 1.5, 2.0]                 # V vs Al, sensitivity sweep
NUCLEATION_NOTE = "Not available in the current dataset; proposed for Phase 2/DFT"

COLUMN_TYPES = {
    "formula": "A", "family": "-", "chemsys": "A", "material_id": "A",
    "energy_above_hull": "A", "formation_energy_per_atom": "A", "band_gap": "A",
    "is_metal": "A", "theoretical": "A", "n_icsd": "A", "spacegroup": "A",
    "density": "A", "work_function": "A", "surface_energy": "A",
    "phi_ox": "A", "phi_ox_censor": "A", "ox_mechanism": "A", "ox_products": "A",
    "phi_chlor": "A",
    "phi_red": "A", "phi_red_censor": "A", "red_products": "A",
    "r0": "A", "kind0": "A", "products0": "A",
    "al_nucleation_affinity": "-",
    "passes_F1": "C", "passes_F2": "C", "status": "C", "pareto_layer": "C",
    "tier": "C", "max_cutoff_passed": "C", "reason": "C",
    "phi_ox_alt": "A", "phi_red_alt": "A", "r0_alt": "A", "phi_ox_method_spread": "A",
    "pareto_layer_alt": "C", "tier_alt": "C", "confidence": "C",
}


# ---------------------------------------------------------------------------
def _scan_system(args):
    """Primary route (MP entries, client-side mixing) + alternative route
    (server thermo documents); the alternative contributes phi_*_alt only."""
    key, formulas = args
    warnings.filterwarnings("ignore")
    import adib_windows as aw
    res = aw.scan(ad.load_entries(key), formulas)
    if ad.thermo_path(key).exists():
        alt = aw.scan(ad.load_thermo_entries(key), formulas)
        for f, r in res.items():
            a = alt.get(f, {})
            r["phi_ox_alt"] = a.get("phi_ox")
            r["phi_red_alt"] = a.get("phi_red")
            r["r0_alt"] = a.get("r0")
    return key, res


def compute_windows(summary, workers=4, refresh=False):
    """phi_ox / phi_red / r0 for every composition in the screening space."""
    if WINDOWS_CSV.exists() and CURVES_NPZ.exists() and not refresh:
        return pd.read_csv(WINDOWS_CSV)
    jobs = {}
    for chemsys, grp in summary.groupby("chemsys"):
        key = ad.entry_key(chemsys)
        jobs.setdefault(key, set()).update(grp["formula"])
    missing = [k for k in jobs if not ad.entry_path(k).exists()]
    if missing:
        raise SystemExit(f"Missing cached phase-diagram entries: {missing}. "
                         "Run python adib_data.py first.")
    rows, curves = [], {}
    with ProcessPoolExecutor(workers) as pool:
        for key, res in pool.map(_scan_system, [(k, sorted(v)) for k, v in jobs.items()]):
            print(f"  {key:<14} {len(res)} compositions", flush=True)
            for formula, r in res.items():
                curve = r.pop("curve", None)
                if curve is not None:
                    curves[f"{key}|{formula}"] = curve
                rows.append({"formula": formula, "pd_key": key, **r})
    win = pd.DataFrame(rows).drop_duplicates("formula")
    RESULTS.mkdir(exist_ok=True)
    win.to_csv(WINDOWS_CSV, index=False)
    import adib_windows as aw
    np.savez_compressed(CURVES_NPZ, phi=aw.PHI, **curves)
    return win


# ---------------------------------------------------------------------------
def representative_entries(summary):
    """One row per composition: the lowest-E_hull METALLIC polymorph if any.

    F1/F2 are then judged on that entry. A composition with no metallic
    polymorph is represented by its ground state and fails F2.
    """
    s = summary.sort_values("energy_above_hull")
    metal = s[s["is_metal"] == True].drop_duplicates("formula")          # noqa: E712
    rest = s[~s["formula"].isin(metal["formula"])].drop_duplicates("formula")
    rep = pd.concat([metal, rest], ignore_index=True)
    rep["n_polymorphs"] = rep["formula"].map(summary["formula"].value_counts())
    return rep


def pareto_layers(obj):
    """Non-dominated sorting. obj: (n, k) array, every column to MAXIMISE."""
    n = len(obj)
    layer = np.zeros(n, dtype=int)
    remaining = np.arange(n)
    k = 1
    while len(remaining):
        sub = obj[remaining]
        dominated = np.zeros(len(sub), dtype=bool)
        for i in range(len(sub)):
            better_eq = np.all(sub >= sub[i], axis=1)
            strictly = np.any(sub > sub[i], axis=1)
            dominated[i] = np.any(better_eq & strictly)
        front = remaining[~dominated]
        layer[front] = k
        remaining = remaining[dominated]
        k += 1
    return layer


def reason_text(r):
    """Plain-language reason a candidate survives, from computed values only."""
    bits = [f"E_hull {1000 * r.energy_above_hull:.0f} meV/atom"
            + (" (ICSD-known)" if not r.theoretical else " (hypothetical in MP)"),
            "metallic (PBE)"]
    bits.append(f"no anodic reaction below {r.phi_ox:.2f} V vs Al"
                + (f" (then {r.ox_mechanism}: {r.ox_products})" if r.ox_products else ""))
    if r.kind0 == "stable":
        bits.append("stable against Al metal at 0 V")
    else:
        bits.append(f"reacts with Al at 0 V ({r.products0}; {r.r0:.2f} eV/atom)")
    return "; ".join(bits)


def screen(summary, windows):
    rep = representative_entries(summary)
    df = rep.merge(windows, on="formula", how="left")
    lim = df["family"].map(lambda f: EHULL_MAX.get(f, EHULL_MAX["default"]))
    df["passes_F1"] = df["energy_above_hull"] <= lim
    df["passes_F2"] = df["is_metal"] == True                             # noqa: E712
    df["al_nucleation_affinity"] = NUCLEATION_NOTE

    df["status"] = "candidate"
    df.loc[~df["passes_F2"], "status"] = "rejected: F2 not metallic (PBE)"
    df.loc[~df["passes_F1"], "status"] = "rejected: F1 E_hull above metastability window"
    df.loc[df["formula"] == "Al", "status"] = "reference: plated metal"

    cand = df["status"] == "candidate"
    obj = np.column_stack([df.loc[cand, "phi_ox"], df.loc[cand, "r0"],
                           -df.loc[cand, "energy_above_hull"]])
    df["pareto_layer"] = pd.NA
    df.loc[cand, "pareto_layer"] = pareto_layers(obj)
    df["tier"] = "-"
    df.loc[cand, "tier"] = np.select(
        [df.loc[cand, "pareto_layer"] <= 2, df.loc[cand, "pareto_layer"] <= 4],
        ["high priority", "medium priority"], "low priority")
    df["max_cutoff_passed"] = [
        max([c for c in ANODIC_CUTOFFS if p >= c], default=0.0) if s == "candidate" else np.nan
        for p, s in zip(df["phi_ox"], df["status"])]
    # method uncertainty: same ranking with the alternative energy route
    df["phi_ox_method_spread"] = (df["phi_ox"] - df["phi_ox_alt"]).abs()
    obj_alt = np.column_stack([df.loc[cand, "phi_ox_alt"].fillna(df.loc[cand, "phi_ox"]),
                               df.loc[cand, "r0_alt"].fillna(df.loc[cand, "r0"]),
                               -df.loc[cand, "energy_above_hull"]])
    df["pareto_layer_alt"] = pd.NA
    df.loc[cand, "pareto_layer_alt"] = pareto_layers(obj_alt)
    df["tier_alt"] = "-"
    df.loc[cand, "tier_alt"] = np.select(
        [df.loc[cand, "pareto_layer_alt"] <= 2, df.loc[cand, "pareto_layer_alt"] <= 4],
        ["high priority", "medium priority"], "low priority")
    df["confidence"] = "-"
    same = df["tier"] == df["tier_alt"]
    df.loc[cand, "confidence"] = np.select(
        [same[cand] & ~df.loc[cand, "theoretical"].astype(bool),
         same[cand]],
        ["higher: tier robust to energy route; ICSD-known",
         "medium: tier robust to energy route; hypothetical in MP"],
        "lower: tier changes with energy route")
    df["reason"] = ""
    df.loc[cand, "reason"] = [reason_text(r) for r in df[cand].itertuples()]
    order = {"high priority": 0, "medium priority": 1, "low priority": 2, "-": 3}
    return df.sort_values(["tier", "pareto_layer", "phi_ox"],
                          key=lambda c: c.map(order) if c.name == "tier" else c,
                          ascending=[True, True, False]).reset_index(drop=True)


def sensitivity(df):
    """Survivor counts over the anodic cut-off and the E_hull window."""
    rows = []
    for scale in (0.5, 1.0, 1.5):
        lim = df["family"].map(lambda f: EHULL_MAX.get(f, EHULL_MAX["default"])) * scale
        base = (df["energy_above_hull"] <= lim) & df["passes_F2"] & (df["formula"] != "Al")
        for cut in [0.0] + ANODIC_CUTOFFS:
            ok = base & (df["phi_ox"] >= cut)
            rows.append({"ehull_window_scale": scale, "anodic_cutoff_V": cut,
                         "n_candidates": int(ok.sum()),
                         "also_stable_vs_Al_at_0V": int((ok & (df["kind0"] == "stable")).sum()),
                         "families": ", ".join(f"{k}:{v}" for k, v in
                                               df[ok]["family"].value_counts().items())})
    return pd.DataFrame(rows)


def funnel(summary, df):
    cand = df[df["status"] == "candidate"]
    return pd.DataFrame([
        ("MP entries in screening space", len(summary)),
        ("unique compositions", df["formula"].nunique()),
        ("pass F1 (E_hull window)", int(df["passes_F1"].sum())),
        ("pass F1 + F2 (metallic)", int((df["passes_F1"] & df["passes_F2"]).sum()) - 1),
        ("phi_ox >= 0.5 V", int((cand["phi_ox"] >= 0.5).sum())),
        ("phi_ox >= 1.0 V", int((cand["phi_ox"] >= 1.0).sum())),
        ("phi_ox >= 1.0 V and stable vs Al at 0 V",
         int(((cand["phi_ox"] >= 1.0) & (cand["kind0"] == "stable")).sum())),
        ("high-priority tier (Pareto layers 1-2)", int((cand["tier"] == "high priority").sum())),
        ("high-priority tier and phi_ox >= 0.5 V (reported shortlist)",
         int(((cand["tier"] == "high priority") & (cand["phi_ox"] >= ANODIC_CUTOFFS[0])).sum())),
    ], columns=["stage", "count"])


def main():
    ap = argparse.ArgumentParser(description="ADIB substrate screening")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh-windows", action="store_true")
    args = ap.parse_args()

    summary = ad.fetch_summary()
    print(f"{len(summary)} MP entries, {summary['chemsys'].nunique()} chemical systems, "
          f"{summary['formula'].nunique()} compositions")
    windows = compute_windows(summary, args.workers, args.refresh_windows)
    df = screen(summary, windows)

    RESULTS.mkdir(exist_ok=True)
    cols = [c for c in COLUMN_TYPES if c in df.columns] + ["n_polymorphs", "crystal_system",
                                                          "vol_per_atom", "window"]
    df[cols].to_csv(RESULTS / "screening_candidates.csv", index=False)
    pd.DataFrame({"column": list(COLUMN_TYPES), "knowledge_type": list(COLUMN_TYPES.values())}
                 ).to_csv(RESULTS / "screening_columns.csv", index=False)
    sens = sensitivity(df)
    sens.to_csv(RESULTS / "screening_sensitivity.csv", index=False)
    fun = funnel(summary, df)
    fun.to_csv(RESULTS / "screening_funnel.csv", index=False)

    print("\nScreening funnel")
    print(fun.to_string(index=False))
    print("\nSensitivity (E_hull window at 1.0x)")
    print(sens[sens["ehull_window_scale"] == 1.0].to_string(index=False))
    show = ["formula", "family", "energy_above_hull", "theoretical", "phi_ox", "ox_products",
            "kind0", "r0", "pareto_layer", "tier"]
    print("\nHigh- and medium-priority candidates for future validation")
    print(df[df["tier"].isin(["high priority", "medium priority"])][show]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
