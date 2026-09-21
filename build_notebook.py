"""
Builds adib_workflow.ipynb from the pipeline modules (so the notebook cannot
drift from the code), then execute it with:

    jupyter nbconvert --to notebook --execute --inplace adib_workflow.ipynb

Run: python build_notebook.py
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# ADIB anode-free substrate screening: full workflow

**Question.** Which conductive materials can serve as substrates (anodic current collectors)
for stable, reversible Al deposition in chloroaluminate Al dual-ion batteries?
Al itself is always the plated species (4 Al₂Cl₇⁻ + 3 e⁻ → Al + 7 AlCl₄⁻); only the
substrate is screened.

**Framing.** Physics-informed screening with ML assistance, not black-box ML prediction of
battery performance. Status: *computational screening framework and preliminary candidate
identification*.

**Knowledge types, kept apart everywhere:**
**A computed** (Materials Project or derived from MP energies) ·
**B literature** (verified sources with DOI) · **C inferred** (ML predictions, tiers).

Thresholds and validation rules were fixed before any result was seen: `docs/descriptors.md`."""),
    md("""## 0. Settings

* `DEMO = True`: the Phase-1 ML cells use **synthetic** data (a unit test of the ML code; runs
  without an API key). All screening cells below use the **real, cached** Materials Project data
  in `data/` and need no key or network.
* `REFIT_REAL_ML = False`: the real Phase-1 benchmark (1,566 entries, 76 LOGO + 23 element
  folds, permutation importance) takes ~1.5 h on a laptop; its saved outputs in `results/` are
  loaded instead. Set `True` (with `MP_API_KEY`) to recompute."""),
    code("""import os, sys, io, contextlib, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)

DEMO = True
REFIT_REAL_ML = False"""),
    md("""## 1. Phase-1 ML infrastructure: formation energy, leakage-controlled

Formation energy is a *benchmark* target with full MP coverage; it is not the screening
criterion. For materials already in MP, the DFT value is used directly."""),
    code("""import substrate_screening as ss
if DEMO:
    print(ss.DEMO_FEATURESET_WARNING)
    df = ss.make_synthetic()
    with contextlib.redirect_stdout(io.StringIO()):
        df, labels = ss.add_magpie(df)
        sets = ss.feature_sets(df, labels, False)
    feats = list(dict.fromkeys(c for cols in sets.values() for c in cols))
    audit = ss.audit_leakage(df, feats + list(ss.LEAKY_EXCLUDED), df["group"])
    print("SYNTHETIC leakage audit - the planted E_hull leak must be caught:")
    display(audit[audit["flags"].str.contains("LEAK") | (audit["role"] == "excluded")])
    print(f"{df['group'].nunique()} synthetic groups -> "
          f"{len(ss.grouped_folds(df, df['target'], df['group']))} LOGO folds")"""),
    code("""if REFIT_REAL_ML:
    subprocess.run([sys.executable, "substrate_screening.py", "--key", os.environ["MP_API_KEY"],
                    "--out", "results/model_comparison.csv", "--audit-out", "results/leakage_audit.csv",
                    "--oof-out", "results/oof_predictions.csv", "--data-out", "results/mp_ml_dataset.csv"],
                   check=True)
audit = pd.read_csv("results/leakage_audit.csv")
print("REAL-DATA leakage audit (1,566 MP entries, 76 chemical systems):")
print(audit["flags"].str.split(",").explode().value_counts().to_string())
display(audit[audit["flags"].str.contains("LEAK|review") | (audit["role"] == "excluded")])"""),
    code("""mc_path = "results/model_comparison.csv"
if os.path.exists(mc_path):
    mc = pd.read_csv(mc_path)
    for fs in mc["feature_set"].unique():
        print(f"\\n{fs} features - pooled / system-balanced R2 (+/- group-bootstrap SE)")
        display(pd.concat([ss.results_table(mc[mc.feature_set == fs].reset_index(drop=True), w)
                           .set_index("model").add_suffix(f" [{w}]")
                           for w in ss.WEIGHTINGS], axis=1).filter(like="R2"))
    for fs in mc["feature_set"].unique():
        ss.interpret(mc[mc.feature_set == fs].reset_index(drop=True), fs)
else:
    print("results/model_comparison.csv not found - run with REFIT_REAL_ML = True")"""),
    md("""## 2. ML assistance: an unseen family and the applicability domain

Transition-metal aluminides are **not** in the Phase-1 training set, so their DFT formation
energies test the composition-only model on genuinely unseen chemistry."""),
    code("""dom = pd.read_csv("results/ml_domain_summary.csv")
display(dom.round(3))
d = pd.read_csv("results/ml_domain.csv"); a = d[d.family == "aluminide"]
from scipy.stats import spearmanr
for col in ("knn_dist", "rf_std", "model_spread"):
    print(f"Spearman({col}, |RF error|) on aluminides = {spearmanr(a[col], a['abs_err_RandomForest'])[0]:.2f}")"""),
    md("""## 3. Candidate space, retrieval, filters

Six families, each with a stated reason (`adib_data.SCREENING_FAMILIES`).
F1: E_hull ≤ 90th-percentile metastability of known crystals (67 meV/atom; 195 for nitrides;
Sun et al. 2016). F2: metallic at the PBE level."""),
    code("""import adib_data as ad, adib_screening as asc
for fam, f in ad.SCREENING_FAMILIES.items():
    print(f"{fam:<10} {len(f['systems']):>2}  {f['rationale']}")
summary = ad.fetch_summary()          # cached: data/screening_summary.csv
print(f"\\n{len(summary)} MP entries, {summary.chemsys.nunique()} systems, {summary.formula.nunique()} compositions")"""),
    md("""## 4. Electrolyte stability windows

Grand-potential phase diagrams open to Al and Cl, with μ_Al = μ°_Al − 3eφ and
μ_Cl = [g(AlCl₃) − μ_Al]/3. **φ_ox** = anodic limit (chlorination or Al extraction),
**φ_red** = Al-alloying limit, **r0** = reaction driving force with Al at 0 V. φ_ox is a
thermodynamic lower bound. Windows are loaded from `results/stability_windows.csv`; delete it to
recompute (~20 min). One example is recomputed live:"""),
    code("""import adib_windows as aw
live = aw.scan(ad.load_entries("Al-Cl-W"), ["W"])["W"]
print({k: live[k] for k in ("phi_ox", "ox_mechanism", "ox_products", "phi_red", "kind0", "products0")})
windows = asc.compute_windows(summary)
cands = asc.screen(summary, windows)
display(asc.funnel(summary, cands))"""),
    code("""sens = asc.sensitivity(cands)
display(sens.pivot(index="anodic_cutoff_V", columns="ehull_window_scale", values="n_candidates"))"""),
    md("""## 5. Known-material validation (rules fixed beforehand)

Literature table: `data/literature_benchmark.csv`, every row with electrode side, conditions,
DOI and verification status. Unverified rows are never scored."""),
    code("""lit = pd.read_csv("data/literature_benchmark.csv")
print(lit["verification"].value_counts().to_string())
subprocess.run([sys.executable, "adib_validate.py"], check=True)
v = pd.read_csv("results/validation.csv")
display(v[["substrate", "electrode_side", "potential_type", "exp_potential_V_vs_Al", "phi_ox", "verdict"]])"""),
    md("""## 6. Shortlist: high-priority candidates for future validation

Pareto tiers (no weights), shown for the lowest registered anodic scenario φ_ox ≥ 0.5 V.
Confidence records whether the tier survives the alternative MP energy route."""),
    code("""subprocess.run([sys.executable, "adib_shortlist.py"], check=True)
sl = pd.read_csv("results/shortlist.csv")
display(sl[["priority", "material", "family", "stability (A)", "anodic limit phi_ox (A)",
            "Al interaction at plating (A)", "confidence (C)"]])"""),
    md("## 7. Figures"),
    code("""for f in sorted(os.listdir("figures")):
    if f.endswith(".png"):
        print(f); display(Image(filename=os.path.join("figures", f), width=820))"""),
    md("""## 8. Limitations and next steps

Bulk thermodynamics only (φ_ox is a lower bound; TiN/Cr₂N/W/Mo work through kinetics); no surface
or nucleation physics yet ("Not available in the current dataset; proposed for Phase 2/DFT");
idealised electrolyte; DFT mixing error (route spread up to 0.42 V; Cr₂N wrongly rejected by F1);
sparse, mostly positive-electrode literature; ML uncertainty proxies do not track error.
See `docs/report.md` for the full list and the November plan."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "adib_workflow.ipynb")
print(f"adib_workflow.ipynb written ({len(cells)} cells)")
