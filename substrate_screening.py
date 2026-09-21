"""
ADIB anode-free substrate screening - Phase 1 prototype
========================================================

Scope
-----
True ADIB: Al metal plates/strips at the negative electrode, AlCl4- intercalates
a graphite cathode, chloroaluminate electrolyte. Aluminium cannot be "replaced"
as the plated species - Al2Cl7- + e- -> Al IS aluminium deposition. What CAN be
changed is the substrate the aluminium plates onto (the anodic current
collector). This script screens those substrate materials.

What this Phase 1 script does
-----------------------------
Benchmarks three classical models - Ridge, Random Forest, Gradient Boosting
(XGBoost when installed) - on a property with FULL Materials Project coverage
(formation energy per atom). The point is to prove the pipeline and pick a model
BEFORE applying it to the sparse, project-specific targets in Phase 2.

Every model is cross-validated twice: 5-fold shuffled KFold (random) and
leave-one-group-out by chemical system (grouped - one family held out per
fold). Grouped resolution is set by the NUMBER OF CHEMICAL SYSTEMS, not the
number of rows, which is why CANDIDATE_FAMILIES holds 60+ systems. The gap
between the two splits is the honest measure of whether the model generalises
to unseen chemistries rather than memorising families.

A third, harder split holds out every system containing one element
(leave-one-element-out, "element"). LOGO holds out N-Ti while training on Ti
and C-Ti, so it tests generalisation to an unseen chemical SYSTEM, not an
unseen ELEMENT; the element split is the stricter test. A binary appears in
two element folds (N-Ti in the N and Ti folds); its two predictions are
averaged. --no-loeo skips it.

Every score is reported two ways, each with a standard error from a bootstrap
that resamples whole chemical systems:
    pooled    metrics over all out-of-fold predictions. Systems count in
              proportion to their rows, so large systems dominate (oxides are
              ~44% of real rows, V-O alone ~13%).
    balanced  each chemical system counts equally. MAE and RMSE are computed
              per system and averaged. R2 is the system-weighted R2 (each row
              weighted 1 / rows in its system): a plain mean of per-system R2
              is unusable, because a system whose polymorphs span 0.01 eV/atom
              (Ag) gives a hugely negative R2 for any ordinary error.
If pooled and balanced disagree, that is reported as a finding.
Per-fold R2 is not used for the same reason. Every model, split and
weighting shares the same bootstrap draws, so differences get paired SEs.

Feature sets
------------
    full              Magpie (132 ElementProperty stats) + n_elements
                      + structure/DFT descriptors (density, vol_per_atom,
                      band_gap, nsites)
    composition-only  Magpie + n_elements

Density, band gap and volume come out of the same DFT relaxation that yields
formation energy, and nsites needs a known structure. For a genuinely untested
material none of them exist without running DFT first - at which point the
formation energy is already known. So the full set is circular for de novo
prediction; composition-only is the set that matters for Phase 2. Both are run
side by side so the cost of dropping the DFT features is visible.

Target leakage
--------------
Before any model is fitted, every candidate column is audited against the
target (see audit_leakage). energy_above_hull is excluded from all feature
sets: it is computed from formation energies, and for elemental entries it is
numerically identical to the target. It is still retrieved, because it is the
right quantity for a stability FILTER in Phase 2 - just never an input when
formation energy is what is being predicted.

Unphysical entries
------------------
Materials Project holds a few structures far above the hull (elemental Ta at
+5.1 eV/atom, Ta11O2 at +9.0 eV/atom). No such phase could serve as a
substrate, and under a squared-error metric a handful of them dominate R2 and
RMSE. In Materials Project mode, rows with energy_above_hull > MAX_E_HULL
(1.0 eV/atom, fixed before any model was scored) are dropped and every dropped
row is listed. energy_above_hull selects rows only; it is never a feature.
--max-ehull none keeps everything.

For elemental systems E_hull == E_form exactly, so this filter also truncates
the target range of the elemental rows (to E_form <= 1.0 eV/atom). R2 depends
on the target's spread, so R2 from the filtered data is not comparable with an
unfiltered study.

Feature importance
------------------
Permutation importance (sklearn.inspection.permutation_importance): the mean
drop in held-out R2 when one feature is shuffled, on 5-fold GroupKFold by
chemical system. Not on LOGO folds: a one-system fold's R2 is noise, and 76
folds x ~130 features x 3 models is too slow for a laptop. Collinear Magpie
stats share credit, so a feature with a near-twin can score near zero even
when the information it carries matters.

Modes
-----
    python substrate_screening.py --demo
        Runs on SYNTHETIC data with a known ground-truth function. This proves
        only that the code works. It is NOT science and the numbers mean
        nothing about real materials. Use it to check your install.

    python substrate_screening.py --key $MP_API_KEY
        Pulls real candidate substrates from Materials Project and runs the
        same comparison.

    --composition-only
        Run only the composition-only feature set (skips the side-by-side).

    --grouped gss
        Fallback if LOGO is too slow on real data: 20 repeated
        GroupShuffleSplits (20% of systems held out each) instead of LOGO.

    --no-loeo
        Skip the leave-one-element-out split.

    --folds-only
        Print the chemical-system count and grouped fold structure, then stop
        before the leakage audit and model comparison.

Install
-------
    pip install pandas scikit-learn matplotlib matminer pymatgen
    pip install xgboost          # optional, falls back to sklearn if absent
    pip install mp-api           # only needed for --key mode
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     LeaveOneGroupOut)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5                 # random split, and Ridge's inner grouped alpha search
N_GSS_SPLITS = 20           # --grouped gss fallback only
GSS_TEST_SIZE = 0.2
N_BOOT = 1000               # group-bootstrap draws for standard errors
FLAT_TARGET_STD = 0.02      # eV/atom; a held-out system below this has no
                            # usable per-fold R2 (fold-structure report only)
TOP_IMPORTANCES = 15
PI_REPEATS = 5              # shuffles per feature in permutation importance
MAX_E_HULL = 1.0            # eV/atom; MP rows above this are dropped as unphysical

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Literature ground truth. These outcomes are reported in the ADIB literature
# and are the validation set for Phase 3 - if the model ranks these correctly,
# its predictions on untested materials become credible.
# ---------------------------------------------------------------------------
KNOWN_SUBSTRATES = [
    ("W",              "stable",   "Established oxidatively stable collector in ADIBs"),
    ("Mo",             "stable",   "Established oxidatively stable collector in ADIBs"),
    ("glassy carbon",  "stable",   "Established oxidatively stable collector in ADIBs"),
    ("TiN",            "stable",   "Oxidative stability on par with W and Mo in AlCl3:EMIMCl"),
    ("graphite paper", "works",    "Cycles stably as anodic current collector, anode-free cell"),
    ("Ni",             "mixed",    "Cycles, but performance differs by corrosion resistance"),
    ("Cu",             "mixed",    "Cycles, but performance differs by corrosion resistance"),
    ("Ag",             "mixed",    "Cycles, but performance differs by corrosion resistance"),
    ("Mg",             "mixed",    "Cycles, but performance differs by corrosion resistance"),
    ("stainless steel","corrodes", "Corrodes in chloroaluminate melts"),
    ("Al",             "corrodes", "Oxidised in Al anolytes at high cell voltages"),
    ("Fe",             "corrodes", "Oxidised in Al anolytes at high cell voltages"),
]


# ---------------------------------------------------------------------------
# Candidate chemistries to screen. Widen these lists to widen the search.
# ---------------------------------------------------------------------------
# Each entry is one chemical system = one group in grouped CV. The grouped
# score can only be as well resolved as the number of systems here allows.
# The elemental list covers every KNOWN_SUBSTRATES entry that maps to a single
# chemical system (C for glassy carbon / graphite paper; stainless steel does
# not map to one).
CANDIDATE_FAMILIES = {
    "nitride":   ["Ti-N", "Zr-N", "Hf-N", "V-N", "Nb-N", "Ta-N", "Cr-N",
                  "Mo-N", "W-N"],
    "carbide":   ["Ti-C", "Zr-C", "Hf-C", "V-C", "Nb-C", "Ta-C", "Cr-C",
                  "Mo-C", "W-C", "Si-C"],
    "boride":    ["Ti-B", "Zr-B", "Hf-B", "V-B", "Nb-B", "Ta-B", "Cr-B",
                  "Mo-B", "W-B"],
    "oxide":     ["Ti-O", "Zr-O", "Hf-O", "V-O", "Nb-O", "Ta-O", "Cr-O",
                  "Mo-O", "W-O"],
    "silicide":  ["Ti-Si", "Zr-Si", "V-Si", "Nb-Si", "Ta-Si", "Cr-Si",
                  "Mo-Si", "W-Si", "Ni-Si"],
    "phosphide": ["Ti-P", "Mo-P", "W-P", "Fe-P", "Co-P", "Ni-P", "Cu-P"],
    "sulfide":   ["Ti-S", "Mo-S", "W-S", "Co-S", "Ni-S", "Cu-S"],
    "element":   ["W", "Mo", "Ni", "Cu", "Ag", "Ti", "Ta", "Nb", "Cr", "Zr",
                  "Hf", "V", "Fe", "Co", "Mg", "Al", "C"],
}
CANDIDATE_SYSTEMS = [s for systems in CANDIDATE_FAMILIES.values() for s in systems]


def canonical(chemsys):
    """Materials Project chemsys form: elements sorted, dash-joined."""
    return "-".join(sorted(chemsys.split("-")))


FAMILY_OF = {canonical(s): fam
             for fam, systems in CANDIDATE_FAMILIES.items() for s in systems}


def check_candidates():
    """Halt on a bad element symbol or a system listed twice."""
    from pymatgen.core import Element
    seen = {}
    for fam, systems in CANDIDATE_FAMILIES.items():
        for s in systems:
            for el in s.split("-"):
                try:
                    Element(el)
                except ValueError:
                    sys.exit(f"CANDIDATE_FAMILIES[{fam!r}]: bad element {el!r} in {s!r}")
            key = canonical(s)
            if key in seen:
                sys.exit(f"Chemical system {s!r} listed twice ({seen[key]}, {fam})")
            seen[key] = fam

# Need a relaxed structure (and, except nsites, a DFT calculation) to compute.
# Dropped in composition-only mode. Hand-rolled mean_Z/max_Z/min_Z are gone:
# Magpie's mean/minimum/maximum Number are the same quantities.
STRUCTURE_DFT_FEATURES = ["density", "vol_per_atom", "band_gap", "nsites"]
COMPOSITION_EXTRA = ["n_elements"]

# Columns that are retrieved but must never be model inputs while formation
# energy is the target. Checked against every feature set at startup.
LEAKY_EXCLUDED = {
    "energy_above_hull": (
        "E_hull = E_form - E_hull_convex(composition); computed from this "
        "entry's formation energy and its competitors'. Elemental hulls sit "
        "at 0, so E_hull == E_form exactly for every elemental entry."
    ),
}

# Audit thresholds.
EXACT_MATCH_FRAC = 0.01     # share of rows where column == target
SINGLE_FEATURE_R2 = 0.80    # grouped-split R2 from that column alone (warning)
GROUP_CONST_FRAC = 0.95     # share of multi-member groups where column is constant


# ---------------------------------------------------------------------------
# Synthetic chemistry: real element symbols so Magpie can featurise them.
_RATIOS = [(1, 1), (2, 1), (1, 2), (3, 1), (3, 4), (5, 3), (4, 3)]


def make_synthetic(n=900, systems=CANDIDATE_SYSTEMS):
    """SYNTHETIC DATA - pipeline smoke test only, not real materials.

    The groups are exactly CANDIDATE_SYSTEMS, so the demo has the same number
    of chemical systems - and the same LOGO fold count - as a real run. Rows
    per system are uniform-random here and will look nothing like Materials
    Project, where some systems have one entry and others hundreds.

    Ground truth is deliberately nonlinear so the escalation ladder is visible:
    Ridge should underperform the tree models.

    Each group is a chemical system with its own offset added to the target,
    the way real families share a bias. A model can learn that offset from
    family-identifying features when the family is in training, but not for an
    unseen family - so the grouped split must score LOWER than the random one.
    If the random-to-grouped gap is not clearly positive here, the gap
    diagnostic is broken.

    energy_above_hull is generated FROM the target (distance above the lowest
    target in its group), mirroring how the real quantity is derived, and
    elemental systems are referenced to their ground state. Together that makes
    E_hull == target on every elemental row, exactly as in Materials Project,
    so the leakage audit has something real to catch.

    Only density, vol_per_atom and band_gap carry signal beyond the family
    offset, so composition-only scores here are poor by construction.
    """
    from pymatgen.core import Element

    rng = np.random.default_rng(SEED)
    chosen = [tuple(s.split("-")) for s in systems]
    n_groups = len(chosen)
    group_offset = rng.normal(0, 0.4, n_groups)

    gidx = rng.integers(0, n_groups, n)
    formulas, mean_z, n_el = [], [], []
    for g in gidx:
        sys_ = chosen[g]
        if len(sys_) == 1:
            amounts = {sys_[0]: 1}
        else:
            p, q = _RATIOS[rng.integers(len(_RATIOS))]
            amounts = {sys_[0]: p, sys_[1]: q}
        total = sum(amounts.values())
        formulas.append("".join(f"{el}{k if k > 1 else ''}" for el, k in amounts.items()))
        mean_z.append(sum(Element(el).Z * k for el, k in amounts.items()) / total)
        n_el.append(len(amounts))

    df = pd.DataFrame({
        "formula":      formulas,
        "group":        ["-".join(sorted(chosen[g])) for g in gidx],
        "n_elements":   n_el,
        "density":      rng.uniform(2, 18, n),
        "vol_per_atom": rng.uniform(8, 30, n),
        "nsites":       rng.integers(1, 40, n),
        "band_gap":     rng.gamma(1.2, 0.8, n),
    })
    mean_z = np.asarray(mean_z)

    y = (
        -0.35 * np.sqrt(df["density"])
        + 0.02 * df["vol_per_atom"]
        - 0.6 * np.tanh(df["band_gap"])
        - 0.004 * mean_z
        + 0.15 * np.sin(df["density"] / 2)
        + group_offset[gidx]
        + rng.normal(0, 0.12, n)
    )
    # Elements are referenced to their own ground state, so an elemental
    # system's lowest entry sits at exactly 0 - as in Materials Project.
    elemental = df["n_elements"] == 1
    y = y - np.where(elemental, y.groupby(df["group"]).transform("min"), 0.0)
    df["target"] = y
    df["energy_above_hull"] = df["target"] - df.groupby("group")["target"].transform("min")
    return df


def fetch_mp(api_key):
    """Pull real candidate substrates from Materials Project."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("mp-api not installed. Run: pip install mp-api pymatgen")

    fields = ["material_id", "formula_pretty", "chemsys", "nsites", "density",
              "volume", "band_gap", "formation_energy_per_atom",
              "energy_above_hull", "elements", "theoretical"]

    rows = []
    with MPRester(api_key) as mpr:
        for chemsys in CANDIDATE_SYSTEMS:
            try:
                docs = mpr.materials.summary.search(chemsys=chemsys, fields=fields)
            except Exception as exc:
                print(f"  ! {chemsys:<8} failed: {exc}", file=sys.stderr)
                continue
            print(f"  {chemsys:<8} {len(docs):>5} entries")
            for d in docs:
                if d.formation_energy_per_atom is None:
                    continue                       # never impute the target
                rows.append({
                    "material_id": str(d.material_id),
                    "formula": d.formula_pretty,
                    "group": d.chemsys,            # group = chemical system
                    "density": d.density,
                    "vol_per_atom": d.volume / d.nsites if d.nsites else None,
                    "nsites": d.nsites,
                    "band_gap": d.band_gap,
                    "energy_above_hull": d.energy_above_hull,   # audit/filter only
                    "n_elements": len(d.elements),
                    "theoretical": d.theoretical,
                    "target": d.formation_energy_per_atom,
                })

    if not rows:
        sys.exit("No results. Check the API key and network.")

    df = pd.DataFrame(rows)
    n_elem_rows = int((df["n_elements"] == 1).sum())
    print(f"\n  {len(df)} entries, {n_elem_rows} elemental "
          f"({100 * n_elem_rows / len(df):.1f}% of rows)")
    return df


def drop_unphysical(df, max_e_hull):
    """Drop rows more than max_e_hull eV/atom above the hull; list each one.

    Selection only - energy_above_hull stays out of every feature set. Rows
    with no energy_above_hull are kept and counted, never filled in.
    """
    if max_e_hull is None:
        print("  --max-ehull none: keeping all rows")
        return df
    missing = int(df["energy_above_hull"].isna().sum())
    bad = df["energy_above_hull"] > max_e_hull
    print(f"\n  dropped {int(bad.sum())} of {len(df)} rows with energy_above_hull "
          f"> {max_e_hull} eV/atom")
    if bad.any():
        cols = ["group", "material_id", "formula", "target", "energy_above_hull"]
        print(df.loc[bad, cols].sort_values(["group", "energy_above_hull"])
              .round(3).to_string(index=False))
    if missing:
        print(f"  {missing} rows have no energy_above_hull; kept")
    return df[~bad].reset_index(drop=True)


def add_magpie(df):
    """Append the 132 Magpie ElementProperty features. Returns (df, labels)."""
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except ImportError:
        sys.exit("matminer not installed. Run: pip install matminer pymatgen")

    ep = ElementProperty.from_preset("magpie", impute_nan=True)
    labels = ep.feature_labels()
    uniq = df["formula"].unique()                  # polymorphs share a formula
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = pd.DataFrame([ep.featurize(Composition(f)) for f in uniq],
                             index=uniq, columns=labels)
    df = df.join(table, on="formula")
    print(f"  Magpie: {len(labels)} features from {len(uniq)} unique formulas")
    return df, labels


def feature_sets(df, magpie_labels, composition_only):
    """Build the feature sets, dropping columns that are constant in this data."""
    constant = [c for c in magpie_labels if df[c].nunique(dropna=True) <= 1]
    magpie = [c for c in magpie_labels if c not in constant]
    if constant:
        print(f"  dropped {len(constant)} Magpie features constant across "
              f"this dataset")
    sets = {"composition-only": magpie + COMPOSITION_EXTRA}
    if not composition_only:
        sets = {"full": magpie + COMPOSITION_EXTRA + STRUCTURE_DFT_FEATURES, **sets}
    return sets


# ---------------------------------------------------------------------------
def audit_leakage(df, columns, groups):
    """Screen each column for target leakage before any model sees it.

    Three checks per column:
      exact     share of rows where the column equals the target (1e-6 tol),
                ignoring rows where both are 0 - elemental ground states have
                E_form = 0, and so do many range/avg_dev Magpie stats and every
                metal's band gap, which is coincidence rather than leakage
      solo_R2   grouped-split R2 of a small forest trained on that column alone
      grp_const share of multi-member groups in which the column never varies

    exact above threshold -> target leakage; the run halts.
    solo_R2 above threshold -> warning only. A strong single predictor is not
    proof of leakage: elements sit at E_form >= 0 while nitrides and borides
    are strongly negative, so n_elements alone reaches R2 ~0.8 legitimately.
    Review these by hand for a derivation from the target.
    grp_const above threshold -> the column is a chemical-system label. That is
    not target leakage, but it lets the RANDOM split memorise families, so read
    the grouped numbers.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    rows = []
    for col in columns:
        sub = pd.DataFrame({"x": df[col], "y": df["target"], "g": groups}).dropna()
        both_zero = np.isclose(sub["x"], 0) & np.isclose(sub["y"], 0)
        exact = float(np.mean(np.isclose(sub["x"], sub["y"], atol=1e-6) & ~both_zero))
        rho = sub["x"].corr(sub["y"], method="spearman")

        tr, te = next(gss.split(sub, sub["y"], sub["g"]))
        m = RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                                  random_state=SEED, n_jobs=-1)
        m.fit(sub[["x"]].iloc[tr], sub["y"].iloc[tr])
        solo_r2 = r2_score(sub["y"].iloc[te], m.predict(sub[["x"]].iloc[te]))

        sizes = sub.groupby("g")["x"].size()
        multi = sizes[sizes > 1].index
        if len(multi):
            grp_const = float((sub[sub["g"].isin(multi)]
                               .groupby("g")["x"].nunique() == 1).mean())
        else:
            grp_const = float("nan")

        flags = []
        if exact > EXACT_MATCH_FRAC:
            flags.append("LEAK:identity")
        if solo_r2 > SINGLE_FEATURE_R2:
            flags.append("review:solo-R2")
        if grp_const >= GROUP_CONST_FRAC:
            flags.append("group-proxy")
        rows.append({
            "column": col,
            "role": "excluded" if col in LEAKY_EXCLUDED else "feature",
            "exact": exact, "spearman": rho, "solo_R2": solo_r2,
            "grp_const": grp_const, "flags": ",".join(flags) or "ok",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def build_models(ridge_cv=None):
    """ridge_cv: splits used to pick Ridge's alpha (None = efficient LOO).

    Under grouped evaluation this must be a grouped split of the training fold.
    LOO lets group-proxy features memorise families, picks a tiny alpha, and
    the model then extrapolates wildly on held-out chemistries (observed:
    R2 of -1000 on a fold, predictions of ~28 eV/atom).
    """
    models = {
        "Ridge": make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 5, 33), cv=ridge_cv),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, tree_method="hist",
        )
    else:
        models["GradientBoosting*"] = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=SEED,
        )
    return models


METRICS = ("R2", "MAE", "RMSE")
WEIGHTINGS = {"pooled": "", "balanced": "_bal"}      # column-name suffixes
SPLIT_LABELS = {"random": "random", "grouped": "grouped (LOGO)",
                "element": "element (LOEO)"}


def scores(y, p, sid):
    """Pooled and system-balanced R2/MAE/RMSE.

    sid labels each row's chemical system 0..k-1 (a bootstrap copy of a
    system gets its own label). Balanced MAE/RMSE are per-system values
    averaged over systems; balanced R2 weights each row 1 / rows in its system.
    """
    err = y - p
    n = np.bincount(sid)
    w = 1.0 / n[sid]
    ybar = np.average(y, weights=w)
    return {
        "R2":       1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
        "MAE":      np.mean(np.abs(err)),
        "RMSE":     np.sqrt(np.mean(err ** 2)),
        "R2_bal":   1 - np.sum(w * err ** 2) / np.sum(w * (y - ybar) ** 2),
        "MAE_bal":  np.mean(np.bincount(sid, np.abs(err)) / n),
        "RMSE_bal": np.mean(np.sqrt(np.bincount(sid, err ** 2) / n)),
    }


def grouped_folds(X, y, groups, scheme="logo"):
    """Grouped outer splits: LOGO, or repeated GroupShuffleSplit as fallback."""
    if scheme == "logo":
        return list(LeaveOneGroupOut().split(X, y, groups))
    gss = GroupShuffleSplit(n_splits=N_GSS_SPLITS, test_size=GSS_TEST_SIZE,
                            random_state=SEED)
    return list(gss.split(X, y, groups))


def element_folds(groups):
    """Leave-one-element-out: fold per element, holding out every system with it.

    Returns (folds, elements). A binary system is held out in two folds.
    """
    members = groups.str.split("-")
    elements = sorted({el for els in members for el in els})
    folds = []
    for el in elements:
        held = members.apply(lambda els: el in els).to_numpy()
        folds.append((np.flatnonzero(~held), np.flatnonzero(held)))
    return folds, elements


def system_coverage(df):
    """Rows per requested chemical system, with its family."""
    counts = df["group"].value_counts()
    tab = pd.DataFrame({"system": [canonical(s) for s in CANDIDATE_SYSTEMS]})
    tab["family"] = tab["system"].map(FAMILY_OF)
    tab["rows"] = tab["system"].map(counts).fillna(0).astype(int)
    return tab


def fold_structure(y, groups, scheme="logo"):
    """One row per grouped outer fold: what is held out and how much it varies.

    test_target_std is the spread of the target inside the held-out fold.
    Under LOGO that is one system's polymorph spread; where it is below
    FLAT_TARGET_STD (or the fold has one row) a per-fold R2 would be noise,
    which is why evaluate() never averages per-fold R2.
    """
    rows = []
    for i, (tr, te) in enumerate(grouped_folds(y.to_frame(), y, groups, scheme)):
        held = sorted(groups.iloc[te].unique())
        one = len(held) == 1
        rows.append({
            "fold": i,
            "held_out": held[0] if one else f"{len(held)} systems",
            "family": FAMILY_OF.get(held[0], "?") if one else "mixed",
            "n_test": len(te),
            "n_train": len(tr),
            "train_systems": groups.iloc[tr].nunique(),
            "test_target_std": float(y.iloc[te].std(ddof=0)),
        })
    return pd.DataFrame(rows)


def report_structure(df, y, groups, scheme, loeo=True):
    """Print group count, coverage per family and the grouped fold layout."""
    cov = system_coverage(df)
    fam = cov.groupby("family", sort=False).agg(
        requested=("system", "size"),
        with_rows=("rows", lambda r: int((r > 0).sum())),
        rows=("rows", "sum"))
    fam["row_share"] = (fam["rows"] / fam["rows"].sum()).round(3)
    print(f"Chemical systems: {len(cov)} requested, {int((cov['rows'] > 0).sum())} "
          f"with rows, {groups.nunique()} groups in the modelling data")
    print(fam.to_string())
    missing = cov.loc[cov["rows"] == 0, "system"].tolist()
    if missing:
        print(f"  no rows: {', '.join(missing)}")
    extra = sorted(set(groups) - set(cov["system"]))
    if extra:
        print(f"  groups not in CANDIDATE_FAMILIES: {', '.join(extra)}")

    fs = fold_structure(y, groups, scheme)
    label = ("leave-one-group-out" if scheme == "logo"
             else f"{N_GSS_SPLITS} x GroupShuffleSplit, test_size={GSS_TEST_SIZE}")
    print(f"\nGrouped folds ({label}): {len(fs)}")
    print(fs.round(3).to_string(index=False))
    n = fs["n_test"]
    print(f"  test rows per fold: min {n.min()}, median {int(n.median())}, "
          f"max {n.max()}")
    print(f"  training systems per fold: {fs['train_systems'].min()}-"
          f"{fs['train_systems'].max()}")
    if scheme == "logo":
        single = int((n == 1).sum())
        flat = int(((fs["test_target_std"] < FLAT_TARGET_STD) & (n > 1)).sum())
        print(f"  {single} folds hold one row (per-fold R2 undefined); {flat} more "
              f"have target std < {FLAT_TARGET_STD} eV/atom")
    else:
        held = pd.Series([g for (_, te) in grouped_folds(y.to_frame(), y, groups, scheme)
                          for g in groups.iloc[te].unique()]).value_counts()
        never = groups.nunique() - len(held)
        print(f"  systems held out 0 times: {never}; held out per system: "
              f"min {held.min()}, max {held.max()}")

    if loeo:
        folds, elements = element_folds(groups)
        ef = pd.DataFrame({
            "element": elements,
            "systems_held": [groups.iloc[te].nunique() for _, te in folds],
            "n_test": [len(te) for _, te in folds],
        })
        print(f"\nElement folds (leave-one-element-out): {len(ef)}")
        print(ef.to_string(index=False))
    return fs


def inner_group_cv(X, y, groups):
    """Grouped splits of a training fold for Ridge's alpha search."""
    k = min(N_FOLDS, groups.nunique())     # an element fold can leave few systems
    return list(GroupKFold(k).split(X, y, groups))


def evaluate(X, y, groups, scheme="logo", loeo=True):
    """Cross-validate every model on the random, grouped and element splits.

    random   5-fold shuffled KFold
    grouped  leave-one-group-out by chemical system (or the gss fallback)
    element  leave-one-element-out (skipped when loeo=False)

    In grouped and element folds Ridge's alpha is chosen by an inner
    GroupKFold on the training fold only, so tuning never sees held-out
    chemistries.

    Each split yields one out-of-fold prediction per row (averaged where a
    row is held out more than once; rows never held out are dropped).
    Metrics come from scores(), pooled and system-balanced, with SEs from
    N_BOOT resamples of whole chemical systems shared across every model,
    split and weighting.

    Returns (results, oof). results has one row per model with
    <split>_<metric>[_bal] and _se, plus paired differences:
      gap_R2[_bal]           random - grouped
      vs_ridge_R2[_bal]      grouped, model - Ridge
      pooled_vs_bal_R2       grouped pooled - grouped balanced
      elem_drop_R2[_bal]     grouped - element   (loeo only)
    oof holds the out-of-fold predictions for every model and split.
    """
    outer = {
        "random":  list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X)),
        "grouped": grouped_folds(X, y, groups, scheme),
    }
    if loeo:
        outer["element"] = element_folds(groups)[0]
    names = list(build_models())
    oof = {}                                       # (model, split) -> predictions
    for split, folds in outer.items():
        total = {name: np.zeros(len(y)) for name in names}
        hits = np.zeros(len(y))
        for tr, te in folds:
            ridge_cv = None
            if split != "random":
                ridge_cv = inner_group_cv(X.iloc[tr], y.iloc[tr], groups.iloc[tr])
            hits[te] += 1
            for name, model in build_models(ridge_cv).items():
                model.fit(X.iloc[tr], y.iloc[tr])
                total[name][te] += model.predict(X.iloc[te])
        with np.errstate(invalid="ignore", divide="ignore"):
            for name in names:
                oof[(name, split)] = total[name] / hits   # nan if never held out

    scored = np.all([np.isfinite(p) for p in oof.values()], axis=0)
    if not scored.all():
        print(f"  {int((~scored).sum())} rows never held out by the grouped "
              "split; scored on the rest")
    yt = y.to_numpy()[scored]
    sid, uniq = pd.factorize(groups.to_numpy()[scored])
    pred = {k: p[scored] for k, p in oof.items()}

    point = {k: scores(yt, p, sid) for k, p in pred.items()}
    members = [np.flatnonzero(sid == g) for g in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    boot = {}                                      # (model, split, metric) -> draws
    for _ in range(N_BOOT):
        draw = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([members[g] for g in draw])
        copy = np.repeat(np.arange(len(draw)), [len(members[g]) for g in draw])
        for k, p in pred.items():
            for metric, v in scores(yt[rows], p[rows], copy).items():
                boot.setdefault(k + (metric,), []).append(v)
    boot = {k: np.asarray(v) for k, v in boot.items()}

    def diff(a, b, metric):
        """Point difference a - b and its paired bootstrap SE."""
        return (point[a][metric] - point[b][metric],
                (boot[a + (metric,)] - boot[b + (metric,)]).std())

    out = []
    for name in names:
        rec = {"model": name}
        for split in outer:
            for metric, v in point[(name, split)].items():
                rec[f"{split}_{metric}"] = v
                rec[f"{split}_{metric}_se"] = boot[(name, split, metric)].std()
        for suffix in WEIGHTINGS.values():
            m = "R2" + suffix
            rec[f"gap_{m}"], rec[f"gap_{m}_se"] = diff(
                (name, "random"), (name, "grouped"), m)
            rec[f"vs_ridge_{m}"], rec[f"vs_ridge_{m}_se"] = diff(
                (name, "grouped"), ("Ridge", "grouped"), m)
            if loeo:
                rec[f"elem_drop_{m}"], rec[f"elem_drop_{m}_se"] = diff(
                    (name, "grouped"), (name, "element"), m)
        rec["pooled_vs_bal_R2"] = rec["grouped_R2"] - rec["grouped_R2_bal"]
        rec["pooled_vs_bal_R2_se"] = (boot[(name, "grouped", "R2")]
                                      - boot[(name, "grouped", "R2_bal")]).std()
        out.append(rec)

    oof_df = pd.DataFrame({"group": groups.to_numpy(), "target": y.to_numpy()})
    for (name, split), p in oof.items():
        oof_df[f"{name}_{split}"] = p
    return pd.DataFrame(out), oof_df


def pm(row, col, digits=3):
    return f"{row[col]:.{digits}f} +/- {row[col + '_se']:.{digits}f}"


def results_table(results, weighting="pooled"):
    suffix = WEIGHTINGS[weighting]
    out = pd.DataFrame({"model": results["model"]})
    for split in SPLIT_LABELS:
        if f"{split}_R2" not in results:
            continue
        for metric in METRICS:
            out[f"{split}_{metric}"] = [pm(r, f"{split}_{metric}{suffix}")
                                        for _, r in results.iterrows()]
    return out


def importances(X, y, groups, top=TOP_IMPORTANCES):
    """Permutation importance on held-out chemical systems.

    For each of 5 GroupKFold folds by chemical system, every model is fitted
    on the training systems and each feature is shuffled PI_REPEATS times in
    the held-out systems; importance is the mean drop in held-out R2,
    averaged over folds. Top `top` features by mean across models; the rest
    are summed into one row. Collinear Magpie stats share credit.
    """
    per_model = {}
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
        ridge_cv = inner_group_cv(X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        for name, m in build_models(ridge_cv).items():
            m.fit(X.iloc[tr], y.iloc[tr])
            if name != "Ridge":                    # parallelise over features instead
                m.set_params(n_jobs=1)
            r = permutation_importance(m, X.iloc[te], y.iloc[te], scoring="r2",
                                       n_repeats=PI_REPEATS, random_state=SEED,
                                       n_jobs=-1)
            per_model.setdefault(name, []).append(r.importances_mean)
    tab = pd.DataFrame({name: np.mean(v, axis=0) for name, v in per_model.items()},
                       index=X.columns.str.replace("MagpieData ", "", regex=False))
    tab = tab.loc[tab.mean(axis=1).sort_values(ascending=False).index]
    head = tab.head(top).copy()
    if len(tab) > top:
        head.loc[f"(other {len(tab) - top} features, summed)"] = tab.iloc[top:].sum()
    return head.round(4)


def judge_gap(gap, se, what):
    """One line on a paired difference; 'large'/'modest' only beyond 2 SE."""
    if abs(gap) <= se:
        return f"{what} is within one SE of zero; indistinguishable from zero."
    if abs(gap) <= 2 * se:
        return f"{what} is 1-2 SE from zero: unresolved, neither large nor small."
    if gap > 0.15:
        return f"{what} is large (> 2 SE)."
    if gap > 0:
        return f"{what} is modest but real (> 2 SE)."
    return f"{what} is negative by > 2 SE. Unexpected - check the group labels."


def interpret(results, label):
    """Plain-language reading of the scores.

    A difference is only called large or modest when it clears 2 paired SEs.
    The old rule compared the gap with the largest fold std and called 0.470
    against 0.461 "large".
    """
    loeo = "element_R2" in results
    best = results.loc[results["grouped_R2"].idxmax()]
    best_bal = results.loc[results["grouped_R2_bal"].idxmax()]
    ridge = results[results["model"] == "Ridge"].iloc[0]

    print(f"\n[{label}]")
    print(f"  Best on grouped CV, pooled:   {best['model']} "
          f"(R2 {pm(best, 'grouped_R2')}, MAE {pm(best, 'grouped_MAE')})")
    print(f"  Best on grouped CV, balanced: {best_bal['model']} "
          f"(R2 {pm(best_bal, 'grouped_R2_bal')}, MAE {pm(best_bal, 'grouped_MAE_bal')})")

    for suffix, name in (("", "pooled"), ("_bal", "balanced")):
        print(f"  Random-to-grouped R2 gap, {name}: {pm(best, 'gap_R2' + suffix)}")
        print(f"    -> {judge_gap(best['gap_R2' + suffix], best['gap_R2' + suffix + '_se'], 'The gap')}")

    d, d_se = best["pooled_vs_bal_R2"], best["pooled_vs_bal_R2_se"]
    print(f"  Pooled minus balanced grouped R2 ({best['model']}): {d:.3f} +/- {d_se:.3f}")
    if best["model"] != best_bal["model"] or abs(d) > 2 * d_se:
        print("    -> FINDING: pooled and system-balanced scores disagree"
              + (f" (best model {best['model']} pooled vs {best_bal['model']} balanced)"
                 if best["model"] != best_bal["model"] else "")
              + ".")
        print("       The pooled number is carried by the large systems; the balanced")
        print("       number is how the model does on a typical chemical system.")
        print("       Report both.")
    else:
        print("    -> Pooled and balanced agree within 2 SE; the large systems are "
              "not distorting the headline.")

    if loeo:
        print(f"  Leave-one-element-out R2 ({best['model']}): pooled "
              f"{pm(best, 'element_R2')}, balanced {pm(best, 'element_R2_bal')}")
        print(f"    -> {judge_gap(best['elem_drop_R2'], best['elem_drop_R2_se'], 'The drop from grouped to element (pooled)')}")

    if best["model"] != "Ridge":
        d, d_se = best["vs_ridge_R2"], best["vs_ridge_R2_se"]
        if d < 0.05 or d <= 2 * d_se:
            print(f"  {best['model']} beats Ridge by only {d:.3f} +/- {d_se:.3f} "
                  f"pooled grouped R2 (Ridge {pm(ridge, 'grouped_R2')}).")
            print("  Prefer Ridge - the interpretability is worth more than the "
                  "marginal accuracy,")
            print("  and it is a defensible choice to justify in a viva.")


DEMO_FEATURESET_WARNING = (
    "DEMO: the feature-set comparison carries no information until real data "
    "is run -\n  the synthetic target depends mainly on the DFT-style "
    "features, so composition-only\n  loses by construction.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data (pipeline smoke test only)")
    ap.add_argument("--key", help="Materials Project API key")
    ap.add_argument("--composition-only", action="store_true",
                    help="run only Magpie + n_elements (no DFT/structure features)")
    ap.add_argument("--grouped", choices=["logo", "gss"], default="logo",
                    help="grouped split: leave-one-group-out (default) or "
                         f"{N_GSS_SPLITS} x GroupShuffleSplit if LOGO is too slow")
    ap.add_argument("--max-ehull", default=str(MAX_E_HULL),
                    help="MP mode: drop rows with energy_above_hull above this "
                         "(eV/atom); 'none' keeps all rows")
    ap.add_argument("--no-loeo", action="store_true",
                    help="skip the leave-one-element-out split")
    ap.add_argument("--folds-only", action="store_true",
                    help="print group count and fold structure, then stop")
    ap.add_argument("--out", default="model_comparison.csv")
    ap.add_argument("--audit-out", default="leakage_audit.csv")
    ap.add_argument("--oof-out", default="oof_predictions.csv")
    ap.add_argument("--data-out", default=None,
                    help="MP mode: write every retrieved entry (with a "
                         "kept/dropped column) to this CSV")
    args = ap.parse_args()

    if not args.demo and not args.key:
        sys.exit("Give --demo (synthetic smoke test) or --key YOUR_MP_API_KEY.")
    check_candidates()

    print("=" * 68)
    if args.demo:
        print("SYNTHETIC DEMO MODE")
        print("These numbers describe a made-up function, not real materials.")
        print("Use this only to confirm the pipeline runs.")
        print(DEMO_FEATURESET_WARNING)
        df = make_synthetic()
    else:
        print("MATERIALS PROJECT MODE - target: formation energy per atom")
        df = fetch_mp(args.key)
        cutoff = None if args.max_ehull.lower() == "none" else float(args.max_ehull)
        raw = df.copy()
        df = drop_unphysical(df, cutoff)
        if args.data_out:
            raw["family"] = raw["group"].map(FAMILY_OF)
            raw["kept_for_ml"] = raw["material_id"].isin(df["material_id"])
            raw.to_csv(args.data_out, index=False)
            print(f"  all {len(raw)} retrieved entries written to {args.data_out}")
    loeo = not args.no_loeo
    df, magpie_labels = add_magpie(df)
    sets = feature_sets(df, magpie_labels, args.composition_only)
    print("=" * 68)

    for name, cols in sets.items():
        overlap = set(cols) & set(LEAKY_EXCLUDED)
        if overlap:
            sys.exit(f"Leaky columns in feature set '{name}': {sorted(overlap)}")

    all_features = list(dict.fromkeys(c for cols in sets.values() for c in cols))
    before = len(df)
    df = df.dropna(subset=all_features + ["target"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with missing features")

    if not HAS_XGB:
        print("\nNote: xgboost not installed, using sklearn GradientBoosting "
              "instead (marked *).\n      Install with: pip install xgboost\n")

    y, groups = df["target"], df["group"]
    grouped_label = "LOGO" if args.grouped == "logo" else f"{N_GSS_SPLITS}x GSS"
    print(f"{len(df)} samples, {groups.nunique()} groups; random {N_FOLDS}-fold "
          f"KFold, grouped {grouped_label}" + (", element LOEO" if loeo else ""))
    for name, cols in sets.items():
        print(f"  {name:<17} {len(cols)} features")
    print()

    report_structure(df, y, groups, args.grouped, loeo)
    if args.folds_only:
        print("\n--folds-only: stopping before the leakage audit and models.")
        return
    print()

    # --- leakage audit -----------------------------------------------------
    audit = audit_leakage(df, all_features + list(LEAKY_EXCLUDED), groups)
    audit.to_csv(args.audit_out, index=False)
    print("Leakage audit (target vs each column)")
    print("-" * 68)
    shown = audit[audit["flags"].str.contains("LEAK|review") | (audit["role"] == "excluded")]
    shown = shown.assign(column=shown["column"].str.replace("MagpieData ", "", regex=False))
    print(shown.round(3).to_string(index=False) if not shown.empty else "  (no leaks)")
    print(f"  {int((audit['flags'] == 'ok').sum())} of {len(audit)} columns clean; "
          f"group proxies listed in {args.audit_out}")
    review = audit[(audit["role"] == "feature") & audit["flags"].str.contains("review")]
    if not review.empty:
        print(f"\n  {len(review)} features predict the target strongly on their own. "
              "Not halting -\n  no identity with the target - but check none is "
              "derived from it.")
    for col, why in LEAKY_EXCLUDED.items():
        print(f"\n  excluded {col}: {why}")
    bad = audit[(audit["role"] == "feature") & audit["flags"].str.contains("LEAK")]
    if not bad.empty:
        sys.exit(f"\nFeatures flagged for target leakage: {list(bad['column'])}. "
                 "Investigate before modelling.")
    proxies = audit[(audit["role"] == "feature")
                    & audit["flags"].str.contains("group-proxy")]
    if not proxies.empty:
        print(f"\n  {len(proxies)} of {len(all_features)} features are group "
              "proxies (constant within a chemical\n  system - mostly Magpie "
              "minimum/maximum/range stats). The random split can\n  memorise "
              "families through them - trust the grouped CV.")
    print()

    # --- model comparison --------------------------------------------------
    all_results, all_oof = {}, []
    for name, cols in sets.items():
        X = df[cols]
        results, oof = evaluate(X, y, groups, args.grouped, loeo)
        all_results[name] = results
        all_oof.append(oof.add_prefix(f"{name}:").rename(
            columns={f"{name}:group": "group", f"{name}:target": "target"}))

        for weighting in WEIGHTINGS:
            print(f"Model comparison - {name} ({len(cols)} features; {weighting}, "
                  f"+/- group-bootstrap SE)")
            print("-" * 68)
            print(results_table(results, weighting).to_string(index=False))
            print()

        print(f"Feature importance - {name} (permutation, mean drop in held-out "
              f"R2; top {TOP_IMPORTANCES})")
        print("-" * 68)
        imp = importances(X, y, groups)
        imp.to_csv(Path(args.out).with_name(f"permutation_importance_{name}.csv"))
        print(imp.to_string())
        print()

    combined = pd.concat([r.assign(feature_set=n) for n, r in all_results.items()],
                         ignore_index=True)
    combined.to_csv(args.out, index=False)
    oof_all = all_oof[0]
    for extra in all_oof[1:]:
        oof_all = oof_all.join(extra.drop(columns=["group", "target"]))
    oof_all.to_csv(args.oof_out, index=False)

    if len(all_results) == 2:
        full, comp = all_results["full"], all_results["composition-only"]
        side = pd.DataFrame({"model": full["model"]})
        for suffix, tag in (("", "pooled"), ("_bal", "balanced")):
            side[f"full R2 {tag}"] = [pm(r, "grouped_R2" + suffix) for _, r in full.iterrows()]
            side[f"comp-only R2 {tag}"] = [pm(r, "grouped_R2" + suffix) for _, r in comp.iterrows()]
            side[f"delta {tag}"] = (comp["grouped_R2" + suffix]
                                    - full["grouped_R2" + suffix]).round(3)
        print("Cost of dropping DFT/structure features (grouped CV)")
        print("-" * 68)
        if args.demo:
            print(DEMO_FEATURESET_WARNING)
        print(side.to_string(index=False))

    # --- interpretation ----------------------------------------------------
    print("\n" + "=" * 68)
    for name, results in all_results.items():
        interpret(results, name)

    print("\nLimitations")
    print("  - LOGO holds out one chemical system while its elements stay in "
          "training (N-Ti out,\n    Ti and C-Ti in): it tests an unseen SYSTEM, "
          "not an unseen ELEMENT."
          + (" The element\n    split above is the harder test." if loeo else ""))
    if not args.demo and cutoff is not None:
        print(f"  - For elemental systems E_hull == E_form exactly, so the "
              f"E_hull <= {cutoff} eV/atom filter\n    truncates their target "
              "range; R2 is scale-dependent, so these R2 values are not\n"
              "    comparable with an unfiltered study.")
    if args.demo:
        print("  - " + DEMO_FEATURESET_WARNING)

    print("\nValidation set for Phase 3 (literature ground truth):")
    for name, verdict, note in KNOWN_SUBSTRATES:
        print(f"  {name:<16} {verdict:<9} {note}")

    print(f"\nWritten to {args.out}, {args.audit_out} and {args.oof_out}")


if __name__ == "__main__":
    main()
