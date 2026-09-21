"""
ML assistance: external test on an unseen chemical family + applicability domain.

For materials already in Materials Project the DFT formation energy is known, so
an ML prediction adds nothing to the screen itself. What ML can do here is tell
us how far a composition-only model can be trusted for compositions WITHOUT DFT
data (Phase 2: hypothetical ternaries). This script measures that on real data:

* Train Ridge / RandomForest / XGBoost on the Phase-1 training set
  (results/mp_ml_dataset.csv, rows kept after the E_hull <= 1 eV/atom filter),
  composition-only features (Magpie + n_elements), identical hyperparameters to
  substrate_screening.py.
* External test: the transition-metal ALUMINIDES of the screening space. No
  Al-TM binary is in the training set, so this is a genuinely unseen family.
* Every screening composition gets (all C: inferred):
    pred_*            model predictions (eV/atom)
    rf_std            spread of the random-forest trees
    model_spread      max - min over the three models
    knn_dist          mean distance to the 5 nearest training compositions from
                      OTHER chemical systems, in standardised Magpie space
    in_domain         knn_dist <= 95th percentile of the same distance measured
                      for the training compositions themselves (leave-own-system-out)

Output: results/ml_domain.csv, results/ml_domain_summary.csv

Run: python adib_ml_domain.py   (after substrate_screening.py --data-out and adib_data.py)
"""

import io
import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import substrate_screening as ss

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "results" / "mp_ml_dataset.csv"
SCREEN = ROOT / "data" / "screening_summary.csv"
OUT = ROOT / "results" / "ml_domain.csv"
SUMMARY = ROOT / "results" / "ml_domain_summary.csv"
K = 5
AD_PERCENTILE = 95


def features(df):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, labels = ss.add_magpie(df)
    return df, labels


def knn_other_system(Z_query, g_query, Z_train, g_train, k=K):
    """Mean distance to the k nearest training rows from a different system."""
    out = np.empty(len(Z_query))
    for i, (z, g) in enumerate(zip(Z_query, g_query)):
        d = np.sqrt(((Z_train - z) ** 2).sum(axis=1))
        d = d[g_train != g]
        out[i] = np.sort(d)[:k].mean()
    return out


def main():
    tr = pd.read_csv(TRAIN)
    tr = tr[tr["kept_for_ml"]].reset_index(drop=True)
    sc = pd.read_csv(SCREEN)
    sc = sc[sc["energy_above_hull"] <= ss.MAX_E_HULL].reset_index(drop=True)
    sc = sc.rename(columns={"formation_energy_per_atom": "dft_Ef"})
    sc["group"] = sc["chemsys"]
    sc["n_elements"] = sc["chemsys"].str.count("-") + 1

    tr, labels = features(tr)
    sc, _ = features(sc)
    cols = [c for c in labels if tr[c].nunique() > 1] + ["n_elements"]
    X, y, g = tr[cols], tr["target"], tr["group"]

    # one row per unique composition for the distance calculation
    tr_u = tr.drop_duplicates("formula")
    scaler = StandardScaler().fit(tr_u[cols])
    Z_tr, g_tr = scaler.transform(tr_u[cols]), tr_u["group"].to_numpy()
    ref = knn_other_system(Z_tr, g_tr, Z_tr, g_tr)
    threshold = np.percentile(ref, AD_PERCENTILE)

    models = ss.build_models(ss.inner_group_cv(X, y, g))
    preds = {}
    for name, m in models.items():
        m.fit(X, y)
        preds[name] = m.predict(sc[cols])
    rf = models["RandomForest"]
    tree_preds = np.stack([t.predict(sc[cols].to_numpy()) for t in rf.estimators_])

    out = sc[["material_id", "formula", "chemsys", "family", "energy_above_hull",
              "theoretical", "dft_Ef"]].copy()
    for name, p in preds.items():
        out[f"pred_{name}"] = p
    out["rf_std"] = tree_preds.std(axis=0)
    out["model_spread"] = np.ptp(np.column_stack(list(preds.values())), axis=1)
    out["knn_dist"] = knn_other_system(scaler.transform(sc[cols]), sc["group"].to_numpy(),
                                       Z_tr, g_tr)
    out["in_domain"] = out["knn_dist"] <= threshold
    out["system_in_training"] = out["chemsys"].isin(set(g))
    for name in preds:
        out[f"abs_err_{name}"] = (out[f"pred_{name}"] - out["dft_Ef"]).abs()
    out.to_csv(OUT, index=False)

    rows = []
    for label, mask in [
        ("aluminides (unseen family), all", out["family"] == "aluminide"),
        ("aluminides, in domain", (out["family"] == "aluminide") & out["in_domain"]),
        ("aluminides, out of domain", (out["family"] == "aluminide") & ~out["in_domain"]),
        ("other families (systems seen in training)", out["family"] != "aluminide"),
    ]:
        sub = out[mask]
        rec = {"subset": label, "n_entries": len(sub),
               "n_compositions": sub["formula"].nunique()}
        for name in preds:
            rec[f"MAE_{name}"] = sub[f"abs_err_{name}"].mean() if len(sub) else np.nan
        rec["mean_knn_dist"] = sub["knn_dist"].mean() if len(sub) else np.nan
        rows.append(rec)
    summ = pd.DataFrame(rows)
    alu = out[out["family"] == "aluminide"]
    if len(alu) > 5:
        from scipy.stats import spearmanr
        summ.attrs["rho"] = spearmanr(alu["knn_dist"], alu["abs_err_RandomForest"])[0]
    summ.to_csv(SUMMARY, index=False)

    print(f"Training: {len(tr)} rows, {tr['formula'].nunique()} compositions, "
          f"{g.nunique()} systems; {len(cols)} composition features")
    print(f"Applicability-domain threshold (P{AD_PERCENTILE} of training "
          f"leave-own-system-out kNN distance): {threshold:.2f}")
    print("NOTE: 'other families' share chemical systems with training - their error is\n"
          "      in-sample and NOT a generalisation estimate; the aluminides are.")
    print(summ.round(3).to_string(index=False))
    if "rho" in summ.attrs:
        print(f"Spearman(knn_dist, |RF error|) on aluminides: {summ.attrs['rho']:.2f}")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
