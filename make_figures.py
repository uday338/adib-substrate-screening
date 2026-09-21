"""
Figures for the ADIB screening report. Reads only files in results/ and data/;
writes PNGs to figures/. A figure whose inputs are missing is skipped with a
message. Every figure's numbers are also in a results/*.csv table.

Colours: validated categorical slots (blue, orange, aqua; all-pairs CVD-safe),
one-hue blue ramp for ordinal tiers, grey for context. Aqua is below 3:1
contrast, so every chart using it carries direct labels.

Run: python make_figures.py
"""

import io
import contextlib
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, DATA, FIG = ROOT / "results", ROOT / "data", ROOT / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TIER = {"high priority": "#1c5cab", "medium priority": "#3987e5", "low priority": "#86b6ef"}
GREY, GREY_D = "#c9c8c3", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
FAMILIES = ["element", "nitride", "carbide", "boride", "silicide", "aluminide"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GREY_D, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


def need(*paths):
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"  skipped (missing {', '.join(Path(m).name for m in missing)})")
        return False
    return True


def save(fig, name, caption):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.png - {caption}")


# ---------------------------------------------------------------------------
def fig_chemical_space():
    """PCA of Magpie composition features: training set vs screening families."""
    if not need(RES / "mp_ml_dataset.csv", DATA / "screening_summary.csv"):
        return
    import substrate_screening as ss
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    tr = pd.read_csv(RES / "mp_ml_dataset.csv").drop_duplicates("formula")
    sc = pd.read_csv(DATA / "screening_summary.csv").drop_duplicates("formula")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr, labels = ss.add_magpie(tr)
        sc, _ = ss.add_magpie(sc)
    cols = [c for c in labels if tr[c].nunique() > 1]
    scaler = StandardScaler().fit(tr[cols])
    pca = PCA(2, random_state=0).fit(scaler.transform(tr[cols]))
    P_tr = pca.transform(scaler.transform(tr[cols]))
    P_sc = pca.transform(scaler.transform(sc[cols]))
    ev = pca.explained_variance_ratio_ * 100
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharex=True, sharey=True)
    for ax, fam in zip(axes.flat, FAMILIES):
        ax.scatter(P_tr[:, 0], P_tr[:, 1], s=9, color=GREY, lw=0, label="ML training set")
        m = (sc["family"] == fam).to_numpy()
        ax.scatter(P_sc[m, 0], P_sc[m, 1], s=16, color=BLUE, edgecolor=SURFACE, lw=0.6,
                   label=f"{fam} candidates")
        ax.set_title(f"{fam} ({m.sum()} compositions)")
    for ax in axes[1]:
        ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Candidate compositions in the chemical space of the ML training set "
                 "(PCA of Magpie features)", fontweight="bold")
    save(fig, "01_chemical_space", "screening families vs ML training set")


def fig_stability():
    if not need(DATA / "screening_summary.csv"):
        return
    sc = pd.read_csv(DATA / "screening_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    bins = np.linspace(0, 0.5, 26)
    for ax, fam in zip(axes.flat, FAMILIES):
        e = sc.loc[sc["family"] == fam, "energy_above_hull"]
        lim = 0.195 if fam == "nitride" else 0.067
        ax.hist(e.clip(upper=0.5), bins=bins, color=BLUE, edgecolor=SURFACE, lw=1)
        ax.axvline(lim, color=INK, ls="--", lw=1)
        ax.text(lim + 0.01, ax.get_ylim()[1] * 0.9, f"F1 limit\n{1000 * lim:.0f} meV/atom",
                fontsize=7.5, color=INK2, va="top")
        over = int((e > 0.5).sum())
        ax.set_title(f"{fam}: {len(e)} entries, {int((e <= lim).sum())} pass F1")
        if over:
            ax.text(0.98, 0.97, f"last bar: {over} entries ≥ 0.5", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=INK2)
    fig.supxlabel("Energy above hull (eV/atom), Materials Project", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("MP entries")
    fig.suptitle("Thermodynamic stability of screening entries, with the pre-registered F1 "
                 "window (Sun et al. 2016)", fontweight="bold")
    save(fig, "02_stability_distributions", "E_hull by family with F1 threshold")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    data = [sc.loc[sc["family"] == f, "formation_energy_per_atom"] for f in FAMILIES]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                    medianprops={"color": INK, "lw": 1.5}, flierprops={"ms": 3, "mfc": GREY_D, "mec": "none"})
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=BLUE)
    ax.set_yticks(range(1, 7), FAMILIES)
    ax.set_xlabel("Formation energy (eV/atom), Materials Project")
    ax.set_title("Formation-energy distribution of screening entries by family")
    save(fig, "03_formation_energy", "E_form by family")


def fig_model_performance():
    if not need(RES / "model_comparison.csv"):
        return
    mc = pd.read_csv(RES / "model_comparison.csv")
    splits = [("random", "random 5-fold", BLUE, "o"), ("grouped", "leave-one-system-out", ORANGE, "s"),
              ("element", "leave-one-element-out", AQUA, "D")]
    fsets = list(mc["feature_set"].unique())
    fig, axes = plt.subplots(len(fsets), 2, figsize=(10, 2.3 * len(fsets) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    models = list(mc["model"].unique())
    for i, fs in enumerate(fsets):
        sub = mc[mc["feature_set"] == fs].set_index("model").loc[models]
        for j, (suffix, wname) in enumerate([("", "pooled"), ("_bal", "system-balanced")]):
            ax = axes[i, j]
            for k, (sp, lab, col, mk) in enumerate(splits):
                c = f"{sp}_R2{suffix}"
                if c not in sub:
                    continue
                y = np.arange(len(models)) + (k - 1) * 0.22
                ax.errorbar(sub[c], y, xerr=sub[c + "_se"], fmt=mk, color=col, ms=6,
                            mec=SURFACE, mew=0.8, elinewidth=1.5, capsize=0, label=lab)
                for yy, v in zip(y, sub[c]):
                    ax.text(v, yy + 0.09, f"{v:.2f}", fontsize=6.5, color=INK2, ha="center", va="bottom")
            ax.set_yticks(range(len(models)), models)
            ax.set_xlim(0.45, 1.0)
            ax.set_title(f"{fs} features - {wname} R²")
    for ax in axes[-1]:
        ax.set_xlabel("R² on held-out data (± group-bootstrap SE)")
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Formation-energy models: random vs grouped validation (real MP data, "
                 "1566 entries, 76 systems)", fontweight="bold")
    save(fig, "04_model_performance", "R2 by split, model, feature set, weighting")

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    rows = []
    for fs in fsets:
        for _, r in mc[mc["feature_set"] == fs].iterrows():
            for suffix, wname in (("", "pooled"), ("_bal", "balanced")):
                rows.append((f"{r['model']}\n{fs}", wname, r[f"gap_R2{suffix}"], r[f"gap_R2{suffix}_se"]))
    g = pd.DataFrame(rows, columns=["label", "w", "gap", "se"])
    labs = list(dict.fromkeys(g["label"]))
    for k, (w, col, mk) in enumerate((("pooled", BLUE, "o"), ("balanced", ORANGE, "s"))):
        s = g[g["w"] == w].set_index("label").loc[labs]
        x = np.arange(len(labs)) + (k - 0.5) * 0.25
        ax.errorbar(x, s["gap"], yerr=s["se"], fmt=mk, color=col, ms=6, mec=SURFACE,
                    elinewidth=1.5, label=w)
    ax.axhline(0, color=GREY_D, lw=1)
    ax.set_xticks(range(len(labs)), labs, fontsize=7.5)
    ax.set_ylabel("R²(random) − R²(grouped)")
    ax.set_title("How much of the random-split score is family memorisation (paired SE)")
    ax.legend(fontsize=8)
    save(fig, "05_random_vs_grouped_gap", "random-to-grouped R2 gap with paired SE")


def parse_importances(log):
    """Read the permutation-importance tables printed by substrate_screening.py."""
    text = Path(log).read_text(encoding="utf-8", errors="ignore")
    out = {}
    for m in re.finditer(r"Feature importance - (\S+) \(permutation.*?\n-+\n(.*?)\n\n", text, re.S):
        lines = [ln for ln in m.group(2).splitlines()
                 if ln.strip() and not re.search(r"Warning|warnings\.warn", ln)]
        start = next(i for i, ln in enumerate(lines) if ln.split()[:1] == ["Ridge"])
        header = lines[start].split()
        rows = []
        for ln in lines[start + 1:]:
            parts = ln.rsplit(None, len(header))
            try:
                rows.append([parts[0].strip()] + [float(p) for p in parts[1:]])
            except ValueError:
                continue
        out[m.group(1)] = pd.DataFrame(rows, columns=["feature"] + header).set_index("feature")
    return out


def fig_importance():
    log = RES / "phase1_ml_run.log"
    if not need(log):
        return
    imps = parse_importances(log)
    if not imps:
        print("  skipped (no importance table in log yet)")
        return
    for fs, tab in imps.items():
        tab.to_csv(RES / f"permutation_importance_{fs}.csv")
    fs = "composition-only" if "composition-only" in imps else next(iter(imps))
    tab = imps[fs]
    top = tab[~tab.index.str.startswith("(other")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(top))[::-1]
    for k, (col, c, mk) in enumerate(zip(top.columns, (BLUE, ORANGE, AQUA), "osD")):
        ax.plot(top[col], y + (k - 1) * 0.2, mk, color=c, ms=5.5, mec=SURFACE, label=col)
    ax.set_yticks(y, top.index, fontsize=7.5)
    ax.axvline(0, color=GREY_D, lw=1)
    ax.set_xlabel("Mean drop in held-out R² when the feature is shuffled")
    rest = tab.loc[tab.index.str.startswith("(other")]
    note = (f"\nRemaining features summed: " + ", ".join(f"{c} {rest[c].iloc[0]:.3f}" for c in rest.columns)
            if len(rest) else "")
    ax.set_title(f"Permutation importance, {fs} features (5-fold GroupKFold by system)"
                 + note, fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "06_feature_importance", f"permutation importance ({fs})")


def fig_funnel():
    if not need(RES / "screening_funnel.csv"):
        return
    f = pd.read_csv(RES / "screening_funnel.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(f))[::-1]
    ax.barh(y, f["count"], color="#3987e5", height=0.62, edgecolor=SURFACE)
    for yy, c in zip(y, f["count"]):
        ax.text(c, yy, f"  {c}", va="center", fontsize=8.5)
    ax.set_yticks(y, f["stage"], fontsize=8.5)
    ax.set_xlabel("count")
    ax.grid(axis="y", visible=False)
    ax.set_title("Screening funnel (stages 0-5, thresholds pre-registered)")
    save(fig, "07_screening_funnel", "entries/compositions surviving each stage")


def fig_windows():
    if not need(RES / "screening_candidates.csv", DATA / "literature_benchmark.csv"):
        return
    sc = pd.read_csv(RES / "screening_candidates.csv")
    lit = pd.read_csv(DATA / "literature_benchmark.csv")
    refs = [f for f in lit["mp_formula"].dropna().unique() if f in set(sc["formula"])]
    hp = sc[sc["tier"] == "high priority"].head(20)
    show = pd.concat([hp, sc[sc["formula"].isin(refs) & ~sc["formula"].isin(hp["formula"])]])
    show = show.drop_duplicates("formula").sort_values("phi_ox")
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(show) + 1.4))
    y = np.arange(len(show))
    lo = show["phi_red"].clip(lower=-0.5).where(show["phi_red"] > -1.5, -0.5)
    for yy, (_, r), l in zip(y, show.iterrows(), lo):
        is_ref = r["formula"] in refs
        col = ORANGE if is_ref else BLUE
        ax.plot([max(l, -0.5), r["phi_ox"]], [yy, yy], color=col, lw=5, solid_capstyle="round")
        if r["kind0"] != "stable":
            ax.plot(0, yy, "x", color=INK, ms=5, mew=1.2)
    num = lit[lit["potential_type"].isin(["onset", "corrodes_at", "stable_up_to"])
              & lit["mp_formula"].isin(show["formula"])]
    pos = dict(zip(show["formula"], y))
    for _, r in num.iterrows():
        mk = {"onset": "o", "corrodes_at": "v", "stable_up_to": ">"}[r["potential_type"]]
        ax.plot(float(r["exp_potential_V_vs_Al"]), pos[r["mp_formula"]], mk, color=INK, ms=6)
    ax.set_yticks(y, [f"{f}  ({fam})" for f, fam in zip(show["formula"], show["family"])], fontsize=7.5)
    ax.axvline(0, color=GREY_D, lw=1)
    ax.set_xlabel("Potential vs Al/AlCl3 (V)")
    ax.set_xlim(-0.55, 3.0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=5, label="high-priority candidate: computed window"),
        Line2D([], [], color=ORANGE, lw=5, label="literature reference: computed window"),
        Line2D([], [], marker="x", color=INK, ls="", label="reacts with Al at 0 V (thermodynamic)"),
        Line2D([], [], marker="o", color=INK, ls="", label="measured oxidation onset"),
        Line2D([], [], marker="v", color=INK, ls="", label="measured: corrodes at"),
        Line2D([], [], marker=">", color=INK, ls="", label="measured: stable at least to"),
    ], fontsize=7, loc="lower right")
    ax.set_title("Computed electrolyte stability windows (bar ends: φ_red→φ_ox) "
                 "vs measured limits (Wang et al. 2018)", fontsize=9)
    save(fig, "08_stability_windows", "phi_red..phi_ox bars with measured points")


def fig_pareto():
    if not need(RES / "screening_candidates.csv"):
        return
    sc = pd.read_csv(RES / "screening_candidates.csv")
    cand = sc[sc["status"] == "candidate"]
    rej = sc[sc["status"].str.startswith("rejected")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(rej["phi_ox"], rej["r0"], s=10, color=GREY, lw=0, label="rejected by F1/F2")
    for tier, col in TIER.items():
        s = cand[cand["tier"] == tier]
        ax.scatter(s["phi_ox"], s["r0"], s=26, color=col, edgecolor=SURFACE, lw=0.7,
                   label=f"{tier} ({len(s)})")
    lab = cand[cand["pareto_layer"] == 1].sort_values("phi_ox")
    clustered = lab[lab["phi_ox"] < 0.3]
    spread = lab[lab["phi_ox"] >= 0.3].assign(x=lambda t: t["phi_ox"].round(2),
                                              y=lambda t: t["r0"].round(2))
    for i, ((x, y), grp) in enumerate(spread.groupby(["x", "y"], sort=True)):
        ax.annotate(" / ".join(grp["formula"]), (x, y), xytext=(0, 10 + 12 * (i % 2)),
                    textcoords="offset points", fontsize=7, color=INK, ha="center",
                    arrowprops={"arrowstyle": "-", "color": GREY_D, "lw": 0.6})
    if len(clustered):
        ax.annotate(", ".join(clustered["formula"]) + " (first layer, φ_ox ≈ 0):\n"
                    "Al-rich aluminides take up no Al,\nbut Al is extracted just above 0 V",
                    (clustered["phi_ox"].mean(), 0), xytext=(1.25, -2.55), textcoords="data",
                    fontsize=7, color=INK2)
    ax.set_ylim(top=0.45)
    ax.set_xlabel("φ_ox: anodic limit (V vs Al; chlorination or Al extraction), higher = more resistant")
    ax.set_ylabel("r0: reaction driving force with Al at 0 V (eV/atom)\n0 = does not react")
    ax.set_title("Pareto ranking (third objective, E_hull, not shown); labels = first Pareto layer")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "09_pareto", "phi_ox vs Al reactivity, tiers")


def fig_validation():
    if not need(RES / "validation.csv"):
        return
    v = pd.read_csv(RES / "validation.csv")
    v = v[v["potential_type"].isin(["onset", "corrodes_at", "stable_up_to"]) & v["phi_ox"].notna()]
    v = v[v["verification"].isin(["verbatim-full-text", "verbatim-abstract"])]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    xs = np.linspace(-0.6, 3, 10)
    ax.fill_between(xs, xs - 0.2, 3.2, color="#cde2fb", lw=0, label="agrees: measured ≥ bound − 0.2 V")
    ax.plot(xs, xs, color=GREY_D, lw=1, ls="--", label="measured = computed bound")
    for _, r in v.iterrows():
        mk = {"onset": "o", "corrodes_at": "v", "stable_up_to": "^"}[r["potential_type"]]
        col = ORANGE if "disagrees" in r["verdict"] else BLUE
        ax.plot(r["phi_ox"], float(r["exp_potential_V_vs_Al"]), mk, color=col, ms=8, mec=SURFACE)
        ax.annotate(r["substrate"], (r["phi_ox"], float(r["exp_potential_V_vs_Al"])),
                    xytext=(5, -3), textcoords="offset points", fontsize=8)
    ax.set_xlim(-0.6, 3)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("Computed φ_ox, thermodynamic lower bound (V vs Al)")
    ax.set_ylabel("Measured limit in AlCl3:[EMIM]Cl (V vs Al)")
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color=INK, ls="", label="onset"),
          Line2D([], [], marker="v", color=INK, ls="", label="corrodes at"),
          Line2D([], [], marker="^", color=INK, ls="", label="stable at least to (electrolyte-limited)")]
    ax.legend(handles=h, fontsize=7, loc="upper left")
    ax.set_title("Known-material validation: measured limits vs computed bound", fontsize=9.5)
    save(fig, "10_validation", "measured anodic limits vs computed phi_ox")


def fig_domain():
    if not need(RES / "ml_domain.csv"):
        return
    d = pd.read_csv(RES / "ml_domain.csv")
    a = d[d["family"] == "aluminide"]
    thr = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (col, lab) in zip(axes, [("knn_dist", "Distance to 5 nearest training compositions\n"
                                                  "(other systems; standardised Magpie space)"),
                                     ("rf_std", "Random-forest tree spread (eV/atom)")]):
        ax.scatter(a[col], a["abs_err_RandomForest"], s=18,
                   c=np.where(a["in_domain"], BLUE, ORANGE), edgecolor=SURFACE, lw=0.6)
        ax.set_xlabel(lab)
        from scipy.stats import spearmanr
        rho = spearmanr(a[col], a["abs_err_RandomForest"])[0]
        ax.set_title(f"Spearman ρ = {rho:.2f} (n = {len(a)} aluminide entries)", fontsize=9)
    axes[0].set_ylabel("|RF prediction − DFT| formation energy (eV/atom)")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="inside AD threshold"),
                            Line2D([], [], marker="o", color=ORANGE, ls="", label="outside AD threshold")],
                   fontsize=7.5)
    fig.suptitle("Applicability domain on an unseen family (TM aluminides): neither uncertainty "
                 "proxy tracks the real error", fontweight="bold", fontsize=10)
    save(fig, "11_applicability_domain", "error vs AD distance and RF spread")


def fig_sensitivity():
    if not need(RES / "screening_sensitivity.csv"):
        return
    s = pd.read_csv(RES / "screening_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for scale, col, mk in ((0.5, BLUE, "o"), (1.0, ORANGE, "s"), (1.5, AQUA, "D")):
        t = s[s["ehull_window_scale"] == scale]
        ax.plot(t["anodic_cutoff_V"], t["n_candidates"], "-" + mk, color=col, ms=6, mec=SURFACE)
        ax.annotate(f"E_hull window ×{scale}", (t["anodic_cutoff_V"].iloc[-1], t["n_candidates"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
    ax.set_xlabel("Required anodic stability: φ_ox ≥ cut-off (V vs Al)")
    ax.set_ylabel("Candidate compositions surviving")
    ax.set_xlim(-0.1, 2.6)
    ax.set_title("Sensitivity of the shortlist to the two screening thresholds")
    save(fig, "12_sensitivity", "survivors vs anodic cutoff and E_hull window")


def main():
    for f in (fig_chemical_space, fig_stability, fig_model_performance, fig_importance,
              fig_funnel, fig_windows, fig_pareto, fig_validation, fig_domain, fig_sensitivity):
        print(f.__name__)
        f()


if __name__ == "__main__":
    main()
