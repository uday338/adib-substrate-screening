# Physics-informed screening of substrates for aluminium deposition in anode-free chloroaluminate aluminium dual-ion batteries

*Computational screening framework and preliminary candidate identification (mid-semester, September 2026)*

## 1. ADIB chemistry

A chloroaluminate aluminium dual-ion battery (ADIB) stores charge at both electrodes
from the electrolyte:

* **Negative electrode (charge):** 4 Al2Cl7⁻ + 3 e⁻ → Al + 7 AlCl4⁻. Aluminium is
  electroplated. Only a Lewis-acidic melt (AlCl3 : organic chloride > 1) contains
  Al2Cl7⁻ and can deposit and strip Al reversibly (Shi, Zhang & Guo, *ACS Energy
  Lett.* 2019).
* **Positive electrode (charge):** AlCl4⁻ intercalates into graphite (C_n + AlCl4⁻ →
  C_n[AlCl4] + e⁻).
* **Electrolyte:** AlCl3–[EMIm]Cl ionic liquid. It is also the anolyte whose Al2Cl7⁻
  content limits capacity (Kravchyk, Seno & Kovalenko, *ACS Energy Lett.* 2020).

The chloroaluminate melt contains no free Al³⁺, only AlCl4⁻ and Al2Cl7⁻. The
deposited species is necessarily aluminium: "replacing Al" would mean a different
battery.

## 2. Research gap and why the substrate matters

In an **anode-free** ADIB, Al is plated during the first charge onto a bare
current collector, the *substrate*, and stripped from it on discharge. The
substrate is exposed to three challenges an ordinary Al-foil anode never meets:

1. **Chemical/electrochemical attack by the melt.** After all Al is stripped, the
   bare collector is held at potentials positive of Al/Al³⁺. Many metals are
   oxidised (chlorinated) by the chloroaluminate melt in this range (Wang et al.,
   *Adv. Sci.* 2018; Oh, Lee & Tak, *ChemElectroChem* 2018).
2. **Alloying with the deposited Al.** Al can form intermetallics with the
   substrate, consuming Al irreversibly; Al underpotential deposition on V yields
   Al3V/AlV3 (Cvetković et al., *Metals* 2021, in an AlCl3–NaCl melt at 200 °C).
3. **Nucleation and morphology.** Deposit morphology and reversibility depend on
   metal–substrate bonding (Zheng et al., *Nat. Energy* 2021).

Current-collector studies in the ADIB literature concern almost exclusively the
**positive** electrode (collectors held at ~2–2.5 V vs Al). This literature search (September 2026,
limited to accessible abstracts and open full texts) found no systematic screen of
negative-side (anode-free) substrates across materials classes. The
research question is therefore:

> *Which conductive materials can serve as substrates for stable and reversible Al
> deposition in chloroaluminate ADIBs, and what computable descriptors separate
> them?*

## 3. Framework

```
literature ──► candidate space (6 families, stated rationale)
                 │
Materials Project retrieval (830 entries, 67 chemical systems; cached)
                 │
F1 thermodynamic stability  (E_hull ≤ 90th pct metastability, Sun et al. 2016)
F2 metallic (PBE)            (conduction proxy)
                 │
electrolyte stability window (grand-potential phase diagrams open to Al & Cl)
      φ_ox  chlorination onset    φ_red  Al-alloying limit    r0  Al reactivity at 0 V
                 │
Pareto ranking (no weights) ─► sensitivity (anodic cut-off, E_hull window,
                 │               energy route) ─► confidence label
known-material validation (pre-registered rules)
                 │
shortlist for DFT / experiment (Phase 2)
```

ML (Ridge / RandomForest / XGBoost, composition-only Magpie features) runs
alongside: it is benchmarked with leakage-controlled grouped validation and tested
on an unseen family. Its role is to decide whether composition-only prediction
can be trusted for Phase-2 hypothetical compositions. It does not rank materials
that already have DFT data.

### 3.1 The electrolyte stability window (key descriptor)

The melt is treated as a reservoir of Al and Cl tied through AlCl3. At potential φ
vs Al/AlCl3, μ_Al = μ_Al° − 3eφ and μ_Cl = [g(AlCl3) − μ_Al]/3. For each substrate
composition a grand-potential phase diagram (pymatgen) is built on a 0.01 V grid
from −1.5 to 3.5 V using every Materials Project entry in the {substrate, Al, Cl}
space. The substrate reacts when exchanging Al or Cl with the reservoir lowers its
grand potential; the products tell chlorination (Cl-containing products) from
alloying (extra Al). φ_ox is the potential **below which chlorination is
thermodynamically impossible**, a lower bound on the measured anodic limit.

Deliberately neglected (0 K DFT, bulk only): AlCl3 activity in acidic melts,
dissolution/complexation of chloride products (lowers the real onset), gas
entropy, kinetics and passive films, surface energetics.

**Two energy routes.** Materials Project mixes GGA/GGA+U and r²SCAN energies. The
client-side route (`get_entries_in_chemsys`) mixes consistently over each whole
{substrate, Al, Cl} space; the server's thermo documents mix per material's own
chemical system. Chloride formation energies differ between the routes (e.g.
AlCl3: −1.69 vs −1.97 eV/atom), so φ_ox is computed with both. The client route
is primary; the spread is reported as a systematic (method) uncertainty.

### 3.2 Knowledge types

| tag | meaning | where |
|---|---|---|
| A computed | MP properties and quantities derived deterministically from MP energies | `data/`, `results/screening_candidates.csv` (tagged per column) |
| B literature | experimental observations with conditions, DOI, verification status | `data/literature_benchmark.csv` |
| C inferred | ML predictions, tiers, confidence labels | `results/` |

### 3.3 ML methodology

* **Target:** MP formation energy per atom (full coverage). A benchmark target, not
  the screening criterion.
* **Data:** 1,653 entries in 76 chemical systems (nitrides, carbides, borides,
  oxides, silicides, phosphides, sulfides, elements); 87 entries more than
  1 eV/atom above the hull removed by a pre-registered cut-off → 1,566 entries.
* **Features:** 132 Magpie statistics (127 non-constant) + n_elements
  (composition-only); optionally + density, volume/atom, band gap, nsites
  ("full", which is circular for new materials).
* **Leakage audit before fitting:** identity with the target (halts),
  single-feature grouped R², within-system constancy. `energy_above_hull` is
  excluded as a feature (it equals the target for every elemental entry).
* **Validation:** random 5-fold KFold; **leave-one-chemical-system-out** (76
  folds, headline); **leave-one-element-out** (harder: every system containing the
  element is held out). Ridge's α is tuned by grouped CV inside each training
  fold.
* **Scoring:** pooled out-of-fold metrics *and* system-balanced metrics (each of
  the 76 systems counts equally; oxides are 44% of rows), each with a standard
  error from 1,000 bootstrap resamples of whole chemical systems. Differences
  (random vs grouped, model vs Ridge) get paired SEs.
* **Importance:** permutation importance on held-out chemical systems.

## 4. Dataset

| set | content | source |
|---|---|---|
| Screening space | 830 MP entries, 330 compositions, 67 chemical systems in 6 families (element 19, nitride 9, carbide 9, boride 9, silicide 9, aluminide 12) | `data/screening_summary.csv` |
| Electrolyte phase diagrams | every MP entry in 55 {substrate, Al, Cl} spaces, two energy routes | `data/entries/`, `data/thermo/` |
| ML benchmark | 1,653 MP entries, 76 chemical systems; 1,566 after the pre-registered E_hull ≤ 1 eV/atom cut | `results/mp_ml_dataset.csv` |
| Literature | 23 rows; 11 verified verbatim from full text, 5 from abstracts, 2 search summaries (not scored), 5 unsourced labels (kept to expose the gap) | `data/literature_benchmark.csv` |

Surface data are sparse: MP work functions and surface energies exist for 18 of
330 compositions (elements only).

## 5. Results

### 5.1 ML benchmark (formation energy, composition-only)

Real Materials Project data: 1,566 entries, 76 chemical systems, composition-only
features (Magpie + n_elements). R² ± group-bootstrap SE (1,000 resamples of whole
systems). Source: `results/model_comparison.csv`.

| model | random 5-fold | leave-one-system-out (pooled) | leave-one-system-out (balanced) | leave-one-element-out (pooled) | leave-one-element-out (balanced) |
|---|---|---|---|---|---|
| Ridge | 0.940 ± 0.017 | 0.870 ± 0.027 | 0.825 ± 0.047 | 0.747 ± 0.035 | 0.706 ± 0.048 |
| Random forest | 0.966 ± 0.010 | **0.931 ± 0.021** | **0.920 ± 0.021** | 0.634 ± 0.095 | 0.741 ± 0.040 |
| XGBoost | 0.966 ± 0.010 | 0.922 ± 0.014 | 0.920 ± 0.020 | 0.634 ± 0.076 | 0.748 ± 0.035 |

Grouped MAE of the random forest: 0.234 ± 0.021 eV/atom pooled, 0.202 ± 0.011
balanced.

* **Unseen chemical systems.** The best model loses little when whole chemical
  systems are held out: the random-to-grouped gap is 0.035 ± 0.015 (pooled) and
  0.022 ± 0.008 (balanced), modest but beyond 2 SE.
* **Pooled vs balanced agree** within 2 SE (difference 0.011 ± 0.019). The 44%
  oxide share does not distort the headline.
* **Unseen elements are much harder.** Leave-one-element-out drops the tree models
  to R² ≈ 0.63 (pooled). Ridge has the higher point estimate there (0.75), but the
  SEs do not resolve the difference. Composition-only ML should not be trusted for
  chemistries containing an element absent from training.
* **DFT/structure features add nothing on real data.** Grouped R² of the full set
  (+ density, volume, band gap, nsites) minus composition-only: −0.019 (Ridge),
  0.000 (RF), −0.002 (XGBoost) pooled. The composition-only set, the only one usable
  for new materials, costs nothing. (On the synthetic data it lost by construction.)
* **Permutation importance** (held-out systems, `figures/06_feature_importance.png`):
  the trees rely on melting-temperature statistics (minimum and mode MeltingT),
  Mendeleev-number range and valence-shell (NUnfilled) spread; Ridge on elemental
  band-gap and electronegativity statistics. Collinear Magpie statistics share
  credit, so read these as families of descriptors, not single causes.

**Leakage audit (real data, 132 columns):** 67 clean, 64 group proxies (constant
within a chemical system, mostly Magpie min/max/range statistics), 0 features
identical to the target, 0 flagged for single-feature grouped R² > 0.8.
`energy_above_hull` was caught (identical to the target on 8.9% of rows, the
elemental entries) and is excluded.

**External test on an unseen family (aluminides, not in training):**

| subset | entries | RF MAE (eV/atom) | XGB MAE | Ridge MAE |
|---|---|---|---|---|
| aluminides (unseen family) | 128 | 0.158 | 0.193 | 0.276 |
| inside the applicability domain | 104 | 0.166 | 0.202 | 0.256 |
| outside the applicability domain | 24 | 0.121 | 0.153 | 0.365 |
| other families (systems seen in training; in-sample, not a generalisation estimate) | 651 | 0.118 | 0.107 | 0.233 |

The composition-only random forest generalises to an unseen family with
MAE 0.16 eV/atom. **Neither uncertainty proxy tracks its real error:** distance to
the training set gives Spearman ρ = −0.13, random-forest tree spread ρ = 0.02, and
model disagreement ρ = 0.18. The out-of-domain aluminides are not worse. Before ML
is used on hypothetical compositions (Phase 2), a working uncertainty estimate is
needed, e.g. conformal intervals calibrated on the grouped folds.

### 5.2 Screening funnel

| stage | count |
|---|---|
| MP entries in screening space | 830 |
| unique compositions | 330 |
| pass F1 (E_hull within metastability window) | 254 |
| pass F1 + F2 (metallic, PBE) | 250 |
| anodic limit φ_ox ≥ 0.5 V | 93 |
| φ_ox ≥ 1.0 V | 25 |
| φ_ox ≥ 1.0 V **and** stable against Al at 0 V | **0** |

**No composition is both anodically stable to 1 V and thermodynamically inert to
Al at the plating potential.** Almost every transition-metal substrate can
alloy with deposited Al (Al₃Ni, Al₂Cu, Al₅W, Al₄C₃ …), and the few that cannot
(Al-rich aluminides, some borides) have anodic limits below 1 V. The trade-off
between the two objectives is the central result of the screen: it is why a
Pareto ranking, not a single score, is used.

**Sensitivity** (candidates surviving vs the two thresholds):

| φ_ox cut-off (V) | E_hull window ×0.5 | ×1.0 (registered) | ×1.5 |
|---|---|---|---|
| 0.0 | 211 | 249 | 270 |
| 0.5 | 84 | 93 | 102 |
| 1.0 | 20 | 25 | 29 |
| 1.5 | 3 | 3 | 3 |
| 2.0 | 1 | 1 | 1 |

The shortlist depends far more on the required anodic potential than on the
stability window. Above 1.5 V only C (graphite), Au and Pt remain; above 2.0 V
only graphite.

**Method (energy-route) uncertainty:** φ_ox differs between the two MP energy
routes by a median of 0.02 V (90th percentile 0.16 V, maximum 0.42 V). 88% of
candidates keep the same tier under both routes; the rest are labelled
"lower confidence".

### 5.3 Known-material validation (rules fixed beforehand)

| material | measured (Wang et al. 2018, AlCl3:[EMIM]Cl = 2) | computed φ_chlor (V) | verdict |
|---|---|---|---|
| Cr | onset 0.62 V | 0.50 | agrees |
| Au | etched at 2.0 V | 1.73 | agrees |
| Pt | not stable above 2 V | 1.55 | agrees |
| Ti | onset 1.1 V | 0.10 | agrees: kinetically stable beyond the bound |
| Al (positive side) | onset 1.0 V | 0 (reservoir metal) | agrees: native oxide |
| TiN | stable to ≥ 2.5 V (electrolyte-limited) | 0.83 | agrees: kinetically stable beyond the bound |
| Cr2N | stable to ≥ 2.5 V | 0.36 | agrees: kinetically stable beyond the bound |
| V (AlCl3–NaCl, 200 °C; Cvetković 2021) | Al UPD forms Al3V, AlV3 | φ_red = 0.17 V > 0; predicts Al3V at 0 V | agrees |
| W, Mo, glassy carbon, Ni, stainless steel | qualitative only | | insufficient evidence |
| Cu, Ag, Mg, Fe, graphite paper | labels with no verified source | | insufficient evidence |

* **No disagreement:** no material corrodes below its computed bound.
* Spearman ρ = 0.74 between measured and computed onsets (n = 4: Cr, Ti, Au, Pt).
  With four points this shows consistency, not significance.
* **The framework cannot recognise kinetically protected materials.** TiN and
  Cr2N, the best measured positive collectors, survive ~1.7–2.1 V beyond their
  thermodynamic bound, as do Ti and Al with native oxides. The bulk descriptor
  guarantees stability; it does not predict it. This locates the missing physics:
  passivation and dissolution kinetics.
* **F1 wrongly rejects Cr2N.** It is ICSD-known and measured stable, yet MP places
  it 245 meV/atom above the hull, beyond even the nitride window. A plausible but
  unverified cause is DFT error for magnetic Cr nitrides. The threshold was not
  changed; the failure is reported.
* The established collectors W, Mo and TiN all rank **low**: their measured
  stability is kinetic, not a thermodynamic guarantee.

### 5.4 Shortlist: high-priority candidates for future validation

Registered Pareto tiers, shown for the lowest registered anodic scenario
(φ_ox ≥ 0.5 V). Without that scenario the first Pareto layer also contains
Al-rich aluminides (Al6Fe, Al10V, Al9Fe2, Al23V4): they take up no more Al
(r0 = 0), but Al is extracted from them just above 0 V (φ_ox = 0.01 V). A Pareto
front always includes the extreme of each single objective. Full table with
reasons and literature status: `results/shortlist.csv`.

**High priority (18):**

| material | family | φ_ox (V) primary / alt. | anodic products | Al at 0 V | E_hull (meV/atom) | confidence |
|---|---|---|---|---|---|---|
| C (graphite) | element | 2.06 / 1.82 | CCl4 | alloys → Al4C3 | 3 | higher |
| Au | element | 1.73 / 1.58 | AuCl | alloys → Al2Au | 0 | higher |
| Pt | element | 1.55 / 1.45 | PtCl2 | alloys → Al21Pt8 | 0 | higher |
| W2N3 | nitride | 1.30 / 1.22 | N2 + WNCl3 | reacts | 0 (hypothetical in MP) | medium |
| Nb5N6 | nitride | 1.14 / 0.87 | N2 + NbCl5 | reacts | 0 | lower |
| Ag | element | 1.13 / 1.17 | AgCl | alloys → AlAg3 (weak, −0.10 eV/atom) | 0 | higher |
| VN | nitride | 1.13 / 1.26 | N2 + VCl3 | reacts | 0 (hypothetical in MP) | medium |
| WC | carbide | 1.11 / 0.99 | C + W6CCl18 | reacts | 0 | lower |
| WB | boride | 1.08 / 0.94 | BCl3 + WCl2 | reacts (−0.26) | 0 | higher |
| MoB2 | boride | 0.97 / 0.81 | BCl3 + MoB | reacts (−0.17) | 0 | lower |
| WB2 | boride | 0.95 / 0.80 | BCl3 + WB | reacts (−0.13) | 0 | lower |
| NbB2 | boride | 0.93 / 0.82 | BCl3 + Nb3Cl8 | reacts (−0.02) | 0 | lower |
| TiB2 | boride | 0.90 / 0.77 | BCl3 + TiCl4 | **stable** | 0 | lower |
| VB2 | boride | 0.90 / 0.85 | Al2VCl8 + BCl3 | **stable** | 0 | higher |
| V3B100 | boride | 0.87 / 0.71 | BCl3 + VB2 | reacts | 33 (hypothetical) | lower |
| CrB4 | boride | 0.81 / 0.76 | B + CrCl3 | **stable** | 0 | lower |
| TaB2 | boride | 0.75 / 0.75 | BCl3 + TaCl5 | **stable** | 0 | lower |
| V2B3 | boride | 0.59 / 0.85 | Al2VCl8 + VB2 | **stable** | 0 | higher |

**Medium priority (19):** Mo15N16, Nb2N3, Nb4N5, Co, V2N3, V3N4, Ta3N5, Ni, Cu,
MoB, MoB3, Nb2B3, Ta3B4, Mo2N3, W2B, Nb3B4, WSi2, Ta5B6, CrSi2.

**Rejected examples:** Zr3N4, Hf3N4, Nb3N5 (F2: not metallic); CrB2, Cr2B,
CrSi, W9C4, CrC, Al8Cr5 (F1: above metastability window); Al6Fe, Al10V (Al
extracted at 0.01 V); Ti (chlorinated at 0.10 V); Mg (no stability window at any
potential: displaced by Al).

**What the shortlist says.** Two groups stand out for different reasons:

1. **High anodic limit** (graphite, Au, Pt, W/Nb/V nitrides, WC, W borides):
   thermodynamically safe after full Al stripping up to ~1–2 V, but able to
   alloy with or reduce by Al at the plating potential.
2. **Inert to Al at 0 V** (TiB2, VB2, CrB4, TaB2, V2B3): the only metallic,
   stable compounds that cannot react with deposited Al, with anodic limits of
   0.6–0.9 V, adequate if the negative collector is not driven above ~0.6 V.

Of the 250 metallic, stable compositions, only 11 are both inert to Al at 0 V
and anodically stable to ≥ 0.5 V: nine transition-metal borides (TiB2, VB2, CrB4,
TaB2, V2B3, Ta3B4, V3B4, ZrB2, HfB2) and two disilicides (CrSi2, TaSi2). Borides
also populate the high-anodic-limit group (WB, MoB2, WB2), which makes
transition-metal borides the most consistent family in the screen. This literature
search found no chloroaluminate experiment on them, so they are the clearest new,
testable prediction.

## 6. Limitations

1. **Bulk thermodynamics only.** φ_ox is a lower bound; kinetics and passive films
   (which make TiN, Cr2N, W, Mo work in practice) are outside the model. The
   ranking therefore under-rates kinetically protected materials.
2. **Surface physics absent.** Al nucleation affinity (adsorption/binding energy,
   interfacial energy, lattice mismatch) needs slab calculations: "Not available
   in the current dataset; proposed for Phase 2/DFT." Zheng et al. (2021) suggest
   it controls reversibility.
3. **Electrolyte idealised.** AlCl3 at unit activity; chloride products not
   dissolved/complexed (real onsets lower); no organic cation; 0 K.
4. **DFT error.** GGA/GGA+U/r²SCAN mixing: chloride formation energies differ by
   up to 0.4 eV/atom between MP routes (φ_ox spread ≤ 0.42 V); Cr2N misplaced by
   F1. MP gives no per-entry uncertainty for mixed entries.
5. **Conduction proxy.** PBE metallicity ≠ conductivity; native oxides ignored.
6. **Literature sparse and mostly positive-electrode.** 4 numeric onsets from one
   paper; negative-side (anode-free) per-material results were not accessible, so
   the original Ni/Cu/Ag/Mg/graphite-paper labels remain unverified.
7. **ML uncertainty unsolved.** Composition-only models generalise moderately,
   but no tested proxy predicts their error.
8. **Scope of LOGO.** Holding out N-Ti while training on Ti and C-Ti tests
   generalisation to an unseen chemical *system*, not an unseen *element*;
   leave-one-element-out is reported as the harder test.
9. **The E_hull ≤ 1 eV/atom ML filter truncates the target range** for elemental
   rows (E_hull = E_form there). R² is scale-dependent, so the ML R² values are
   not comparable with an unfiltered study.

## 7. Claims this work does not make

It has not discovered a substrate. ML does not predict ADIB performance. No
nucleation, adsorption or corrosion-rate values are reported. Four literature
points support consistency, not statistical significance. The computed potentials
are thermodynamic bounds, not predictions of measured onsets.

## 8. Next steps (November)

1. **DFT slabs** for the Al-inert diborides (TiB2, VB2, CrB4) and the references
   W, TiN, graphite: Al adsorption energy and Al/substrate interfacial energy on
   low-index surfaces, plus lattice mismatch with Al(111).
2. **Kinetic/passivation descriptor:** chloride-product solubility or passive-layer
   formation (e.g. oxide/nitride surface layers), to explain TiN/Cr2N.
3. **Electrolyte realism:** AlCl3 activity in acidic melts; complexed products
   (MCl_n–AlCl3 adducts are already partly captured, e.g. Al2VCl8, AlCuCl4).
4. **Literature:** obtain the per-material data of Wang L. et al. (2022, anode-free)
   and Shi et al. (2019), and resolve the Cu/Ag/Mg/graphite-paper labels.
5. **ML:** conformal intervals on grouped folds; then screen hypothetical ternary
   borides/nitrides with an applicability check.
6. **Experiment (if collaboration is available):** CV of sputtered TiB2 / VB2 vs
   TiN and W in AlCl3:[EMIm]Cl, and Al plating/stripping Coulombic efficiency.

## References (verified; see `data/literature_benchmark.csv` for the exact evidence text)

1. Wang S., Kravchyk K.V., Filippin A.N., Müller U., Tiwari A.N., Buecheler S., Bodnarchuk M.I., Kovalenko M.V. *Adv. Sci.* **5**, 1700712 (2018). doi:10.1002/advs.201700712
2. Oh Y., Lee G., Tak Y. *ChemElectroChem* **5**, 3348–3352 (2018). doi:10.1002/celc.201800892
3. Shi J., Zhang J., Guo J. *ACS Energy Lett.* **4**, 2124–2129 (2019). doi:10.1021/acsenergylett.9b01285
4. Reed L.D., Menke E. *J. Electrochem. Soc.* **160**, A915–A917 (2013). doi:10.1149/2.114306jes
5. Kravchyk K.V., Seno C., Kovalenko M.V. *ACS Energy Lett.* **5**, 545–549 (2020). doi:10.1021/acsenergylett.9b02832
6. Cvetković V.S. et al. *Metals* **11**, 123 (2021). doi:10.3390/met11010123
7. Zheng J. et al. *Nat. Energy* **6**, 398–406 (2021). doi:10.1038/s41560-021-00797-7 (abstract-level summary only)
8. Wang L., Song X., Hu Y., Yan W., Tie Z., Jin Z. *Energy Storage Mater.* **44**, 461–468 (2022). doi:10.1016/j.ensm.2021.10.044 (per-material results not accessed)
9. Sun W. et al. *Sci. Adv.* **2**, e1600225 (2016). doi:10.1126/sciadv.1600225
10. Materials Project, accessed via mp-api 0.46.5 (September 2026); pymatgen 2026.5.4; matminer 0.10.1 (Magpie).
