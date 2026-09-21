# Screening descriptors, filters and validation rules (pre-registered)

Written 2026-09-19, **before** any screening descriptor was computed and before the
framework was compared with any literature result. Thresholds below are not to be
changed after seeing validation outcomes. Any later change must be logged at the
bottom of this file with its reason and date.

Knowledge types used throughout the project:

| tag | meaning |
|---|---|
| **A: computed** | a Materials Project property, or a quantity we derive deterministically from MP energies |
| **B: literature** | an experimental observation from a cited paper, with conditions |
| **C: inferred** | an ML prediction or a screening conclusion drawn from A |

---

## Stage 0: candidate space (why a material is considered at all)

Families and their reasons are in `adib_data.py` (`SCREENING_FAMILIES`): elemental
metals and graphite, TM nitrides, carbides, borides, silicides, and TM aluminides.
Oxides, sulfides and phosphides are used only as Phase-1 ML training data. They are
not screened because no reason was found to treat them as Al-plating substrates;
most are insulators at the PBE level. Conductive oxides (Magnéli Ti_nO_2n-1, RuO2)
are deferred to Phase 2.

Materials that cannot be represented by one MP entry are **literature-only**:
stainless steel (a multicomponent alloy) and glassy carbon (amorphous). Graphite
(MP) is used as a *proxy* for carbon substrates and is flagged as such.

## Stage 1: thermodynamic stability (hard filter F1, A: computed)

Keep entries with `energy_above_hull` ≤ the 90th percentile of the metastability of
experimentally observed inorganic crystals:

* 67 meV/atom for all families, except
* 195 meV/atom for nitrides.

Source: Sun et al., *Sci. Adv.* 2, e1600225 (2016), doi:10.1126/sciadv.1600225,
verified sentence: "The DFT-calculated median metastability of all known inorganic
crystalline materials is 15 ± 0.5 meV/atom, and the 90th percentile is 67 ± 2
meV/atom." The nitride value is from the same paper (median 63, 90th percentile
195 meV/atom).

`energy_above_hull` is used here only as a filter, never as an ML feature.

## Stage 2: electronic conduction proxy (hard filter F2, A: computed)

Keep entries with MP `is_metal == True` (PBE band gap = 0). A current collector
must conduct electrons. Caveats: PBE underestimates band gaps, so a small-gap
semiconductor can be misclassified as a metal; metallic ≠ high conductivity; and
the filter says nothing about surface films (native oxides).

## Stage 3: electrolyte stability window (continuous descriptor, A: computed)

**Model.** The substrate is in contact with a chloroaluminate melt that acts as a
reservoir of Al and Cl tied through AlCl3. At electrode potential φ (V vs Al/AlCl3):

    μ_Al(φ) = μ_Al° − 3eφ
    μ_Cl(φ) = [g(AlCl3) − μ_Al(φ)] / 3

μ_Al° is the energy per atom of Al metal and g(AlCl3) is the hull energy of AlCl3,
both from the same MP entry set. For each substrate composition we build the
grand-potential phase diagram open to Al and Cl at each φ (pymatgen
`GrandPotentialPhaseDiagram`). The substrate is stable at φ when exchanging Al or Cl
with the reservoir cannot lower its grand potential below its closed-system hull
(tolerance 1 meV per non-reservoir atom).

Two numbers per composition:

* **φ_ox (chlorination onset)**: the lowest potential above which the equilibrium
  products contain Cl (the substrate is chlorinated, i.e. corrodes).
  Equivalent to M + (n/3) AlCl3 → MCl_n + (n/3) Al, E = ΔG/(ne).
* **φ_red (Al-alloying limit)**: the highest potential below which the products
  contain extra Al (intermetallic formation or aluminothermic reduction).
  φ_red > 0 means the substrate reacts with Al at the plating potential.

Scan range −1.5 to +3.5 V, step 0.01 V. Below 0 V the imposed μ_Al exceeds that of Al
metal (the melt would deposit Al), so a value φ_ox < 0 is reported only as its sign:
the substrate is chlorinated by the melt at open circuit (displaced by Al).

**What this descriptor is.** φ_ox is the potential up to which corrosion is
thermodynamically *impossible*. It is a **lower bound** on the observed anodic
limit: a material may survive beyond φ_ox kinetically (passivation, slow
dissolution), but should not corrode well below it. Neglected effects:

* activity of AlCl3 in an acidic melt ≠ 1 (≈ tens of mV);
* dissolution and complexation of MCl_n products, which *lowers* the real onset;
* gas entropy (N2, Cl2); temperature (0 K DFT energies);
* kinetics and passive films;
* DFT/correction error.

**How it is used.** No single cut-off. The potential an anode-free negative
collector reaches after complete Al stripping depends on the cell's discharge
cutoff, so screening is reported for cut-offs of 0.5, 1.0, 1.5 and 2.0 V vs Al
(sensitivity analysis). φ_red is reported as a flag ("alloys with Al at 0 V"), not
a hard filter: alloying consumes Al irreversibly and degrades the substrate, but
it may also aid nucleation, and the literature does not settle this.

## Stage 4: surface descriptors (reported, not filters, A: computed)

MP `weighted_work_function` and `weighted_surface_energy` (Wulff-weighted) are
available for elemental crystals only; coverage is reported. **Al nucleation
affinity** (Al adsorption / binding energy, interfacial energy, lattice mismatch)
requires surface-level calculations and will be addressed in the next phase. The
column exists and is empty: "Not available in the current dataset; proposed for
Phase 2/DFT."

## Stage 5: ranking (C: inferred)

Among compositions passing F1 and F2 (the lowest-E_hull polymorph represents each
composition), rank by **non-dominated sorting** (Pareto layers) on three computed
objectives, with no weights:

1. φ_ox, maximise;
2. Al-reaction driving force at 0 V (≤ 0 eV/atom), maximise, i.e. least reactive;
3. energy_above_hull, minimise.

Tiers: Pareto layers 1-2 = high priority, 3-4 = medium, rest = low. A composition
that fails F1/F2 is "rejected by <filter>". Experimental existence (ICSD entry,
MP `theoretical == False`) is reported next to each candidate as a confidence
indicator, not used in the ranking. Candidates are called "high-priority
candidates for future validation", never "best materials".

## ML role (C: inferred)

For every material already in MP, the DFT value is used directly; an ML
formation-energy prediction adds nothing there. The Phase-1 models are used to:

(i) benchmark how well composition-only models generalise to unseen chemical
systems (LOGO / leave-one-element-out);
(ii) test that on a genuinely unseen family: aluminides are *not* in the
Phase-1 training set, so their DFT formation energies are an external test set;
(iii) measure an applicability domain (distance to the training set in
standardised Magpie space), to be used in Phase 2 when screening hypothetical
compositions that have no DFT data.

## Validation rules (fixed before comparison)

For each literature material with a measured anodic onset or stability limit
E_exp (V vs Al) in an acidic chloroaluminate electrolyte:

* **framework agrees**: E_exp ≥ φ_ox − 0.2 V (observed stability at least the
  thermodynamic lower bound; 0.2 V allows for product dissolution/activity and DFT
  error).
* **framework disagrees**: E_exp < φ_ox − 0.2 V (corrodes well below where
  thermodynamics forbids it; the model has missed a reaction or the reference is
  wrong).
* **kinetic stability beyond the bound**: E_exp > φ_ox + 0.5 V. Not a
  disagreement, but reported separately. The descriptor cannot explain it, and
  that identifies the missing physics (passivation/kinetics).
* **insufficient evidence**: no numeric onset, a different electrolyte or
  electrode side, or the material has no MP representation.

Rank agreement between φ_ox and E_exp is reported as a Spearman correlation with
the explicit statement that n ≈ 10 supports consistency, not statistical
significance.

---

## Change log

- 2026-09-19, before any validation comparison: added the φ < 0 note above after finding that the grand-potential diagram is ill-defined there (Al supersaturation). Implementation detail only (open diagrams built from closed-hull stable entries); no threshold changed.
