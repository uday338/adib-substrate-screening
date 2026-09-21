"""
Literature benchmark for the ADIB substrate screen (knowledge type B).

Builds data/literature_benchmark.csv. Every row records WHICH electrode the
observation concerns, the electrolyte and conditions as far as they were
verified, the evidence text, and how it was verified:

    verbatim-full-text  sentence copied from the open-access full text
    verbatim-abstract   sentence copied from the published abstract
    search-summary      only a search-engine summary of the paper was accessible;
                        NOT used for scoring
    none                no primary source found; label kept only to show the gap

Rules applied when building this table:
* Positive-current-collector corrosion tests (the collector held at up to
  ~2.5 V vs Al, as a cathode support) are NOT the same experiment as an
  anode-free negative collector (Al plated near 0 V, exposed to higher
  potentials only after complete stripping). The electrode_side column keeps
  them apart. For anodic onsets, the chemistry (substrate oxidation by the
  chloroaluminate melt) is the same, so they test phi_ox directly.
* Nothing is filled in that the source does not state; unknown = "not reported"
  or "not verified".

Run: python literature_benchmark.py
"""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "literature_benchmark.csv"

WANG2018 = ("Wang S., Kravchyk K.V., Filippin A.N., Mueller U., Tiwari A.N., Buecheler S., "
            "Bodnarchuk M.I., Kovalenko M.V., Adv. Sci. 5, 1700712 (2018)", "10.1002/advs.201700712")
WANG2018_ONSET = ("The onset of electrochemical oxidation occurs at 0.62, 0.92, 1, and 1.1 V "
                  "versus Al3+/Al for chromium (Cr), SS, Al, and Ti metals, respectively.")
WANG2018_COND = "AlCl3:[EMIM]Cl"
OH2018 = ("Oh Y., Lee G., Tak Y., ChemElectroChem 5, 3348-3352 (2018)", "10.1002/celc.201800892")
SHI2019 = ("Shi J., Zhang J., Guo J., ACS Energy Lett. 4, 2124-2129 (2019)", "10.1021/acsenergylett.9b01285")
REED2013 = ("Reed L.D., Menke E., J. Electrochem. Soc. 160, A915-A917 (2013)", "10.1149/2.114306jes")
CVET2021 = ("Cvetkovic V.S., Vukicevic N.M., Feldhaus D., Milicevic K., Barudzija T., Friedrich B., "
            "Jovicevic J.N., Metals 11, 123 (2021)", "10.3390/met11010123")
ZHENG2021 = ("Zheng J. et al. (Archer group), Nat. Energy 6, 398-406 (2021)", "10.1038/s41560-021-00797-7")
WANGL2021 = ("Wang L., Song X., Hu Y., Yan W., Tie Z., Jin Z., Energy Storage Mater. 44, 461-468 (2022)",
             "10.1016/j.ensm.2021.10.044")
KRAV2020 = ("Kravchyk K.V., Seno C., Kovalenko M.V., ACS Energy Lett. 5, 545-549 (2020)",
            "10.1021/acsenergylett.9b02832")
KRAV2020B = ("Kravchyk K.V., Kovalenko M.V., Commun. Chem. 3, 120 (2020)",
             "10.1038/s42004-020-00365-2")

COLS = ["substrate", "mp_formula", "electrode_side", "electrolyte", "alcl3_ratio",
        "temperature", "method", "al_deposition", "corrosion_stability",
        "exp_potential_V_vs_Al", "potential_type", "outcome_category", "cycling_or_CE",
        "citation", "doi", "evidence_text", "verification", "notes"]

NR = "not reported in verified text"


def row(**kw):
    return {c: kw.get(c, NR) for c in COLS}


ROWS = [
    # ---- Wang et al. 2018: CV of candidate positive current collectors vs Al --------
    row(substrate="Cr", mp_formula="Cr", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al (Al foil RE/CE)",
        corrosion_stability="oxidised", exp_potential_V_vs_Al=0.62, potential_type="onset",
        outcome_category="corrosion", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text=WANG2018_ONSET, verification="verbatim-full-text"),
    row(substrate="stainless steel", mp_formula="", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al (Al foil RE/CE)",
        corrosion_stability="oxidised", exp_potential_V_vs_Al=0.92, potential_type="onset",
        outcome_category="corrosion", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text=WANG2018_ONSET, verification="verbatim-full-text",
        notes="Multicomponent alloy: no single MP entry; literature-only."),
    row(substrate="Al", mp_formula="Al", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al (Al foil RE/CE)",
        corrosion_stability="oxidised", exp_potential_V_vs_Al=1.0, potential_type="onset",
        outcome_category="corrosion", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text=WANG2018_ONSET, verification="verbatim-full-text",
        notes="Al is the plated metal: at the NEGATIVE electrode its stripping is the working "
              "reaction, so 'corrodes' only has meaning on the positive side."),
    row(substrate="Ti", mp_formula="Ti", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al (Al foil RE/CE)",
        corrosion_stability="oxidised", exp_potential_V_vs_Al=1.1, potential_type="onset",
        outcome_category="corrosion", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text=WANG2018_ONSET, verification="verbatim-full-text"),
    row(substrate="Au", mp_formula="Au", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al",
        corrosion_stability="etched", exp_potential_V_vs_Al=2.0, potential_type="corrodes_at",
        outcome_category="corrosion", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="Even the noble metal gold is rapidly etched at 2 V versus Al3+/Al.",
        verification="verbatim-full-text"),
    row(substrate="Pt", mp_formula="Pt", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al",
        corrosion_stability="not stable above 2 V", exp_potential_V_vs_Al=2.0,
        potential_type="corrodes_at", outcome_category="corrosion",
        citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="Even gold and platinum are not stable at potentials above 2 V.",
        verification="verbatim-full-text"),
    row(substrate="TiN", mp_formula="TiN", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al; AlCl3-graphite cells",
        corrosion_stability="stable until electrolyte oxidation", exp_potential_V_vs_Al=2.5,
        potential_type="stable_up_to", outcome_category="stable substrate",
        citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="The increase in the current above 2.5-3 V versus Al3+/Al for the TiN and "
                      "Cr2N current collectors is primarily due to the oxidation of the ionic "
                      "liquid. / The oxidative stability of TiN in AlCl3-[EMIM]Cl exceeds those of "
                      "Mo, W, and GC.",
        verification="verbatim-full-text",
        notes="Sputtered TiN film; limit set by the electrolyte, so the true onset is >= 2.5 V."),
    row(substrate="Cr2N", mp_formula="Cr2N", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al",
        corrosion_stability="stable until electrolyte oxidation", exp_potential_V_vs_Al=2.5,
        potential_type="stable_up_to", outcome_category="stable substrate",
        citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="The increase in the current above 2.5-3 V versus Al3+/Al for the TiN and "
                      "Cr2N current collectors is primarily due to the oxidation of the ionic liquid.",
        verification="verbatim-full-text"),
    row(substrate="W", mp_formula="W", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al",
        corrosion_stability="stable (known alternative), less than TiN",
        exp_potential_V_vs_Al="", potential_type="qualitative",
        outcome_category="stable substrate", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="TiN current collectors exhibit satisfactory corrosion resistance; in fact, "
                      "their corrosion resistance is even higher than those of known alternatives "
                      "(tungsten, molybdenum, and glassy carbon).",
        verification="verbatim-full-text", notes="No numeric onset for W in the verified text."),
    row(substrate="Mo", mp_formula="Mo", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al",
        corrosion_stability="stable (known alternative), less than TiN",
        exp_potential_V_vs_Al="", potential_type="qualitative",
        outcome_category="stable substrate", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="...even higher than those of known alternatives (tungsten, molybdenum, and "
                      "glassy carbon).", verification="verbatim-full-text"),
    row(substrate="glassy carbon", mp_formula="", electrode_side="positive current collector test",
        electrolyte=WANG2018_COND, alcl3_ratio="2.0", method="CV vs Al",
        corrosion_stability="stable (known alternative), less than TiN",
        exp_potential_V_vs_Al="", potential_type="qualitative",
        outcome_category="stable substrate", citation=WANG2018[0], doi=WANG2018[1],
        evidence_text="...even higher than those of known alternatives (tungsten, molybdenum, and "
                      "glassy carbon).", verification="verbatim-full-text",
        notes="Amorphous: no MP entry. Graphite (MP) is only a proxy."),
    # ---- Oh, Lee, Tak 2018 ---------------------------------------------------------
    row(substrate="Ni", mp_formula="Ni", electrode_side="positive current collector test",
        electrolyte="AlCl3-[EMIm]Cl (acidic)", method="cycling of graphite cathodes on the collector",
        corrosion_stability="electrochemically and chemically unstable",
        exp_potential_V_vs_Al="", potential_type="qualitative", outcome_category="corrosion",
        citation=OH2018[0], doi=OH2018[1],
        evidence_text="...confirming that nickel metal is electrochemically and chemically unstable "
                      "in the expanded operating potential window, thus leading to the failure of "
                      "cell operation and maintenance.", verification="verbatim-abstract",
        notes="Upper potential of the 'expanded window' not in the abstract."),
    row(substrate="Mo", mp_formula="Mo", electrode_side="positive current collector test",
        electrolyte="AlCl3-[EMIm]Cl (acidic)", method="cycling of graphite cathodes on the collector",
        corrosion_stability="stable", exp_potential_V_vs_Al="", potential_type="qualitative",
        outcome_category="stable substrate", cycling_or_CE="~85 mAh/g at 4000 mA/g (graphite cathode)",
        citation=OH2018[0], doi=OH2018[1],
        evidence_text="The molybdenum current collector exhibits fairly stable and reversible redox "
                      "cycling in the expanded potential window without any side effects or reactions",
        verification="verbatim-abstract"),
    # ---- Shi, Zhang, Guo 2019 ----------------------------------------------------------
    row(substrate="stainless steel", mp_formula="", electrode_side="positive current collector test",
        electrolyte="AlCl3-[EMIm]Cl", alcl3_ratio="1.3",
        method="galvanostatic reduction/oxidation of bare collectors vs Al",
        corrosion_stability="corrodes; corrosion current can be mistaken for battery capacity",
        exp_potential_V_vs_Al="", potential_type="qualitative", outcome_category="corrosion",
        citation=SHI2019[0] + "; original observation " + REED2013[0], doi=SHI2019[1],
        evidence_text="most of the studied metals can be (electrochemically) corroded in deep "
                      "eutectic chloroaluminate electrolytes, and the corrosive current can be "
                      "mistaken as the current from battery reactions, as demonstrated by Reed and "
                      "Menke in the case of stainless steel.",
        verification="verbatim-abstract",
        notes="Per-material results of Shi et al. (their Fig. 2) not accessed: only a search "
              "summary stating that GC and Mo behaved differently from the other collectors."),
    # ---- Negative-electrode (Al deposition) observations -------------------------------
    row(substrate="V", mp_formula="V", electrode_side="negative (Al deposition)",
        electrolyte="equimolar AlCl3 + NaCl melt (not an ionic liquid)", alcl3_ratio="1.0",
        temperature="200 C", method="potentiostatic deposition; XRD/SEM/EDS",
        al_deposition="yes, with alloying", corrosion_stability=NR,
        exp_potential_V_vs_Al="", potential_type="alloying_observed",
        outcome_category="mixed/condition-dependent", citation=CVET2021[0], doi=CVET2021[1],
        evidence_text="intermetallic alloys were synthetized during aluminium underpotential "
                      "deposition onto vanadium ... The deposits were identified as Al3V and AlV3 "
                      "alloys.", verification="verbatim-abstract",
        notes="Different electrolyte and 200 C: tests the SIGN of the alloying tendency only."),
    row(substrate="Al (negative current collector)", mp_formula="", electrode_side="negative (Al deposition)",
        electrolyte="AlCl3-EMIMCl", method="Al-graphite dual-ion battery",
        al_deposition="yes (Al electroplating on the negative side)", potential_type="qualitative",
        outcome_category="reversible deposition", citation=KRAV2020[0], doi=KRAV2020[1],
        evidence_text="The basic configuration of an AGDIB contains a graphite cathode, AlCl3-EMIMCl "
                      "... anolyte, and metallic aluminum current collector ... Concurrently, the "
                      "aluminum electroplating reaction takes place on the negative side of AGDIBs.",
        verification="verbatim-abstract", exp_potential_V_vs_Al=""),
    row(substrate="carbon substrate with O-mediated Al bonding", mp_formula="",
        electrode_side="negative (Al deposition)", electrolyte="chloroaluminate ionic liquid (ratio not verified)",
        method="Al plating/stripping", al_deposition="yes, morphology controlled",
        outcome_category="reversible deposition",
        cycling_or_CE="reversibility 99.6-99.8%; >3,600 h (search summary)",
        potential_type="qualitative", exp_potential_V_vs_Al="",
        citation=ZHENG2021[0], doi=ZHENG2021[1],
        evidence_text="electrodes designed to promote strong oxygen-mediated chemical bonding "
                      "between Al deposits and the substrate enable fine control of deposition "
                      "morphology and provide exceptional reversibility (99.6-99.8%)",
        verification="search-summary",
        notes="Surface chemistry (Al-O-C bonding) is exactly the Al-nucleation descriptor the bulk "
              "screen lacks; motivates Phase-2 adsorption calculations."),
    row(substrate="various anodic current collectors (carbon and metal)", mp_formula="",
        electrode_side="negative (anode-free)", electrolyte="AlCl3/ionic liquid", method="anode-free full cells",
        al_deposition="yes (reversible plating/stripping)", potential_type="qualitative",
        outcome_category="mixed/condition-dependent", exp_potential_V_vs_Al="",
        citation=WANGL2021[0], doi=WANGL2021[1],
        evidence_text="various anodic current collectors ... can be stably charged/discharged ... "
                      "their anti-corrosion capabilities against Cl- ions greatly affect the battery "
                      "performances",
        verification="search-summary",
        notes="The per-material results (the source of the original Ni/Cu/Ag/Mg 'mixed' and "
              "graphite-paper 'works' labels) were NOT accessible; those labels are unverified."),
    row(substrate="Al foil (negative electrode of a conventional cell)", mp_formula="Al",
        electrode_side="negative (Al metal anode)",
        electrolyte="acidic chloroaluminate RTIL", method="review of the studies of Tak and co-workers",
        corrosion_stability="Al surface corrodes; corrosion increases with acidity",
        potential_type="qualitative", exp_potential_V_vs_Al="", outcome_category="corrosion",
        citation=KRAV2020B[0] + " (review; primary work: Choi et al. PCCP 2017 and Lee et al., Tak group)",
        doi=KRAV2020B[1],
        evidence_text="the research on the chemical stability of Al foil on the negative side of the "
                      "AGDIB in acidic RTILs was performed recently by Tak et al. It has been "
                      "demonstrated that Al surface corrodes in chloroaluminate melts. Furthermore, "
                      "the corrosion enhances significantly upon increasing their acidity.",
        verification="verbatim-full-text (secondary source: review)",
        notes="Motivates replacing the Al foil rather than the plated species; not scored, because "
              "the primary papers were not read."),
    # ---- Labels carried over from the original handoff with no verified source --------
    *[row(substrate=s, mp_formula=s, electrode_side="negative (anode-free), per original notes",
          potential_type="qualitative", exp_potential_V_vs_Al="",
          outcome_category="unknown", citation="none verified", doi="",
          evidence_text="original label: 'mixed - cycles, but performance differs by corrosion "
                        "resistance'", verification="none",
          notes="Kept to expose the gap; excluded from scoring.") for s in ("Cu", "Ag", "Mg")],
    row(substrate="graphite paper", mp_formula="", electrode_side="negative (anode-free), per original notes",
        potential_type="qualitative", exp_potential_V_vs_Al="", outcome_category="unknown",
        citation="none verified (probably " + WANGL2021[0] + ")", doi="",
        evidence_text="original label: 'works - cycles stably as anodic current collector'",
        verification="none", notes="Excluded from scoring until the source is read."),
    row(substrate="Fe", mp_formula="Fe", electrode_side="per original notes", potential_type="qualitative",
        exp_potential_V_vs_Al="", outcome_category="unknown", citation="none verified", doi="",
        evidence_text="original label: 'corrodes - oxidised in Al anolytes at high cell voltages'",
        verification="none", notes="Excluded from scoring."),
]


def main():
    df = pd.DataFrame(ROWS, columns=COLS)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} literature rows -> {OUT}")
    print(df.groupby("verification").size().to_string())


if __name__ == "__main__":
    main()
