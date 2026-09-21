"""
Candidate space and Materials Project data layer for the ADIB substrate screen.

Everything retrieved from Materials Project is cached under data/ so the
screening, figures and notebook run offline and reproducibly once fetched:

    data/screening_summary.csv        one row per MP entry in the screening space
    data/entries/<Al-Cl-...>.json.gz  every MP entry in {substrate elements, Al, Cl},
                                      used for the electrolyte phase diagrams

Fetch (needs MP_API_KEY in the environment):
    python adib_data.py            # skips anything already cached
    python adib_data.py --refresh  # re-download everything

Materials Project data are DIRECTLY COMPUTED quantities (GGA/GGA+U/r2SCAN mixed
thermodynamics with MP2020 corrections, as served by the API). Nothing here is
estimated or filled in: missing fields stay empty.
"""

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
ENTRY_DIR = DATA / "entries"
SUMMARY_CSV = DATA / "screening_summary.csv"

# ---------------------------------------------------------------------------
# Screening space. Each family carries the reason it is a plausible anodic
# current collector / Al-plating substrate. Oxides, sulfides and phosphides
# appear in the Phase-1 ML training set (substrate_screening.py) only: they
# were added there to give grouped CV enough chemical systems, not because
# they are substrate candidates.
# ---------------------------------------------------------------------------
SCREENING_FAMILIES = {
    "element": {
        "systems": ["W", "Mo", "Ni", "Cu", "Ag", "Au", "Pt", "Ti", "Ta", "Nb",
                    "Cr", "Zr", "Hf", "V", "Fe", "Co", "Mg", "Al", "C"],
        "rationale": "Conventional current-collector metals and every single-element "
                     "reference material in the literature table (C = graphite, the "
                     "nearest MP proxy for graphite paper / glassy carbon).",
    },
    "nitride": {
        "systems": ["Ti-N", "Zr-N", "Hf-N", "V-N", "Nb-N", "Ta-N", "Cr-N", "Mo-N", "W-N"],
        "rationale": "Metallic transition-metal nitrides; TiN and Cr2N current "
                     "collectors were measured stable to 2.5-3 V vs Al in "
                     "AlCl3:[EMIM]Cl (Wang et al., Adv. Sci. 2018).",
    },
    "carbide": {
        "systems": ["Ti-C", "Zr-C", "Hf-C", "V-C", "Nb-C", "Ta-C", "Cr-C", "Mo-C", "W-C"],
        "rationale": "Metallic refractory carbides, chemically analogous to the "
                     "nitrides. Hypothesis: no chloroaluminate test found.",
    },
    "boride": {
        "systems": ["Ti-B", "Zr-B", "Hf-B", "V-B", "Nb-B", "Ta-B", "Cr-B", "Mo-B", "W-B"],
        "rationale": "Metallic transition-metal borides (e.g. TiB2, ZrB2), "
                     "conductive and chemically inert. Hypothesis: no "
                     "chloroaluminate test found.",
    },
    "silicide": {
        "systems": ["Ti-Si", "Zr-Si", "V-Si", "Nb-Si", "Ta-Si", "Cr-Si", "Mo-Si",
                    "W-Si", "Ni-Si"],
        "rationale": "Metallic transition-metal silicides (e.g. MoSi2, WSi2, TiSi2). "
                     "Hypothesis: no chloroaluminate test found.",
    },
    "aluminide": {
        "systems": ["Al-Ti", "Al-Zr", "Al-V", "Al-Nb", "Al-Ta", "Al-Cr", "Al-Mo",
                    "Al-W", "Al-Fe", "Al-Co", "Al-Ni", "Al-Cu"],
        "rationale": "Pre-aluminised substrates: an Al-rich aluminide in equilibrium "
                     "with Al cannot alloy further during plating. Hypothesis, "
                     "motivated by Al-V intermetallics forming during Al "
                     "underpotential deposition on V (Cvetkovic et al., Metals 2021).",
    },
}


def canonical(chemsys):
    """Materials Project chemsys form: elements sorted, dash-joined."""
    return "-".join(sorted(chemsys.split("-")))


SCREENING_SYSTEMS = [canonical(s) for f in SCREENING_FAMILIES.values() for s in f["systems"]]
FAMILY_OF = {canonical(s): fam for fam, f in SCREENING_FAMILIES.items() for s in f["systems"]}

SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "nsites", "density", "volume",
    "band_gap", "is_metal", "formation_energy_per_atom", "energy_above_hull",
    "theoretical", "symmetry", "weighted_work_function", "weighted_surface_energy",
    "database_IDs",
]


def api_key():
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY is not set and the data are not cached. "
                 "Set it (see README) or copy a populated data/ folder.")
    return key


# ---------------------------------------------------------------------------
def fetch_summary(refresh=False):
    """One row per MP entry in the screening systems; cached to CSV."""
    if SUMMARY_CSV.exists() and not refresh:
        return pd.read_csv(SUMMARY_CSV)
    from mp_api.client import MPRester

    rows = []
    with MPRester(api_key()) as mpr:
        for chemsys in SCREENING_SYSTEMS:
            docs = mpr.materials.summary.search(chemsys=chemsys, fields=SUMMARY_FIELDS)
            print(f"  {chemsys:<8} {len(docs):>4} entries")
            for d in docs:
                sym = d.symmetry
                icsd = (d.database_IDs or {}).get("icsd", []) if d.database_IDs else []
                rows.append({
                    "material_id": str(d.material_id),
                    "formula": d.formula_pretty,
                    "chemsys": d.chemsys,
                    "family": FAMILY_OF[canonical(d.chemsys)],
                    "nsites": d.nsites,
                    "density": d.density,
                    "vol_per_atom": d.volume / d.nsites if d.nsites else None,
                    "band_gap": d.band_gap,
                    "is_metal": d.is_metal,
                    "formation_energy_per_atom": d.formation_energy_per_atom,
                    "energy_above_hull": d.energy_above_hull,
                    "theoretical": d.theoretical,
                    "n_icsd": len(icsd),
                    "crystal_system": str(sym.crystal_system) if sym else None,
                    "spacegroup": sym.symbol if sym else None,
                    "work_function": d.weighted_work_function,
                    "surface_energy": d.weighted_surface_energy,
                })
    df = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"  {len(df)} entries -> {SUMMARY_CSV}")
    return df


# ---------------------------------------------------------------------------
def entry_key(chemsys):
    """Element set of the electrolyte phase diagram for one screening system."""
    return "-".join(sorted(set(chemsys.split("-")) | {"Al", "Cl"}))


def entry_path(key):
    return ENTRY_DIR / f"{key}.json.gz"


def load_entries(key):
    from monty.json import MontyDecoder
    with gzip.open(entry_path(key), "rt", encoding="utf-8") as fh:
        return json.load(fh, cls=MontyDecoder)


def _jsonable(obj):
    """Stringify non-JSON dict keys (MP puts Element keys in oxidation_states)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _slim(entry):
    """ComputedEntry with the same energies and MP2020 adjustments, no structure.

    The phase diagrams need composition, uncorrected energy and the energy
    adjustments (which carry the correction uncertainties); structures are
    dropped to keep the cache small.
    """
    from pymatgen.entries.computed_entries import ComputedEntry
    return ComputedEntry(entry.composition, entry.uncorrected_energy,
                         energy_adjustments=entry.energy_adjustments,
                         parameters=_jsonable(entry.parameters),
                         data=_jsonable(entry.data), entry_id=entry.entry_id)


def _fetch_one(key):
    from monty.json import MontyEncoder
    from mp_api.client import MPRester
    for attempt in range(3):
        try:
            t = time.time()
            with MPRester(api_key()) as mpr:
                entries = [_slim(e) for e in mpr.get_entries_in_chemsys(key.split("-"))]
            tmp = entry_path(key).with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(entries, fh, cls=MontyEncoder)
            tmp.replace(entry_path(key))
            return f"  {key:<14} {len(entries):>5} entries ({time.time() - t:.0f}s)"
        except Exception as exc:                      # network hiccup: retry
            err = exc
            time.sleep(5 * (attempt + 1))
    return f"  ! {key} failed: {err}"


def fetch_entries(refresh=False, workers=6):
    """Cache every MP entry in {substrate elements, Al, Cl} for each system.

    Processes, not threads: mp-api applies the GGA/r2SCAN mixing on the client,
    which is CPU-bound and would serialise on the GIL.
    """
    ENTRY_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({entry_key(s) for s in SCREENING_SYSTEMS})
    todo = [k for k in keys if refresh or not entry_path(k).exists()]
    print(f"  {len(keys)} electrolyte phase diagrams needed, {len(todo)} to fetch")
    with ProcessPoolExecutor(workers) as pool:
        for line in pool.map(_fetch_one, todo):
            print(line, flush=True)
    missing = [k for k in keys if not entry_path(k).exists()]
    if missing:
        print(f"  still missing: {missing}")
    return keys


THERMO_DIR = DATA / "thermo"


def thermo_path(key):
    return THERMO_DIR / f"{key}.csv"


def fetch_thermo(refresh=False):
    """Second, independent energy route for the method-uncertainty check.

    Server-side thermo documents (GGA_GGA+U_R2SCAN) for every sub-system of each
    {substrate, Al, Cl} space; only formation energies are kept (elements = 0),
    which is all a phase diagram needs. The server mixes GGA and r2SCAN per
    material's own chemical system, the client route mixes over the whole
    space, so chloride energies differ; the spread in phi_ox between the two
    routes is reported as a systematic uncertainty.
    """
    import itertools
    from mp_api.client import MPRester
    THERMO_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({entry_key(s) for s in SCREENING_SYSTEMS})
    todo = [k for k in keys if refresh or not thermo_path(k).exists()]
    print(f"  {len(todo)} thermo-document sets to fetch")
    with MPRester(api_key()) as mpr:
        for key in todo:
            els = key.split("-")
            subs = ["-".join(sorted(c)) for r in range(1, len(els) + 1)
                    for c in itertools.combinations(els, r)]
            docs = mpr.materials.thermo.search(
                chemsys=subs, thermo_types=["GGA_GGA+U_R2SCAN"],
                fields=["material_id", "composition", "formation_energy_per_atom",
                        "energy_type"])
            pd.DataFrame([{"material_id": str(d.material_id),
                           "composition": str(d.composition.formula),
                           "formation_energy_per_atom": d.formation_energy_per_atom,
                           "energy_type": d.energy_type} for d in docs]
                         ).to_csv(thermo_path(key), index=False)
            print(f"  thermo {key:<14} {len(docs):>5} docs", flush=True)


def load_thermo_entries(key):
    """PDEntries from cached thermo docs, formation-energy scale, elements at 0."""
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import Composition
    t = pd.read_csv(thermo_path(key))
    ents = [PDEntry(Composition(c), e * Composition(c).num_atoms, name=m)
            for c, e, m in zip(t["composition"], t["formation_energy_per_atom"], t["material_id"])]
    return ents + [PDEntry(Composition(el), 0.0, name=f"{el}-ref") for el in key.split("-")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    args = ap.parse_args()
    print("Screening space:", len(SCREENING_SYSTEMS), "chemical systems")
    for fam, f in SCREENING_FAMILIES.items():
        print(f"  {fam:<10} {len(f['systems']):>2}  {f['rationale']}")
    fetch_summary(args.refresh)
    fetch_thermo(args.refresh)
    fetch_entries(args.refresh)


if __name__ == "__main__":
    main()
