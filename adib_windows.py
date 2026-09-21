"""
Electrolyte stability window of a substrate in a chloroaluminate melt.

The melt is modelled as a reservoir of Al and Cl tied through AlCl3. At electrode
potential phi (V vs Al/AlCl3):

    mu_Al(phi) = mu_Al0 - 3 e phi
    mu_Cl(phi) = (g_AlCl3 - mu_Al(phi)) / 3

For each substrate composition the grand-potential phase diagram open to Al and
Cl is built at every phi on a grid. The reaction driving force with the
reservoir is

    r(phi) = [Omega_hull(phi) - Omega_closed(phi)] / N_substrate_atoms   (eV/atom)

where Omega_closed is the grand potential of the composition's closed-system hull.
r = 0 means the reservoir cannot lower its energy (stable); r < 0 means a
reaction is favourable, and the products say which kind:

    products gain Cl               -> "chlorination"   (anodic corrosion)
    products lose Al, no Cl gained -> "Al extraction"  (anodic dealloying of an
                                                        Al-containing substrate)
    products gain Al               -> "alloying"       (intermetallic formation /
                                                        aluminothermic reduction)

The stable set of r(phi) = 0 is an interval (r is concave in phi), so:

phi_ox    = upper edge of the stability window: onset of the first ANODIC
            reaction (chlorination or Al extraction); ox_mechanism says which
phi_chlor = onset of chlorination specifically (compared with literature onsets)
phi_red   = highest phi at which the substrate still takes up Al (alloying)

phi_ox is a thermodynamic LOWER BOUND on the observed anodic limit: kinetics or
passive films can extend stability beyond it, but corrosion below it is not
expected. See docs/descriptors.md for every approximation.
"""

import numpy as np
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.core import Composition, Element

AL, CL = Element("Al"), Element("Cl")
PHI = np.round(np.arange(-1.5, 3.5 + 1e-9, 0.01), 2)   # V vs Al/AlCl3
TOL = 1e-3                                             # eV per substrate atom


def reservoir(pd_):
    """mu_Al0 (eV/atom) and g(AlCl3) (eV per formula unit) from one entry set."""
    mu_al0 = pd_.el_refs[AL].energy_per_atom
    g_alcl3 = pd_.get_hull_energy(Composition("AlCl3"))
    return mu_al0, g_alcl3


def chempots(phi, mu_al0, g_alcl3):
    mu_al = mu_al0 - 3.0 * phi
    return {AL: mu_al, CL: (g_alcl3 - mu_al) / 3.0}


def _nonopen(comp):
    return Composition({el: amt for el, amt in comp.items() if el not in (AL, CL)})


def _formula(e):
    return getattr(e, "original_entry", e).composition.reduced_formula


def _kind(dec, comp, nonopen):
    """Classify a reaction by the Al and Cl exchanged with the reservoir."""
    d_al = d_cl = 0.0
    for e, amt in dec.items():
        oc = getattr(e, "original_entry", e).composition
        n_no = sum(v for el, v in oc.items() if el not in (AL, CL))
        d_al += amt * oc[AL] / n_no
        d_cl += amt * oc[CL] / n_no
    d_al -= comp[AL] / nonopen.num_atoms
    d_cl -= comp[CL] / nonopen.num_atoms
    if d_cl > 1e-6:
        return "chlorination"
    if d_al < -1e-6:
        return "Al extraction"
    if d_al > 1e-6:
        return "alloying"
    return "other"


def _onset(k, phis, kinds_set):
    """Start of the final contiguous run of kinds_set reaching the top of the scan."""
    if not len(k) or k[-1] not in kinds_set:
        return None
    j = len(k) - 1
    while j - 1 >= 0 and k[j - 1] in kinds_set:
        j -= 1
    return j


def scan(entries, compositions, phis=PHI):
    """Stability windows for several compositions sharing one entry set.

    Returns {reduced_formula: dict} with phi_ox, phi_red, r at 0 V, the
    products just beyond each limit, and the full r(phi) curve.
    """
    pd_ = PhaseDiagram(entries)
    mu_al0, g_alcl3 = reservoir(pd_)
    # Only closed-hull phases can appear on the grand-potential hull, so the
    # open diagrams are built from the stable entries alone (exact, much faster).
    entries = list(pd_.stable_entries)
    comps = {Composition(c).reduced_formula: Composition(c) for c in compositions}
    closed = {f: pd_.get_hull_energy(c) for f, c in comps.items()}
    nonopen = {f: _nonopen(c) for f, c in comps.items()}

    curves = {f: np.zeros(len(phis)) for f in comps}
    kinds = {f: [""] * len(phis) for f in comps}
    prods = {f: [""] * len(phis) for f in comps}
    if not any(no.num_atoms for no in nonopen.values()):
        phis = phis[:0]                            # only Al itself: nothing to scan
    for i, phi in enumerate(phis):
        mu = chempots(phi, mu_al0, g_alcl3)
        gpd = GrandPotentialPhaseDiagram(entries, mu)
        for f, c in comps.items():
            no = nonopen[f]
            if no.num_atoms == 0:                  # Al itself: reservoir metal
                continue
            omega_closed = closed[f] - mu[AL] * c[AL] - mu[CL] * c[CL]
            r = (gpd.get_hull_energy(no) - omega_closed) / no.num_atoms
            curves[f][i] = r
            if r < -TOL:
                dec = gpd.get_decomposition(no)
                names = sorted({_formula(e) for e in dec})
                kinds[f][i] = _kind(dec, c, no)
                prods[f][i] = " + ".join(names)

    out = {}
    for f in comps:
        if nonopen[f].num_atoms == 0:
            out[f] = {"phi_ox": 0.0, "phi_chlor": 0.0, "ox_mechanism": "Al stripping",
                      "phi_red": 0.0, "r0": 0.0,
                      "ox_products": "n/a (plated metal)", "red_products": "",
                      "window": 0.0, "note": "Al is the reservoir metal; onset 0 V by definition"}
            continue
        k = np.array(kinds[f])
        red = np.flatnonzero(k == "alloying")
        j = _onset(k, phis, {"chlorination", "Al extraction"})
        if j is not None:
            phi_ox, ox_prod, ox_mech = float(phis[j]), prods[f][j], k[j]
            ox_cens = "=" if j > 0 else "<="
        else:
            phi_ox, ox_prod, ox_mech, ox_cens = float(phis[-1]), "", "", ">="
        jc = _onset(k, phis, {"chlorination"})
        phi_chlor = float(phis[jc]) if jc is not None else float(phis[-1])
        if len(red):
            j = red[-1]
            phi_red, red_prod = float(phis[j]), prods[f][j]
            red_cens = "=" if j < len(phis) - 1 else ">="
        else:
            phi_red, red_prod, red_cens = float(phis[0]), "", "<="
        i0 = int(np.argmin(np.abs(phis)))
        out[f] = {
            "phi_ox": phi_ox, "phi_ox_censor": ox_cens, "ox_products": ox_prod,
            "ox_mechanism": ox_mech, "phi_chlor": phi_chlor,
            "phi_red": phi_red, "phi_red_censor": red_cens, "red_products": red_prod,
            "r0": float(curves[f][i0]), "kind0": k[i0] or "stable",
            "products0": prods[f][i0],
            "window": max(0.0, phi_ox - max(phi_red, 0.0)),
            "curve": curves[f],
        }
    return out
