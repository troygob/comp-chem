#!/usr/bin/env python3
"""
DFT Molecular Geometry & Energy Simulator (AB2 triatomics: H2O, H2S)
Stack: Python 3, PySCF (DFT engine), NumPy, SciPy, Matplotlib

Inputs  : molecule, XC functional, basis set, optional starting geometry
Process : Kohn-Sham SCF energy E(r, theta) -> minimized with Nelder-Mead
Outputs : optimized bond length / angle, total energy, dipole, HOMO-LUMO gap,
          comparison to experiment, runtime

Usage examples:
  python dft_sim.py --molecule H2O --functional b3lyp --basis 6-31g*
  python dft_sim.py --molecule H2S --functional pbe --basis cc-pvdz
  python dft_sim.py --sweep                 # functional x basis sweep + plots
"""
import argparse, time, csv, itertools
import numpy as np
from scipy.optimize import minimize
from pyscf import gto, dft

HARTREE_TO_EV = 27.2114
EXPT = {  # experimental reference (NIST CCCBDB)
    "H2O": dict(r=0.9572, theta=104.52, dipole=1.855, center="O", ligand="H"),
    "H2S": dict(r=1.3356, theta=92.12, dipole=0.978, center="S", ligand="H"),
}

def build(mol_name, r, theta_deg, basis):
    e = EXPT[mol_name]
    t = np.radians(theta_deg) / 2
    atom = f"""{e['center']} 0 0 0
{e['ligand']} {r*np.sin(t):.8f} {r*np.cos(t):.8f} 0
{e['ligand']} {-r*np.sin(t):.8f} {r*np.cos(t):.8f} 0"""
    return gto.M(atom=atom, basis=basis, verbose=0)

def scf_energy(mol_name, r, theta, functional, basis, return_mf=False):
    mol = build(mol_name, r, theta, basis)
    mf = dft.RKS(mol)
    mf.xc = functional
    mf.grids.level = 3
    e = mf.kernel()
    return (e, mf) if return_mf else e

def optimize(mol_name, functional, basis, r0=1.0, theta0=100.0):
    f = lambda x: scf_energy(mol_name, x[0], x[1], functional, basis)
    res = minimize(f, [r0, theta0], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-8, maxiter=80))
    return res.x[0], res.x[1], res.fun

def properties(mol_name, r, theta, functional, basis):
    e, mf = scf_energy(mol_name, r, theta, functional, basis, return_mf=True)
    dip = np.linalg.norm(mf.dip_moment(verbose=0))
    nocc = mf.mol.nelectron // 2
    gap = (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV
    return e, dip, gap

def run_one(mol_name, functional, basis, r0=1.0, theta0=100.0):
    t0 = time.time()
    r, th, _ = optimize(mol_name, functional, basis, r0, theta0)
    e, dip, gap = properties(mol_name, r, th, functional, basis)
    ex = EXPT[mol_name]
    return dict(molecule=mol_name, functional=functional, basis=basis,
                r_A=r, theta_deg=th, energy_Eh=e, dipole_D=dip, gap_eV=gap,
                err_r_pct=100 * (r - ex["r"]) / ex["r"],
                err_theta_deg=th - ex["theta"],
                err_dipole_pct=100 * (dip - ex["dipole"]) / ex["dipole"],
                time_s=time.time() - t0)

def show(d):
    ex = EXPT[d["molecule"]]
    print(f"\n=== {d['molecule']} | {d['functional']} / {d['basis']} ===")
    print(f"Bond length : {d['r_A']:.4f} A   (expt {ex['r']})  err {d['err_r_pct']:+.2f}%")
    print(f"Bond angle  : {d['theta_deg']:.2f} deg (expt {ex['theta']})  err {d['err_theta_deg']:+.2f} deg")
    print(f"Energy      : {d['energy_Eh']:.6f} Eh  ({d['energy_Eh']*HARTREE_TO_EV:.2f} eV)")
    print(f"Dipole      : {d['dipole_D']:.3f} D   (expt {ex['dipole']})  err {d['err_dipole_pct']:+.1f}%")
    print(f"HOMO-LUMO   : {d['gap_eV']:.2f} eV")
    print(f"Runtime     : {d['time_s']:.1f} s")

def sweep():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    funcs = ["lda,vwn", "pbe", "b3lyp"]
    bases = ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"]
    rows = []
    for fn, bs in itertools.product(funcs, bases):
        d = run_one("H2O", fn, bs)
        rows.append(d); print(f"done {fn:8s} {bs:8s} r={d['r_A']:.4f} th={d['theta_deg']:.2f} t={d['time_s']:.0f}s", flush=True)
    keys = list(rows[0].keys())
    with open("sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for fn in funcs:
        sub = [r for r in rows if r["functional"] == fn]
        x = [bases.index(r["basis"]) for r in sub]
        ax[0].plot(x, [r["r_A"] for r in sub], "o-", label=fn)
        ax[1].plot(x, [r["theta_deg"] for r in sub], "o-", label=fn)
        ax[2].plot(x, [r["time_s"] for r in sub], "o-", label=fn)
    ax[0].axhline(EXPT["H2O"]["r"], ls="--", c="k", label="experiment")
    ax[1].axhline(EXPT["H2O"]["theta"], ls="--", c="k", label="experiment")
    for a, t, y in zip(ax, ["O-H bond length", "H-O-H angle", "Cost (runtime)"], ["Angstrom", "degrees", "seconds"]):
        a.set_xticks(range(len(bases))); a.set_xticklabels(bases); a.set_title(t); a.set_ylabel(y); a.set_xlabel("basis set (small -> large)"); a.legend()
    plt.tight_layout(); plt.savefig("sweep_plot.png", dpi=130)

    # correlation: basis-function count / cost vs accuracy
    cost = np.array([r["time_s"] for r in rows]); err = np.abs([r["err_r_pct"] for r in rows])
    print("\nCorrelation (runtime vs |bond-length error|): %.2f" % np.corrcoef(cost, err)[0, 1])

def energy_surface(mol_name, functional, basis):
    """Potential-energy curve along the bond angle (r fixed at optimum)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, th, _ = optimize(mol_name, functional, basis)
    angles = np.linspace(70, 180, 12)
    E = [scf_energy(mol_name, r, a, functional, basis) for a in angles]
    E = (np.array(E) - min(E)) * 627.509
    plt.figure(figsize=(6, 4)); plt.plot(angles, E, "o-")
    plt.xlabel("H-X-H angle (deg)"); plt.ylabel("Relative energy (kcal/mol)")
    plt.title(f"{mol_name} bending curve ({functional}/{basis})"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("bending_curve.png", dpi=130)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", default="H2O", choices=list(EXPT))
    p.add_argument("--functional", default="b3lyp")
    p.add_argument("--basis", default="6-31g*")
    p.add_argument("--r0", type=float, default=1.0, help="starting bond length (A)")
    p.add_argument("--theta0", type=float, default=100.0, help="starting angle (deg)")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--curve", action="store_true", help="angle-bending energy curve")
    a = p.parse_args()
    if a.sweep: sweep()
    elif a.curve: energy_surface(a.molecule, a.functional, a.basis)
    else: show(run_one(a.molecule, a.functional, a.basis, a.r0, a.theta0))
