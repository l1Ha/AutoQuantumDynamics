#!/usr/bin/env python3
"""识别参考 HeLip.txt 的相关方法 (CCSD(T) / CCSD / MP2, cc-pVQZ 未校正)。"""
import numpy as np
from pyscf import gto, scf, cc, mp
HA_EV = 27.211386245988
REF = {3.69: -85.908, 5.16: -29.444, 6.62: -9.683}

def e_method(basis, atom, charge, method):
    mol = gto.M(atom=atom, basis=basis, charge=charge, unit="Bohr", verbose=0)
    mf = scf.RHF(mol); e0 = mf.kernel()
    if method == "ccsd(t)":
        mycc = cc.CCSD(mf); ec = mycc.kernel()[0]; return e0 + ec + mycc.ccsd_t()
    if method == "ccsd":
        mycc = cc.CCSD(mf); ec = mycc.kernel()[0]; return e0 + ec
    if method == "mp2":
        emp2, _ = mp.MP2(mf).kernel(); return emp2
    if method == "scf":
        return e0

print(f"{'方法':<10} " + " ".join(f"R={R:<6}" for R in REF) + "   | max|Δ| vs 参考")
for meth in ["ccsd(t)", "ccsd", "mp2", "scf"]:
    vals = []
    for R in REF:
        a = e_method("cc-pVQZ", f"He 0 0 0; Li 0 0 {R}", 1, meth)
        h = e_method("cc-pVQZ", "He 0 0 0", 0, meth)
        l = e_method("cc-pVQZ", "Li 0 0 0", 1, meth)
        vals.append((a - h - l) * HA_EV * 1e3)
    d = np.abs(np.array(vals) - np.array(list(REF.values())))
    print(f"{meth:<10} " + " ".join(f"{v:+9.3f}" for v in vals) + f"   | {d.max():.3f} meV")
print(f"{'参考':<10} " + " ".join(f"{v:+9.3f}" for v in REF.values()))
