#!/usr/bin/env python3
"""识别参考 HeLip.txt 的确切基组组合 (混合基组扫描)。"""
import numpy as np
from pyscf import gto, scf, cc
HA_EV = 27.211386245988
# 参考数据点 (eV→meV)
REF = {3.69: -85.908, 5.16: -29.444, 6.62: -9.683}

def e_ccsdt(basis, atom, charge=0):
    mol = gto.M(atom=atom, basis=basis, charge=charge, unit="Bohr", verbose=0)
    mf = scf.RHF(mol); e0 = mf.kernel()
    mycc = cc.CCSD(mf); ec = mycc.kernel()[0]
    return e0 + ec + mycc.ccsd_t()

CAND = [("cc-pVQZ", "cc-pVQZ"), ("cc-pVQZ", "cc-pVTZ"), ("cc-pVTZ", "cc-pVQZ"),
        ("cc-pV5Z", "cc-pVQZ"), ("cc-pVTZ", "cc-pVTZ"), ("cc-pV5Z", "cc-pVTZ")]
print(f"{'He 基组':<10} {'Li 基组':<10} " + " ".join(f"R={R:<5}" for R in REF))
for bhe, bli in CAND:
    B = {"He": bhe, "Li": bli}
    row = []
    for R, vref in REF.items():
        a = e_ccsdt(B, f"He 0 0 0; Li 0 0 {R}", 1)
        h = e_ccsdt(B, "He 0 0 0"); l = e_ccsdt(B, "Li 0 0 0", 1)
        v = (a - h - l) * HA_EV * 1e3
        row.append(f"{v:+8.3f}")
    print(f"{bhe:<10} {bli:<10} " + " ".join(row))
print(f"{'参考':<10} {'(目标)':<10} " + " ".join(f"{v:+8.3f}" for v in REF.values()))
