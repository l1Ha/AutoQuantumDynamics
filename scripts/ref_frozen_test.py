#!/usr/bin/env python3
"""测试参考是否为冻结核 CCSD(T)/cc-pVQZ (Li 1s / He 1s 冻结)。"""
import numpy as np
from pyscf import gto, scf, cc
HA_EV = 27.211386245988
REF = {3.69: -85.908, 5.16: -29.444, 6.62: -9.683}

def e_frozen(basis, atom, charge, frozen):
    mol = gto.M(atom=atom, basis=basis, charge=charge, unit="Bohr", verbose=0)
    mf = scf.RHF(mol); e0 = mf.kernel()
    # 仅当冻结后仍剩占据轨道时才冻结 (He/Li⁺ 无核可冻)
    nocc = int((mf.mo_occ > 0).sum())
    fr = [f for f in (frozen or []) if f < nocc - 1]
    fr = fr or None
    mycc = cc.CCSD(mf, frozen=fr); ec = mycc.kernel()[0]
    return e0 + ec + mycc.ccsd_t()

for tag, frozen in [("全电子 (无冻结)", None), ("冻 Li 1s", [0]), ("冻全部核心", [0,1])]:
    vals = []
    for R in REF:
        a = e_frozen("cc-pVQZ", f"He 0 0 0; Li 0 0 {R}", 1, frozen)
        h = e_frozen("cc-pVQZ", "He 0 0 0", 0, frozen)
        l = e_frozen("cc-pVQZ", "Li 0 0 0", 1, frozen)
        vals.append((a - h - l) * HA_EV * 1e3)
    d = np.abs(np.array(vals) - np.array(list(REF.values())))
    print(f"{tag:<16} " + " ".join(f"{v:+9.3f}" for v in vals) +
          f"  | max|Δ| {d.max():.3f} meV")
print(f"{'参考':<16} " + " ".join(f"{v:+9.3f}" for v in REF.values()))
