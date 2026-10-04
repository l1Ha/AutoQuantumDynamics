#!/usr/bin/env python3
"""诊断: (1) 参考值是否为冻结核数据; (2) CCSD(T) 梯度是否含 (T) 项。

运行 (服务器): python scripts/diag_corr_issues.py
"""
from __future__ import annotations
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyscf import gto, scf, cc, mp, fci, ao2mo

BOHR = 0.529177210903


def geo(name):
    if name == "H2O":
        r, a = 0.9584 / BOHR, np.radians(104.52)
        return [[0, 0, 0], [r, 0, 0], [r*np.cos(a), r*np.sin(a), 0]]
    if name == "N2":
        return [[0, 0, -0.54885/BOHR], [0, 0, 0.54885/BOHR]]
    if name == "H2":
        return [[0, 0, 0], [0, 0, 0.7414/BOHR]]
    raise KeyError


print("=" * 78)
print("[1] H2/cc-pVDZ: CCSD 是否等于 FCI (2 电子必须严格相等) → 判定实现正确性")
print("=" * 78)
mol = gto.M(atom=[(s, c) for s, c in zip(["H", "H"], geo("H2"))],
            basis="cc-pvdz", unit="Bohr", verbose=0)
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf); e_cc = mycc.kernel()[0]
h1 = mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff
eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
e_fci = fci.direct_spin1.FCI().kernel(h1, eri, mol.nao, (1, 1))[0] + mol.energy_nuc()
print(f"  RHF  = {mf.e_tot:.6f}")
print(f"  CCSD = {mf.e_tot + e_cc:.6f}  (E_corr = {e_cc*1e3:.3f} mHa)")
print(f"  FCI  = {e_fci:.6f}")
print(f"  |CCSD - FCI| = {abs(mf.e_tot + e_cc - e_fci)*1e3:.4f} mHa  "
      f"({'✓ 实现正确' if abs(mf.e_tot+e_cc-e_fci) < 1e-6 else '✗ 实现有问题'})")

print()
print("=" * 78)
print("[2] 参考值出处判定: 全电子 vs 冻结核 (MP2, cc-pVDZ)")
print("=" * 78)
print(f"{'体系':<6} {'全电子 (Ha)':>16} {'冻核 (Ha)':>16} {'差 (mHa)':>10}")
for name, sym in [("H2O", ["O","H","H"]), ("N2", ["N","N"]), ("H2", ["H","H"])]:
    m = gto.M(atom=[(s, c) for s, c in zip(sym, geo(name))], basis="cc-pvdz",
              unit="Bohr", verbose=0)
    mf = scf.RHF(m).run()
    e_all = mf.e_tot + mp.MP2(mf).kernel()[0]
    try:
        from pyscf.data import elements
        fr = elements.chemcore(m)
        e_fc = mf.e_tot + mp.MP2(mf, frozen=fr).kernel()[0]
        d = (e_all - e_fc) * 1e3
    except Exception:
        e_fc, d = float("nan"), float("nan")
    print(f"{name:<6} {e_all:16.6f} {e_fc:16.6f} {d:10.3f}")

print()
print("=" * 78)
print("[3] CCSD(T) 梯度: 是否自动包含 (T) 项?")
print("=" * 78)
m = gto.M(atom=[(s, c) for s, c in zip(["H","H"], geo("H2"))], basis="cc-pvdz",
          unit="Bohr", verbose=0)
mf = scf.RHF(m).run()
mycc = cc.CCSD(mf).run()
mycc.ccsd_t()
g_t = mycc.nuc_grad_method().kernel()
mycc2 = cc.CCSD(mf).run()
g_no_t = mycc2.nuc_grad_method().kernel()
print(f"  |g(CCSD(T)) - g(CCSD)| = {np.abs(g_t - g_no_t).max():.3e} "
      f"({'含 (T) 项' if np.abs(g_t-g_no_t).max() > 1e-6 else '⚠ 不含 (T) 项'})")
print(f"  (T) 能量 = {mycc.e_tot + mycc.ccsd_t() - mf.e_tot - mycc.kernel()[0]:.6f} Ha")
# 有限差分对照
h = 1e-4
def e_ccsdt(coords):
    mm = gto.M(atom=[(s, float(x), float(y), float(z)) for s, (x, y, z) in
                     zip(["H","H"], coords)], basis="cc-pvdz", unit="Bohr", verbose=0)
    mmf = scf.RHF(mm).run()
    c2 = cc.CCSD(mmf).run(); c2.ccsd_t()
    return mmf.e_tot + c2.e_corr + c2.ccsd_t()
c0 = np.array(geo("H2"), dtype=float)
g_fd = np.zeros_like(c0)
for i in range(2):
    for j in range(3):
        cp, cm = c0.copy(), c0.copy(); cp[i,j] += h; cm[i,j] -= h
        g_fd[i,j] = (e_ccsdt(cp) - e_ccsdt(cm)) / (2*h)
print(f"  |g(解析) - g(FD)| = {np.abs(g_t - g_fd).max():.3e} (H2 对称性下应≈0)")

# 用 H2O 做更严格的对照 (有梯度的非零分量)
m3 = gto.M(atom=[(s, c) for s, c in zip(["O","H","H"], geo("H2O"))],
           basis="cc-pvdz", unit="Bohr", verbose=0)
mf3 = scf.RHF(m3).run()
c3 = cc.CCSD(mf3).run(); c3.ccsd_t()
g3 = c3.nuc_grad_method().kernel()
c0w = np.array(geo("H2O"), dtype=float)
g3_fd = np.zeros_like(c0w)
for i in range(3):
    for j in range(3):
        cp, cm = c0w.copy(), c0w.copy(); cp[i,j] += h; cm[i,j] -= h
        def e3(cc_):
            mm = gto.M(atom=[(s, float(x), float(y), float(z)) for s,(x,y,z) in
                             zip(["O","H","H"], cc_)], basis="cc-pvdz",
                       unit="Bohr", verbose=0)
            mmf = scf.RHF(mm).run(); c = cc.CCSD(mmf).run(); c.ccsd_t()
            return mmf.e_tot + c.e_corr + c.ccsd_t()
        g3_fd[i,j] = (e3(cp) - e3(cm)) / (2*h)
print(f"  H2O: |g_an(CCSD(T)) - g_FD| = {np.abs(g3 - g3_fd).max():.3e}")
print(f"  H2O: |g_an(CCSD)    - g_FD| = "
      f"{np.abs(cc.CCSD(mf3).run().nuc_grad_method().kernel() - g3_fd).max():.3e}")
