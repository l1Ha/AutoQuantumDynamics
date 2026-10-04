#!/usr/bin/env python3
"""诊断 PySCF CASSCF 解析梯度的正确用法与可用性。"""
import numpy as np
from pyscf import gto, scf, mcscf, grad

print("=" * 70)
print("[1] H2/cc-pVDZ CAS(2,2): 解析梯度 vs 有限差分")
print("=" * 70)
mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="cc-pvdz", unit="Bohr", verbose=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 2, 2); mc.kernel()
g_an = mc.nuc_grad_method().kernel()
print("  解析梯度:\n", np.array2string(np.asarray(g_an), precision=4))
h = 1e-4
def e_cas(coords):
    m = gto.M(atom=[("H", c) for c in coords], basis="cc-pvdz", unit="Bohr", verbose=0)
    r = scf.RHF(m).run(); x = mcscf.CASSCF(r, 2, 2); return x.kernel()[0]
c0 = np.array([[0.,0.,0.],[0.,0.,1.4]])
g_fd = np.zeros_like(c0)
for i in range(2):
    for j in range(3):
        cp, cm = c0.copy(), c0.copy(); cp[i,j]+=h; cm[i,j]-=h
        g_fd[i,j] = (e_cas(cp)-e_cas(cm))/(2*h)
print("  有限差分:\n", np.array2string(g_fd, precision=4))
print(f"  max|Δ| = {np.abs(np.asarray(g_an)-g_fd).max():.3e}")

print()
print("=" * 70)
print("[2] 不同梯度入口对比 (H2O/cc-pVDZ CAS(4,4))")
print("=" * 70)
from pyscf.grad import mcscf as grad_mcscf
print("  grad.mcscf 模块存在:", grad_mcscf is not None)
print("  可用 grad 模块:", [m for m in dir(grad) if 'mcscf' in m or 'casscf' in m])
mo = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
           basis="cc-pvdz", unit="Angstrom", verbose=0)
mf2 = scf.RHF(mo).run()
mc2 = mcscf.CASSCF(mf2, 4, 4); e2 = mc2.kernel()[0]
print(f"  CASSCF(4,4) E = {e2:.6f}")
try:
    g2 = mc2.nuc_grad_method()
    print("  nuc_grad_method 类型:", type(g2).__name__)
    k2 = g2.kernel()
    print("  梯度范数:", np.linalg.norm(k2), "max:", np.abs(k2).max())
except Exception as exc:
    print("  失败:", type(exc).__name__, exc)
# 与 RHF 梯度对比 (若 CASSCF 梯度退化为 RHF 梯度则二者相同)
g_rhf = mf2.nuc_grad_method().kernel()
print("  RHF 梯度范数:", np.linalg.norm(g_rhf), "max:", np.abs(g_rhf).max())
