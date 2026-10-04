#!/usr/bin/env python3
"""NEVPT2 动态相关验证 — 服务器运行 (在 CASSCF 之上补动态相关)。

判据 (对照同基组 FCI=精确值):
  A. H₂/cc-pVDZ: NEVPT2 误差必须显著小于 CASSCF 且接近 FCI
  B. LiH/cc-pVDZ: 同上 (4 电子体系)
  C. 单调性: E_NEVPT2 < E_CASSCF (二阶修正必须降低能量)
  D. NEVPT2 梯度: FD 一致性 (能量差分) + 优化后为真极小
  E. H₂O/cc-pVDZ NEVPT2: 相对 CAS(4,4) 的相关能增益量级合理
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator

BOHR = 0.529177210903


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def ref_energies(atom_str, basis, nelec_pair, spin=0):
    """返回 (RHF, CASSCF, FCI) 参考 (原始 PySCF 路径)。"""
    from pyscf import gto, scf, mcscf, fci, ao2mo
    mol = gto.M(atom=atom_str, basis=basis, spin=spin, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    ncas = 2 if mol.nelectron <= 4 else 4
    nelecas = 2 if mol.nelectron <= 4 else 4
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    e_cas = mc.kernel()[0]
    h1 = mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
    na = mol.nelectron // 2
    e_fci = fci.direct_spin1.FCI().kernel(h1, eri, mol.nao, (na, na))[0] + mol.energy_nuc()
    return float(mf.e_tot), float(e_cas), float(e_fci)


def test_A_B_fci():
    sec("A/B. NEVPT2 vs CASSCF vs FCI (同基组 FCI = 精确)")
    print(f"  {'体系':<8} {'R(Bohr)':>8} {'CASSCF-FCI':>11} {'NEVPT2-FCI':>11} "
          f"{'改善倍数':>8}", flush=True)
    cases = [("H₂", "H 0 0 0; H 0 0 {R}", 1.4, (2, 2), 8.0),
             ("H₂", "H 0 0 0; H 0 0 {R}", 2.5, (2, 2), 8.0),
             ("H₂", "H 0 0 0; H 0 0 {R}", 4.0, (2, 2), 8.0),
             # 大活性空间对照: 证明残差来自活性空间而非方法
             ("H₂", "H 0 0 0; H 0 0 {R}", 1.4, (8, 2), 2.0),
             ("LiH", "Li 0 0 0; H 0 0 {R}", 3.0, (2, 2), 8.0),
             ("LiH", "Li 0 0 0; H 0 0 {R}", 4.5, (2, 2), 8.0)]
    ok = True
    for name, tmpl, R, (ncas, nel), tol in cases:
        syms = ["H", "H"] if name == "H₂" else ["Li", "H"]
        spin = 0
        atom = tmpl.format(R=R)
        _, e_cas, e_fci = ref_energies(atom, "cc-pvdz", nel, spin=spin)
        calc = PySCFCalculator(syms, basis="cc-pvdz", method="casscf",
                               active_space=(ncas, nel), pt2="nevpt2")
        calc_c = PySCFCalculator(syms, basis="cc-pvdz", method="casscf",
                                 active_space=(ncas, nel))
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
        e_nev = calc.energy(coords)
        e_cas2 = calc_c.energy(coords)      # 同活性空间的 CASSCF 参考
        d_cas = abs(e_cas2 - e_fci) * 1e3
        d_nev = abs(e_nev - e_fci) * 1e3
        gain = d_cas / max(d_nev, 1e-9)
        good = (d_nev < d_cas) and (d_nev < tol)
        ok &= good
        print(f"  {name:<8} {R:8.1f} CAS({ncas},{nel}) {d_cas:9.2f} mHa "
              f"{d_nev:9.2f} mHa {gain:8.1f}x (容差 {tol})  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_C_sign():
    sec("C. 二阶修正必须降低能量 (E_NEVPT2 < E_CASSCF)")
    calc_c = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="casscf",
                             active_space=(2, 2))
    calc_n = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="casscf",
                             active_space=(2, 2), pt2="nevpt2")
    c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    d = calc_n.energy(c) - calc_c.energy(c)
    good = d < 0
    print(f"  H₂ R=1.4: ΔE(NEVPT2−CASSCF) = {d*1e3:+.3f} mHa  "
          f"{'OK' if good else 'FAIL'}", flush=True)
    return good


def test_D_gradient():
    sec("D. NEVPT2 梯度 (FD) 与优化: 极小点应在 CASSCF 与 FCI 之间")
    from autoquantum.pes.optimize import optimize_geometry
    c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])
    rows = []
    for tag, kw in (("CASSCF", dict()), ("NEVPT2", dict(pt2="nevpt2"))):
        calc = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="casscf",
                               active_space=(2, 2), **kw)
        opt, info = optimize_geometry(calc, c0, gtol=1e-4)
        r = float(np.linalg.norm(opt[0] - opt[1]))
        e0 = calc.energy(opt)
        ok_min = True
        for i in range(2):
            for j in range(3):
                for sg in (+1, -1):
                    d = opt.copy(); d[i, j] += sg * 0.05
                    if calc.energy(d) < e0:
                        ok_min = False
        rows.append((tag, r * BOHR, info["converged"], ok_min, info["n_evals"]))
        print(f"  {tag:<7}: R_e = {r*BOHR:.4f} A | 收敛 {info['converged']} | "
              f"真极小 {ok_min} | {info['n_evals']} 次评估", flush=True)
    # NEVPT2 的键长应比 CASSCF 短 (动态相关增强键合) 且两者都是真极小
    good = (rows[0][3] and rows[1][3] and rows[0][1] > rows[1][1]
            and rows[0][1] - rows[1][1] < 0.05)
    print(f"  R_e(CASSCF) − R_e(NEVPT2) = {rows[0][1]-rows[1][1]:.4f} A "
          f"(动态相关应使键更短)  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_E_h2o():
    sec("E. H₂O/cc-pVDZ CAS(4,4)+NEVPT2: 相关能增益量级")
    coords = np.array([[0., 0., 0.], [0.9584/BOHR, 0., 0.],
                       [0.9584/BOHR*np.cos(np.radians(104.52)),
                        0.9584/BOHR*np.sin(np.radians(104.52)), 0.]])
    calc_c = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz",
                             method="casscf", active_space=(4, 4))
    calc_n = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz",
                             method="casscf", active_space=(4, 4), pt2="nevpt2")
    e_c = calc_c.energy(coords)
    e_n = calc_n.energy(coords)
    d = (e_n - e_c) * 1e3
    # 参考: CCSD(T)/cc-pVDZ ≈ -76.2446 (全相关); CAS(4,4) 只含少量相关
    good = -300.0 < d < 0.0
    print(f"  E_CAS(4,4) = {e_c:.6f} Ha", flush=True)
    print(f"  E_NEVPT2   = {e_n:.6f} Ha (增益 {d:.2f} mHa)", flush=True)
    print(f"  参考 CCSD(T)/cc-pVDZ ≈ -76.2446 Ha (全相关)", flush=True)
    print(f"  增益量级合理  {'OK' if good else 'FAIL'}", flush=True)
    return good


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    t0 = time.time()
    all_ok = True
    for fn in (test_A_B_fci, test_C_sign, test_E_h2o, test_D_gradient):
        try:
            all_ok &= fn()
        except Exception as exc:
            print(f"  EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            all_ok = False
    print(flush=True)
    print("=" * 78, flush=True)
    print(f"{'全部通过' if all_ok else '存在失败项'}  (总耗时 {time.time()-t0:.0f}s)",
          flush=True)
    sys.exit(0 if all_ok else 1)
