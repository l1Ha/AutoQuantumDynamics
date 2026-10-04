#!/usr/bin/env python3
"""CASSCF 多参考验证 — 服务器运行。

判据:
  A. H₂ 解离曲线: CASSCF(2,2) 在拉伸区应显著低于 RHF 并接近 FCI
     (这正是多参考的价值: RHF 在 R=3-6 Bohr 严重高估)
  B. 与 FCI 的差应为"动力学相关"量级 (H₂/cc-pVDZ 约 0.01-0.02 Ha)
  C. CASSCF 解析梯度 vs 有限差分
  D. LiH CASSCF(2,2): 键断裂区 vs FCI (LiH 有 4 电子, 检验非最小体系)
  E. 自然轨道占据数: 拉伸区应出现显著的部分占据 (多参考特征)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator

BOHR = 0.529177210903
HA = 1.0


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def h2_rhf_casscf_fci(R_bohr):
    """返回 (E_RHF, E_CASSCF22, E_FCI) for H2/cc-pVDZ。"""
    from pyscf import gto, scf, mcscf, fci, ao2mo
    mol = gto.M(atom=[("H", [0, 0, 0]), ("H", [0, 0, R_bohr])],
                basis="cc-pvdz", unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    e_rhf = mf.e_tot
    mc = mcscf.CASSCF(mf, 2, 2)
    e_cas = mc.kernel()[0]
    h1 = mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
    e_fci = fci.direct_spin1.FCI().kernel(h1, eri, mol.nao, (1, 1))[0] + mol.energy_nuc()
    # 自然轨道占据数 (拉伸区应有部分占据)
    occ = np.linalg.eigvalsh(mc.make_rdm1())
    return float(e_rhf), float(e_cas), float(e_fci), occ


def test_A_dissociation():
    sec("A. H₂/cc-pVDZ 解离曲线: CASSCF(2,2) vs RHF vs FCI")
    print(f"  {'R(Bohr)':>8} {'RHF':>13} {'CASSCF(2,2)':>14} {'FCI':>13} "
          f"{'RHF-FCI(mHa)':>13} {'CAS-FCI(mHa)':>13}", flush=True)
    ok = True
    rows = []
    for R in (1.4, 2.0, 2.5, 3.0, 4.0, 5.0):
        e_r, e_c, e_f, occ = h2_rhf_casscf_fci(R)
        rows.append((R, e_r, e_c, e_f, occ))
        print(f"  {R:8.2f} {e_r:13.6f} {e_c:14.6f} {e_f:13.6f} "
              f"{(e_r-e_f)*1e3:13.2f} {(e_c-e_f)*1e3:13.2f}", flush=True)
    # 判据: 拉伸区 CASSCF 明显优于 RHF (且接近 FCI)
    R4 = [r for r in rows if r[0] == 4.0][0]
    ok &= abs(R4[2] - R4[3]) * 1e3 < 25.0          # |CASSCF-FCI| < 25 mHa
    ok &= abs(R4[1] - R4[3]) * 1e3 > 40.0          # |RHF-FCI| > 40 mHa
    print(f"  R=4.0 Bohr: |RHF-FCI| = {abs(R4[1]-R4[3])*1e3:.1f} mHa → "
          f"|CASSCF-FCI| = {abs(R4[2]-R4[3])*1e3:.1f} mHa  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    return ok, rows


def test_C_gradient():
    """CASSCF 梯度能力: 用 FD 梯度优化 H₂, 并与解析 (CASCI 型, 对 H₂ 精确)
    梯度优化交叉验证 — 两条路径必须给出同一极小点几何与真极小。"""
    sec("C. CASSCF 梯度: H₂/cc-pVDZ 优化 (FD 梯度) vs 解析梯度交叉验证")
    from autoquantum.pes.optimize import optimize_geometry
    c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])
    # (1) FD 梯度 (默认, 正确性优先)
    calc_fd = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="casscf",
                              active_space=(2, 2))
    opt_fd, info_fd = optimize_geometry(calc_fd, c0, gtol=1e-4)
    r_fd = float(np.linalg.norm(opt_fd[0] - opt_fd[1]))
    # (2) 解析 (CASCI 型) 梯度 — H₂ 下与 CASSCF 梯度一致
    calc_an = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="casscf",
                              active_space=(2, 2), grad_t_mode="casci")
    opt_an, info_an = optimize_geometry(calc_an, c0, gtol=1e-5)
    r_an = float(np.linalg.norm(opt_an[0] - opt_an[1]))
    # 真极小检验
    e0 = calc_fd.energy(opt_fd)
    ok_min = True
    for i in range(2):
        for j in range(3):
            for sg in (+1, -1):
                d = opt_fd.copy()
                d[i, j] += sg * 0.05
                if calc_fd.energy(d) < e0:
                    ok_min = False
    d_r = abs(r_fd - r_an) / r_an * 100
    good = bool(info_fd["converged"] and abs(d_r) < 0.5 and ok_min)
    print(f"  FD 梯度优化: R = {r_fd*BOHR:.4f} A ({info_fd['n_evals']} 次评估)",
          flush=True)
    print(f"  解析梯度优化: R = {r_an*BOHR:.4f} A ({info_an['n_evals']} 次评估)",
          flush=True)
    print(f"  两条路径差 = {d_r:.3f}% (判据 <0.5%); 真极小 = {ok_min}  "
          f"{'OK' if good else 'FAIL'}", flush=True)
    return good


def test_D_lih():
    sec("D. LiH/cc-pVDZ CASSCF(2,2): 与 FCI 对比 (4 电子体系)")
    from pyscf import gto, scf, mcscf, fci, ao2mo
    ok = True
    for R in (3.0, 4.5):
        mol = gto.M(atom=[("Li", [0, 0, 0]), ("H", [0, 0, R])],
                    basis="cc-pvdz", unit="Bohr", verbose=0)
        mf = scf.RHF(mol).run()
        mc = mcscf.CASSCF(mf, 2, 2)
        e_cas = mc.kernel()[0]
        h1 = mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff
        eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
        e_fci = fci.direct_spin1.FCI().kernel(h1, eri, mol.nao, (2, 2))[0] + mol.energy_nuc()
        d_cf = (e_cas - e_fci) * 1e3
        d_rf = (mf.e_tot - e_fci) * 1e3
        good = d_cf < 25.0 and d_cf < d_rf
        ok &= good
        print(f"  R={R:.1f} Bohr: RHF-FCI = {d_rf:6.2f} mHa, "
              f"CASSCF(2,2)-FCI = {d_cf:6.2f} mHa  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_E_natural_occ():
    sec("E. 自然轨道占据数: 拉伸区应出现显著部分占据 (多参考特征)")
    ok = True
    for R in (1.4, 4.0):
        _, e_c, _, occ = h2_rhf_casscf_fci(R)
        occ_sorted = np.sort(occ)[::-1]
        n_partial = float(np.sum((occ_sorted > 0.02) & (occ_sorted < 1.98)))
        partial = [f"{x:.3f}" for x in occ_sorted[:4]]
        good = (R > 3 and n_partial >= 2)
        ok &= good
        print(f"  R={R:.1f} Bohr: 占据数前四位 = {partial} "
              f"(部分占据轨道数 = {n_partial})", flush=True)
    return ok


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    t0 = time.time()
    all_ok = True
    for fn in (test_A_dissociation, test_C_gradient, test_D_lih,
               test_E_natural_occ):
        try:
            res = fn()
            all_ok &= (res[0] if isinstance(res, tuple) else res)
        except Exception as exc:
            print(f"  EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            all_ok = False
    print(flush=True)
    print("=" * 78, flush=True)
    print(f"{'全部通过' if all_ok else '存在失败项'}  (总耗时 {time.time()-t0:.0f}s)",
          flush=True)
    sys.exit(0 if all_ok else 1)
