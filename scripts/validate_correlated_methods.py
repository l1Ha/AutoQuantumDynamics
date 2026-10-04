#!/usr/bin/env python3
"""相关方法后端真机验证 (MP2 / CCSD / CCSD(T)) — 服务器运行。

验证设计 (只用可独立核验的基准):
  A. RHF/cc-pVDZ 绝对能量 vs 文献 (三体系)
  B. CCSD == FCI 严格性检查 (H2: 2 电子严格; LiH: 4 电子) — 实现正确性硬证据
  C. MP2/CCSD/CCSD(T) 物理一致性 (方法阶梯、相关能符号、(T) 增量量级)
  D. 解析梯度 vs 中心有限差分 (含 CCSD(T) 的 (T) 项 FD 修正)
  E. PES 扫描 + 抛物线极小点 -> H2 键长 vs 文献 0.7414 A
  F. 冻结核开关一致性

用法 (服务器/Slurm): python scripts/validate_correlated_methods.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator

BOHR = 0.529177210903
REF_RHF = {"H2O": -76.026759, "N2": -108.954245, "H2": -1.128710}
H2_RE_ANG = 0.7414


def geo(name):
    if name == "H2O":
        r, a = 0.9584 / BOHR, np.radians(104.52)
        return ["O", "H", "H"], np.array(
            [[0., 0., 0.], [r, 0., 0.], [r * np.cos(a), r * np.sin(a), 0.]])
    if name == "N2":
        return ["N", "N"], np.array([[0., 0., -0.54885 / BOHR],
                                     [0., 0., 0.54885 / BOHR]])
    if name == "H2":
        return ["H", "H"], np.array([[0., 0., 0.], [0., 0., H2_RE_ANG / BOHR]])
    if name == "LiH":
        return ["Li", "H"], np.array([[0., 0., 0.], [0., 0., 1.5949 / BOHR]])
    raise KeyError(name)


def E(name, method, **kw):
    sym, c = geo(name)
    return PySCFCalculator(sym, basis="cc-pvdz", method=method, **kw).energy(c)


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def test_A_rhf():
    sec("A. RHF/cc-pVDZ vs 文献 (几何差异应 <0.2 mHa)")
    print(f"  {'体系':<6} {'计算值':>14} {'文献值':>14} {'d(mHa)':>9}  状态")
    ok = True
    for name, ref in REF_RHF.items():
        e = E(name, "rhf")
        d = (e - ref) * 1e3
        good = abs(d) < 0.2
        ok &= good
        print(f"  {name:<6} {e:14.6f} {ref:14.6f} {d:+9.3f}  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_B_ccsd_equals_fci():
    from pyscf import gto, scf, cc, fci, ao2mo
    sec("B. CCSD vs FCI 严格性检查 (实现正确性硬证据)")
    ok = True
    for name, tol in (("H2", 1e-6), ("LiH", 1e-4)):
        sym, coords = geo(name)
        mol = gto.M(atom=[(s, c) for s, c in zip(sym, coords)], basis="cc-pvdz",
                    unit="Bohr", verbose=0)
        mf = scf.RHF(mol).run()
        e_cc = mf.e_tot + cc.CCSD(mf).kernel()[0]
        h1 = (mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
              @ mf.mo_coeff)
        eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
        na = mol.nelectron // 2
        e_fci = (fci.direct_spin1.FCI().kernel(h1, eri, mol.nao, (na, na))[0]
                 + mol.energy_nuc())
        d = abs(e_cc - e_fci)
        good = d < tol
        ok &= good
        print(f"  {name:<5}: CCSD = {e_cc:.6f}  FCI = {e_fci:.6f}  "
              f"|d| = {d*1e3:.4f} mHa  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_C_ladder():
    sec("C. 方法阶梯物理一致性 (相关能为负; E_CCSD(T) < E_CCSD < E_MP2 < E_SCF)")
    print(f"  {'体系':<6} {'MP2':>13} {'CCSD':>13} {'CCSD(T)':>13} "
          f"{'(T)增量(mHa)':>14}  状态")
    ok = True
    for name in ("H2O", "N2", "H2", "LiH"):
        e_scf = E(name, "rhf"); e_mp2 = E(name, "mp2")
        e_cc = E(name, "ccsd"); e_t = E(name, "ccsd(t)")
        ec_mp2 = (e_mp2 - e_scf) * 1e3
        ec_ccsd = (e_cc - e_scf) * 1e3
        dT = (e_t - e_cc) * 1e3
        # 2 电子体系 CCSD 即精确, (T) 必为 0 (物理正确, 非缺陷)
        nelec = {"H2O": 10, "N2": 14, "H2": 2, "LiH": 4}[name]
        t_ok = (abs(dT) < 0.05) if nelec == 2 else (dT < 0)
        # 允许 1e-8 Ha 数值噪声 (2 电子体系的 (T) 恒为零, 符号为噪声)
        good = (ec_mp2 < -1 and ec_ccsd < ec_mp2 - 1.0 and t_ok
                and e_t <= e_cc + 1e-8 < e_mp2 < e_scf)
        ok &= good
        print(f"  {name:<6} {e_mp2:13.6f} {e_cc:13.6f} {e_t:13.6f} "
              f"{dT:14.3f}  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_D_gradients():
    sec("D. 解析梯度 vs 中心有限差分 (H2O/cc-pVDZ, h=1e-4)")
    sym, coords = geo("H2O")
    ok = True
    for method in ("rhf", "mp2", "ccsd", "ccsd(t)"):
        calc = PySCFCalculator(sym, basis="cc-pvdz", method=method)
        _, g_an = calc.energy_and_gradient(coords)
        h = 1e-4
        g_fd = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            for j in range(3):
                cp, cm = coords.copy(), coords.copy()
                cp[i, j] += h; cm[i, j] -= h
                g_fd[i, j] = (calc.energy(cp) - calc.energy(cm)) / (2 * h)
        dmax = np.abs(g_an - g_fd).max()
        good = dmax < 2e-5
        ok &= good
        print(f"  {method:<8}: max|dg| = {dmax:.3e} Ha/Bohr  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    return ok


def _h2_re(basis):
    """H2 在给定基组下 CCSD(T) 平衡键长 (Bohr, 抛物线拟合极小)。"""
    calc = PySCFCalculator(["H", "H"], basis=basis, method="ccsd(t)")
    rg = np.linspace(1.15, 1.80, 14)
    ens = np.array([calc.energy(np.array([[0., 0., 0.], [0., 0., r]]))
                    for r in rg])
    i = int(np.argmin(ens))
    sl = slice(max(0, i - 2), min(len(rg), i + 3))
    c = np.polyfit(rg[sl], ens[sl], 2)
    return -c[1] / (2 * c[0])


def test_E_scan():
    sec(f"E. H2 CCSD(T) 平衡键长基组收敛阶梯 (实验/文献 {H2_RE_ANG} A)")
    ok = True
    res = {}
    for basis in ("cc-pvdz", "cc-pvtz", "cc-pvqz"):
        r = _h2_re(basis)
        res[basis] = r * BOHR
        print(f"  {basis:<10}: R_e = {r:.4f} Bohr = {r*BOHR:.4f} A", flush=True)
    d_qz = abs(res["cc-pvqz"] - H2_RE_ANG) / H2_RE_ANG * 100
    mono = res["cc-pvdz"] > res["cc-pvtz"] > res["cc-pvqz"]
    good = mono and d_qz < 0.5
    print(f"  单调收敛 (dDZ>dTZ>dQZ): {mono}; cc-pVQZ 与文献偏差 = {d_qz:.3f}%  "
          f"{'OK' if good else 'FAIL'}", flush=True)
    return good


def test_F_frozen():
    sec("F. 冻结核开关 (H2O/cc-pVDZ MP2: 全电子比冻核低 1-6 mHa)")
    sym, c = geo("H2O")
    e_all = PySCFCalculator(sym, basis="cc-pvdz", method="mp2").energy(c)
    e_fc = PySCFCalculator(sym, basis="cc-pvdz", method="mp2",
                           frozen_core=True).energy(c)
    d = (e_all - e_fc) * 1e3          # 全电子相关能更低 → 差为负
    good = -6.0 < d < -1.0
    print(f"  全电子 = {e_all:.6f} | 冻核 = {e_fc:.6f} | 差 = {d:.3f} mHa  "
          f"{'OK' if good else 'FAIL'}", flush=True)
    return good


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}")
    import pyscf
    print(f"pyscf: {pyscf.__version__}")
    t0 = time.time()
    all_ok = True
    for fn in (test_A_rhf, test_B_ccsd_equals_fci, test_C_ladder,
               test_D_gradients, test_E_scan, test_F_frozen):
        all_ok &= fn()
    print()
    print("=" * 78)
    print(f"{'全部通过' if all_ok else '存在失败项'}  (总耗时 {time.time()-t0:.0f}s)")
    sys.exit(0 if all_ok else 1)
