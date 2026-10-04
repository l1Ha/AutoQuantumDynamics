#!/usr/bin/env python3
"""ddCOSMO 隐式溶剂验证 — 服务器运行。

判据 (前四条是可严格成立的物理约束, 不依赖记忆中的文献值):
  A. ε→1 极限: 溶剂化能必须 → 0
  B. 介电单调性: ΔE_solv 随 ε 增大而单调更负
  C. 极性趋势: 极性溶质 (H₂O) 的溶剂化能显著大于非极性 (CH₄ hmm 用 He/Ne 类)
  D. 离子 Born 标度: 单价离子的 ΔE_solv ≈ −(1−1/ε)·q²/(2R) 量级 (比值 O(1))
  E. 已知量级: H₂O/6-31G* 在 ε=78.36 下 ≈ −5 ~ −10 kcal/mol
  F. 溶剂化梯度: 解析 vs 有限差分
  G. 方法路径覆盖: SCF / post-SCF(MP2) / TD-DFT 均能接入溶剂
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator

BOHR = 0.529177210903
KCAL = 627.5094740631
WATER = np.array([[0., 0., 0.], [0.9584/BOHR, 0., 0.],
                  [0.9584/BOHR*np.cos(np.radians(104.52)),
                   0.9584/BOHR*np.sin(np.radians(104.52)), 0.]])
H2O = (["O", "H", "H"], WATER)


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def e_solv(sym, coords, eps=None, basis="6-31g*", method="rhf", **kw):
    """返回 (E_gas, E_solv) — eps=None 即气相。"""
    common = dict(basis=basis, method=method, **kw)
    c = PySCFCalculator(sym, **common)
    e_gas = c.energy(coords)
    if eps is None:
        return e_gas, e_gas
    cs = PySCFCalculator(sym, solvent_eps=eps, **common)
    return e_gas, cs.energy(coords)


def test_A_limit():
    sec("A. ε→1 极限: 溶剂化能必须 → 0 (10⁻⁵ Ha 量级)")
    sym, c = H2O
    e_gas, e_1 = e_solv(sym, c, eps=1.0)
    d = abs(e_1 - e_gas) * KCAL
    good = d < 0.01
    print(f"  ε=1: ΔE_solv = {d:.6f} kcal/mol  "
          f"{'OK' if good else 'FAIL'}", flush=True)
    return good


def test_B_monotone():
    sec("B. 介电单调性: ΔE_solv 随 ε 增大单调更负")
    sym, c = H2O
    eps_list = [2.0, 10.0, 35.9, 78.3553]
    vals = []
    for eps in eps_list:
        _, e_s = e_solv(sym, c, eps=eps)
        e_g, _ = e_solv(sym, c)
        vals.append((e_s - e_g) * KCAL)
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    print("  ε:      " + "  ".join(f"{e:8.2f}" for e in eps_list), flush=True)
    print("  ΔE_solv:" + "  ".join(f"{v:8.3f}" for v in vals) +
          "  kcal/mol", flush=True)
    print(f"  严格单调递减: {mono}  {'OK' if mono else 'FAIL'}", flush=True)
    return mono


def test_C_polarity():
    sec("C. 极性趋势: 极性溶质溶剂化能 > (更负) 非极性溶质")
    res = {}
    for name, sym, c in (("H₂O", ) + H2O,
                         ("CH₄", ["C", "H", "H", "H", "H"],
                          np.array([[0., 0., 0.], [0.63, 0.63, 0.63],
                                    [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63],
                                    [-0.63, -0.63, 0.63]])),
                         ("He", ["He"], np.zeros((1, 3)))):
        e_g, e_s = e_solv(sym, c, eps=78.3553)
        res[name] = (e_s - e_g) * KCAL
        print(f"  {name:5s}: ΔE_solv = {res[name]:8.3f} kcal/mol", flush=True)
    # H2O 极性远大于 CH4/He; He 几乎为零
    good = bool(res["H₂O"] < res["CH₄"] < 0.1 and abs(res["He"]) < 0.05)
    print(f"  H₂O 更负且 He ≈ 0: {good}  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_D_ion_born():
    sec("D. 离子 Born 标度: ΔE_solv ≈ −(1−1/ε)·q²/(2R_eff)")
    # Li⁺ (1 电子体系, 半径取 vdW ~1.5 Bohr?) 用 ddCOSMO 自身给出能量,
    # 与 Born 模型对比只看量级 (比值 O(1) 即视为物理合理)
    sym = ["Li"]
    c = np.zeros((1, 3))
    eps = 78.3553
    e_g, e_s = e_solv(sym, c, eps=eps, basis="6-31g*",
                      method="rhf", charge=1)
    d = (e_s - e_g) * KCAL
    # Born: -(1-1/eps) * q^2/(2R) ; R 取 Li+ 的 vdW 半径 ~1.6 Å
    born = -(1 - 1/eps) * 332.0637 / (2 * 1.6)      # kcal/mol (原子单位制换算)
    ratio = d / born
    good = bool(d < 0 and 0.3 < ratio < 3.0)
    print(f"  Li⁺: ΔE_solv = {d:.2f} kcal/mol | Born 估计 (R=1.6 Å) = "
          f"{born:.2f} | 比值 {ratio:.2f} (应 O(1))", flush=True)
    print(f"  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_E_magnitude():
    sec("E. 已知量级: H₂O/6-31G* 在 ε=78.36 下的溶剂化能")
    sym, c = H2O
    e_g, e_s = e_solv(sym, c, eps=78.3553)
    d = (e_s - e_g) * KCAL
    good = -12.0 < d < -3.0
    print(f"  ΔE_solv = {d:.3f} kcal/mol (合理带 [-12, -3]; 实验 ΔG_hyd ≈ -6.3)",
          flush=True)
    print(f"  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_F_gradient():
    sec("F. 溶剂化解析梯度 vs 有限差分 (H₂O/6-31G*, ε=78.36)")
    sym, c = H2O
    calc = PySCFCalculator(sym, basis="6-31g*", method="rhf",
                           solvent_eps=78.3553)
    _, g = calc.energy_and_gradient(c)
    h = 1e-4
    g_fd = np.zeros_like(c)
    for i in range(3):
        for j in range(3):
            cp, cm = c.copy(), c.copy()
            cp[i, j] += h; cm[i, j] -= h
            g_fd[i, j] = (calc.energy(cp) - calc.energy(cm)) / (2 * h)
    dmax = np.abs(g - g_fd).max()
    good = dmax < 1e-5
    print(f"  max|Δg| = {dmax:.3e} Ha/Bohr  {'OK' if good else 'FAIL'}",
          flush=True)
    return good


def test_G_paths():
    sec("G. 方法路径覆盖: post-SCF(MP2) 与 TD-DFT 都能接入溶剂")
    sym, c = H2O
    ok = True
    # MP2
    try:
        e_g, e_s = e_solv(sym, c, eps=78.3553, method="mp2")
        d = (e_s - e_g) * KCAL
        good = d < 0
        ok &= good
        print(f"  MP2/6-31G*: ΔE_solv = {d:.3f} kcal/mol  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    except Exception as exc:
        print(f"  MP2 路径失败: {type(exc).__name__}: {exc}", flush=True)
        ok = False
    # TD-DFT (气相 vs 溶剂化激发能位移)
    try:
        td_gas = PySCFCalculator(sym, basis="6-31g*", method="tddft",
                                 xc="b3lyp", nstates=3)
        td_sol = PySCFCalculator(sym, basis="6-31g*", method="tddft",
                                 xc="b3lyp", nstates=3, solvent_eps=78.3553)
        e1g = td_gas.excitation_spectrum(c)[0]
        e1s = td_sol.excitation_spectrum(c)[0]
        shift = e1s[0] - e1g[0]
        good = abs(shift) < 1.0        # 溶剂位移应在亚 eV 量级
        ok &= good
        print(f"  TD-DFT: 首激发 气相 {e1g[0]:.3f} eV → 溶剂 {e1s[0]:.3f} eV "
              f"(位移 {shift:+.3f} eV)  {'OK' if good else 'FAIL'}", flush=True)
    except Exception as exc:
        print(f"  TD-DFT 路径失败: {type(exc).__name__}: {exc}", flush=True)
        ok = False
    return ok


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    t0 = time.time()
    all_ok = True
    for fn in (test_A_limit, test_B_monotone, test_C_polarity, test_D_ion_born,
               test_E_magnitude, test_F_gradient, test_G_paths):
        try:
            all_ok &= bool(fn())
        except Exception as exc:
            print(f"  EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            all_ok = False
    print(flush=True)
    print("=" * 78, flush=True)
    print(f"{'全部通过' if all_ok else '存在失败项'}  (总耗时 {time.time()-t0:.0f}s)",
          flush=True)
    sys.exit(0 if all_ok else 1)
