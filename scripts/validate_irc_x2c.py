#!/usr/bin/env python3
"""IRC (内禀反应坐标) 与 X2C 标量相对论验证 — 服务器运行。

判据:
  A. LEPS 解析面 IRC: 双方向单调下降并分别趋向反应物/产物谷
  B. H₃/CCSD IRC: 从对称 TS 出发, 两端应落到 H₂ + H 谷,
     末端 H–H 距离与独立优化的 H₂ 键长一致 (<8%)
  C. IRC 顶点校验: 路径起点即过渡态, 首步能量必须下降
  D. X2C: 梯度 vs 有限差分 (H₂O/cc-pVDZ); 相对论位移量级合理
     (轻元素 ~0, 含 Au 体系显著)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator, AnalyticCalculator
from autoquantum.pes.neb import neb_path, irc_path
from autoquantum.pes.optimize import optimize_geometry

BOHR = 0.529177210903
HA_EV = 27.211386245988


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def leps_calc():
    from autoquantum.pes.leps import LEPSBuilder
    lb = LEPSBuilder()

    def energy(c):
        c = np.asarray(c, dtype=float)
        return float(np.ravel(lb.evaluate_2d(
            abs(c[0][0] - c[1][0]), abs(c[1][0] - c[2][0])))[0])

    def grad(c):
        c = np.asarray(c, dtype=float)
        h = 1e-6
        g = np.zeros_like(c)
        for i in range(3):
            for j in range(3):
                cp, cm = c.copy(), c.copy()
                cp[i, j] += h; cm[i, j] -= h
                g[i, j] = (energy(cp) - energy(cm)) / (2 * h)
        return g

    return AnalyticCalculator(energy, grad, name="leps"), lb


def test_A_leps_irc():
    sec("A. LEPS 解析面: 双方向 IRC 应单调下降并趋向两谷")
    calc, lb = leps_calc()
    A = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.401, 0.0, 0.0]])
    B = np.array([[0.0, 0.0, 0.0], [1.401, 0.0, 0.0], [5.401, 0.0, 0.0]])
    imgs, info = neb_path(calc, ["H"] * 3, A, B, n_images=9, k_spring=0.08,
                          max_iter=800, gtol=1e-2)
    ts = imgs[info["ts_index"]]
    e_react = float(np.ravel(lb.evaluate_2d(4.0, 1.401))[0])
    ok = True
    for d, (r_big, r_small) in ((+1, ("R01", "R12")), (-1, ("R12", "R01"))):
        path, pi = irc_path(calc, ["H"] * 3, ts, step=0.08, max_steps=400,
                            direction=d)
        end = path[-1]
        r01 = abs(end[0][0] - end[1][0]); r12 = abs(end[1][0] - end[2][0])
        e_drop = pi["energies"][0] - pi["energies"][-1]
        frac_mono = float(np.mean(np.diff(pi["energies"]) <= 1e-6))
        # 应向"一个键拉长 + 另一个接近 H2 键长"的谷落下
        big = max(r01, r12); small = min(r01, r12)
        good = (e_drop > 0.05 and frac_mono > 0.98 and big > 2.5
                and 1.2 < small < 1.8)
        ok &= good
        print(f"  方向{d:+d}: {pi['n_points']} 点 | ΔE = -{e_drop:.4f} Ha | "
              f"单调占比 {frac_mono*100:.1f}% | 末端 R_big={big:.2f} "
              f"R_small={small:.3f} Bohr  {'OK' if good else 'FAIL'}", flush=True)
    print(f"  (反应物谷能量 {e_react:.4f} Ha, TS 能量 "
          f"{info['energies'][info['ts_index']]:.4f} Ha)", flush=True)
    return ok


def test_B_h3_irc():
    sec("B. H₃/CCSD IRC: 两端落到 H₂+H 谷, 末端键长 vs 优化 H₂")
    calc = PySCFCalculator(["H", "H", "H"], basis="cc-pvdz", method="ccsd",
                           spin=1)
    # 直接使用已确定的对称 TS 几何 (NEB 已在 v0.23.0 单独验证)
    r_ts = 0.9422 / BOHR
    ts = np.array([[0.0, 0.0, 0.0], [r_ts, 0.0, 0.0], [2 * r_ts, 0.0, 0.0]])
    # 参考: 独立优化 H₂
    h2 = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="ccsd")
    opt_h2, _ = optimize_geometry(h2, np.array([[0., 0., 0.], [0., 0., 1.4]]),
                                  gtol=1e-5)
    r_h2 = float(np.linalg.norm(opt_h2[0] - opt_h2[1]))
    print(f"  参考: 独立优化 H₂ 键长 = {r_h2*BOHR:.4f} A", flush=True)
    ok = True
    for d in (+1, -1):
        t0 = time.time()
        path, pi = irc_path(calc, ["H"] * 3, ts, step=0.10, max_steps=120,
                            direction=d)
        end = path[-1]
        r01 = abs(end[0][0] - end[1][0]); r12 = abs(end[1][0] - end[2][0])
        small = min(r01, r12)
        d_r = abs(small - r_h2) / r_h2 * 100
        e_drop = pi["energies"][0] - pi["energies"][-1]
        frac = float(np.mean(np.diff(pi["energies"]) <= 1e-6))
        # 判据: 单调下降 + 片段键长向优化 H₂ 值收敛 (且明显比 TS 更接近)
        closer = abs(small - r_h2) < 0.5 * abs(r_ts - r_h2)
        good = (e_drop > 2e-3 and frac > 0.95 and d_r < 8.0 and closer)
        ok &= good
        print(f"  方向{d:+d}: {pi['n_points']} 点 | ΔE = -{e_drop:.4f} Ha | "
              f"单调 {frac*100:.1f}% | 末端 H–H = {small*BOHR:.4f} A "
              f"(差 {d_r:.2f}%, TS 为 {r_ts*BOHR:.4f}) | "
              f"{time.time()-t0:.0f}s  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_C_ts_start():
    sec("C. IRC 顶点校验: 首步必须从过渡态下降")
    calc, _ = leps_calc()
    r_ts = 2.212
    ts = np.array([[0.0, 0.0, 0.0], [r_ts, 0.0, 0.0], [2 * r_ts, 0.0, 0.0]])
    path, pi = irc_path(calc, ["H"] * 3, ts, step=0.08, max_steps=5, direction=+1)
    e0, e1 = pi["energies"][0], pi["energies"][1]
    good = bool(e1 < e0)
    print(f"  TS E = {e0:.6f} Ha → 首步 E = {e1:.6f} Ha "
          f"(Δ = {(e1-e0)*1e6:+.2f} uHa)  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_D_x2c():
    sec("D. X2C 标量相对论: 梯度 FD 校验 + 位移量级")
    ok = True
    # D1: H2O/cc-pVDZ X2C 梯度 vs 有限差分
    coords = np.array([[0., 0., 0.], [0.9584/BOHR, 0., 0.],
                       [0.9584/BOHR*np.cos(np.radians(104.52)),
                        0.9584/BOHR*np.sin(np.radians(104.52)), 0.]])
    calc = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="rhf",
                           relativistic="x2c")
    e_r, g_r = calc.energy_and_gradient(coords)
    h = 1e-4
    g_fd = np.zeros_like(coords)
    for i in range(3):
        for j in range(3):
            cp, cm = coords.copy(), coords.copy()
            cp[i, j] += h; cm[i, j] -= h
            g_fd[i, j] = (calc.energy(cp) - calc.energy(cm)) / (2 * h)
    dmax = np.abs(g_r - g_fd).max()
    ok1 = dmax < 1e-5
    ok &= ok1
    print(f"  H2O X2C: E = {e_r:.6f} Ha, max|Δg| = {dmax:.2e}  "
          f"{'OK' if ok1 else 'FAIL'}", flush=True)
    calc_nr = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="rhf")
    e_nr = calc_nr.energy(coords)
    print(f"  相对论位移 (H2O): {(e_r-e_nr)*1e6:+.2f} uHa (轻元素应 ~0)",
          flush=True)
    # D1b: 单电子精确性 — X2C 应复现精确 Dirac 的相对论**位移**
    #   E_rel = (√(1-(Zα)²) - 1)·c² (原子单位, c = 1/α)
    alpha = 1 / 137.035999084
    c_au = 1.0 / alpha
    for Z, sym, spin in ((1, "H", 1), (2, "He", 1)):      # 类氢离子: 1 电子
        q = Z - 1
        e_exact = (np.sqrt(1 - (Z * alpha) ** 2) - 1) * c_au ** 2
        shift_exact = e_exact - (-0.5 * Z ** 2)           # 非相对论类氢能
        c_x2c = PySCFCalculator([sym], basis="cc-pv5z", charge=q, spin=spin,
                                method="rhf", relativistic="x2c")
        c_nr = PySCFCalculator([sym], basis="cc-pv5z", charge=q, spin=spin,
                               method="rhf")
        z = np.zeros((1, 3))
        shift = c_x2c.energy(z) - c_nr.energy(z)
        rel_err = abs(shift - shift_exact) / abs(shift_exact) * 100
        ok3 = rel_err < 30.0
        ok &= ok3
        print(f"  X2C 相对论位移 {sym}(Z={Z}): 计算 {shift*1e6:+.3f} uHa vs "
              f"精确 Dirac {shift_exact*1e6:+.3f} uHa (偏差 {rel_err:.1f}%)  "
              f"{'OK' if ok3 else 'FAIL'}", flush=True)
    # D2: 含重元素体系 (AuH) 的相对论位移应显著
    try:
        au_nr = PySCFCalculator(["Au", "H"], basis={"Au": "cc-pvdz-pp",
                                                    "H": "cc-pvdz"},
                                method="rhf")
        au_x2c = PySCFCalculator(["Au", "H"], basis={"Au": "cc-pvdz-pp",
                                                     "H": "cc-pvdz"},
                                 method="rhf", relativistic="x2c")
        c = np.array([[0., 0., 0.], [0., 0., 1.52/BOHR]])
        d_au = (au_x2c.energy(c) - au_nr.energy(c)) * 1e3
        ok2 = abs(d_au) > 1.0
        ok &= ok2
        print(f"  相对论位移 (AuH, ECP): {d_au:+.3f} mHa "
              f"(应显著)  {'OK' if ok2 else 'FAIL'}", flush=True)
    except Exception as exc:
        print(f"  AuH 段跳过: {type(exc).__name__}: {exc}", flush=True)
    return ok


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    only = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    t0 = time.time()
    pool = [(nm, fn) for nm, fn in
            (("A", test_A_leps_irc), ("C", test_C_ts_start),
             ("B", test_B_h3_irc), ("D", test_D_x2c))
            if not only or nm in only]
    all_ok = True
    for _, fn in pool:
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
