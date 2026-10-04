#!/usr/bin/env python3
"""过渡态搜索 (CI-NEB) 验证 — 服务器运行。

判据:
  A. 解析面自洽: H + H₂ (LEPS) — NEB 势垒 vs 沿 MEP 解析扫描的势垒
  B. 真实从头算: H₃ 交换反应 CCSD/cc-pVTZ — 势垒 vs 文献 9.6 kcal/mol (0.416 eV)
  C. TS 验证: 鞍点处应恰有 1 个虚频 (数值 Hessian)
  D. TS 几何: 线性对称 (R_HH 相等) 与 D∞h 对称性

用法 (服务器/Slurm): python -u scripts/validate_transition_state.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator, AnalyticCalculator
from autoquantum.pes.neb import neb_path, imaginary_mode_count
from autoquantum.pes.optimize import harmonic_frequencies

BOHR = 0.529177210903
HA_EV = 27.211386245988
HA_KCAL = 627.5094740631


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def leps_h3_calculator():
    """共线 H + H₂ 的 LEPS 解析面 (沿用仓库教学参数), 3 原子沿 x 轴。"""
    from autoquantum.pes.leps import LEPSBuilder
    builder = LEPSBuilder()

    def energy(c):
        c = np.asarray(c, dtype=float)
        # 共线: 取原子 0-1-2 的一维坐标 (R = |x0-x1|, r = |x1-x2|)
        R = abs(c[0][0] - c[1][0])
        r = abs(c[1][0] - c[2][0])
        return float(np.ravel(builder.evaluate_2d(R, r))[0])

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

    return AnalyticCalculator(energy, grad, name="leps-h3")


def test_A_leps():
    sec("A. 解析面 (LEPS H+H₂): CI-NEB 势垒 vs MEP 解析扫描")
    calc = leps_h3_calculator()
    # 反应物: H_a + H_b–H_c (R=4.0, r=1.401); 产物: 交换后 (R=1.401, r=4.0)
    A = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.401, 0.0, 0.0]])
    B = np.array([[0.0, 0.0, 0.0], [1.401, 0.0, 0.0], [5.401, 0.0, 0.0]])
    # 沿 R = 1.5 r 的交换线扫描 (LEPS 的 MEP 近似) 作为解析参照
    from autoquantum.pes.leps import LEPSBuilder
    lb = LEPSBuilder()
    # 对称线 r_AB = r_BC: 鞍点是该线上的**最小**点 (垂直反应坐标方向),
    # 不是最大点 —— 早先取 max 得到 18 eV 的非物理参照。
    rr = np.linspace(1.2, 4.0, 1200)
    v = np.ravel(lb.evaluate_2d(rr, rr))
    e_react = float(np.ravel(lb.evaluate_2d(4.0, 1.401))[0])
    barrier_ref = float(v.min() - e_react) * HA_EV
    r_ts_ref = float(rr[int(np.argmin(v))])
    images, info = neb_path(calc, ["H", "H", "H"], A, B, n_images=9,
                            k_spring=0.08, climb=True, max_iter=800,
                            gtol=1e-2)
    ts = images[info["ts_index"]]
    r1 = abs(ts[0][0] - ts[1][0]); r2 = abs(ts[1][0] - ts[2][0])
    d = abs(info["barrier_eV"] - barrier_ref) / barrier_ref * 100
    good = d < 8.0
    print(f"  NEB 势垒 = {info['barrier_eV']:.4f} eV | 解析鞍点势垒 = "
          f"{barrier_ref:.4f} eV (对称线最小 @ r = {r_ts_ref:.3f} Bohr) | "
          f"差 {d:.2f}%  {'OK' if good else 'FAIL'}", flush=True)
    print(f"  TS 估计: R1 = {r1:.3f}, R2 = {r2:.3f} Bohr (交换对称应接近相等) | "
          f"收敛 {info['converged']}, {info['n_evals']} 次评估", flush=True)
    sym_ok = abs(r1 - r2) / max(r1, r2) < 0.05
    print(f"  对称性 |R1-R2|/R = {abs(r1-r2)/max(r1,r2)*100:.2f}% "
          f"{'OK' if sym_ok else 'FAIL'}", flush=True)
    return bool(good and sym_ok)


_H3_CACHE = {}


def _h3_neb(basis="cc-pvdz", method="ccsd"):
    """H₃ 交换反应 CI-NEB (缓存, 供 B/C/D 共用)。"""
    key = (basis, method)
    if key not in _H3_CACHE:
        calc = PySCFCalculator(["H", "H", "H"], basis=basis, method=method,
                               spin=1)
        A = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.41, 0.0, 0.0]])
        B = np.array([[0.0, 0.0, 0.0], [1.41, 0.0, 0.0], [6.41, 0.0, 0.0]])
        t0 = time.time()
        images, info = neb_path(calc, ["H", "H", "H"], A, B, n_images=7,
                                k_spring=0.08, climb=True, max_iter=500,
                                gtol=5e-3)
        _H3_CACHE[key] = (calc, images, info, time.time() - t0)
    return _H3_CACHE[key]


def test_B_h3_barrier():
    sec("B. 真实从头算 (H₃ 交换, CCSD/cc-pVDZ): CI-NEB 势垒")
    calc, images, info, dt_s = _h3_neb()
    bar_kcal = info["barrier_eV"] * 23.0605488
    ts = images[info["ts_index"]]
    r1 = abs(ts[0][0] - ts[1][0]); r2 = abs(ts[1][0] - ts[2][0])
    # 文献: H₃ 交换势垒 CCSD(T)/CBS = 9.6 kcal/mol (0.416 eV);
    # 小基组 (cc-pVDZ) 系统性高估 → 判据带 [9.6, 13.0]
    good = 9.6 <= bar_kcal <= 13.0
    print(f"  CI-NEB 势垒 = {info['barrier_eV']:.4f} eV = {bar_kcal:.3f} kcal/mol",
          flush=True)
    print(f"  文献 CCSD(T)/CBS = 9.60 kcal/mol; 小基组高估属预期 → 判据带 "
          f"[9.6, 13.0]  {'OK' if good else 'FAIL'}", flush=True)
    print(f"  TS: R1 = {r1*BOHR:.4f} A, R2 = {r2*BOHR:.4f} A | "
          f"收敛 {info['converged']}, {info['n_evals']} 次评估, {dt_s:.0f}s",
          flush=True)
    return bool(good)


def test_C_imaginary_mode():
    sec("C. TS 验证: 鞍点恰有 1 虚频 (正交补内对角化; 含量级判据)")
    calc, images, info, _ = _h3_neb()
    ts = images[info["ts_index"]]
    freqs, finfo = harmonic_frequencies(calc, ["H", "H", "H"], ts)
    n_imag = finfo["n_imag"]
    nu_imag = float(freqs[0]) if n_imag == 1 else float("nan")
    ok_ts = bool(n_imag == 1 and 500.0 < abs(nu_imag) < 3000.0)
    print(f"  TS: 虚频数 = {n_imag} (应为 1), 虚频 = {nu_imag:.1f} cm^-1 "
          f"(物理量级 500-3000)  {'OK' if ok_ts else 'FAIL'}", flush=True)
    print(f"  TS 全部频率: {' '.join(f'{x:.1f}' for x in freqs)}", flush=True)
    # 对照: 孤立 H₂ 分子 (束缚物种) 应 0 虚频
    from autoquantum.pes.optimize import optimize_geometry
    h2 = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="ccsd")
    opt, _ = optimize_geometry(h2, np.array([[0., 0., 0.], [0., 0., 1.4]]),
                               gtol=1e-5)
    f2, fi2 = harmonic_frequencies(h2, ["H", "H"], opt)
    print(f"  对照 H₂ (优化后): 频率 = {f2[0]:.1f} cm^-1, 虚频数 = {fi2['n_imag']}"
          f"  {'OK' if fi2['n_imag'] == 0 else 'FAIL'}", flush=True)
    return bool(ok_ts and fi2["n_imag"] == 0)


def test_D_ts_geometry():
    sec("D. TS 几何: 线性对称 D∞h + R_HH vs 文献 0.93 A")
    calc, images, info, _ = _h3_neb()
    ts = images[info["ts_index"]]
    r1 = abs(ts[0][0] - ts[1][0]); r2 = abs(ts[1][0] - ts[2][0])
    lin = float(np.abs(ts[:, 1:]).max())
    sym = abs(r1 - r2) / max(r1, r2)
    r_ref = 0.93 / BOHR
    d_r = abs(0.5 * (r1 + r2) - r_ref) / r_ref * 100
    good = bool(sym < 0.02 and lin < 1e-6 and d_r < 10.0)
    print(f"  R1 = {r1*BOHR:.4f} A, R2 = {r2*BOHR:.4f} A "
          f"(对称偏差 {sym*100:.3f}%)", flush=True)
    print(f"  线性残差 = {lin:.2e}; 平均 R_HH = {0.5*(r1+r2)*BOHR:.4f} A "
          f"vs 文献 0.93 A (差 {d_r:.2f}%)  {'OK' if good else 'FAIL'}",
          flush=True)
    return good


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    t0 = time.time()
    all_ok = True
    for fn in (test_A_leps, test_B_h3_barrier, test_C_imaginary_mode,
               test_D_ts_geometry):
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
