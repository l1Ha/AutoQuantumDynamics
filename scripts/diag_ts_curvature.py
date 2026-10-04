#!/usr/bin/env python3
"""诊断 H3 TS 的曲率: (1) 沿反对称坐标手查鞍点; (2) Hessian 步长敏感性。"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator
from autoquantum.pes.optimize import harmonic_frequencies
from autoquantum.pes.neb import neb_path

BOHR = 0.529177210903
calc = PySCFCalculator(["H", "H", "H"], basis="cc-pvdz", method="ccsd", spin=1)
A = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.41, 0.0, 0.0]])
B = np.array([[0.0, 0.0, 0.0], [1.41, 0.0, 0.0], [6.41, 0.0, 0.0]])
imgs, info = neb_path(calc, ["H"] * 3, A, B, n_images=7, k_spring=0.08,
                      max_iter=500, gtol=5e-3)
ts = imgs[info["ts_index"]].copy()
print(f"TS (NEB): R1={abs(ts[0][0]-ts[1][0])*BOHR:.4f} A, "
      f"R2={abs(ts[1][0]-ts[2][0])*BOHR:.4f} A, E={info['energies'][info['ts_index']]:.8f}")
e0 = calc.energy(ts)
print(f"\n[1] 沿反对称坐标 (R-d, R+d) 手查鞍点: E(TS)={e0:.8f}")
for d in (0.005, 0.02, 0.05):
    cp, cm = ts.copy(), ts.copy()
    cp[1, 0] += d; cm[1, 0] -= d      # 原子 1 沿 +x/-x → 反对称
    ep, em = calc.energy(cp), calc.energy(cm)
    tag = "鞍点 ✓" if (ep > e0 and em > e0) else ("谷底/极小 ✗" if (ep < e0 and em < e0) else "非对称")
    print(f"  d={d:.3f}: E(+d)-E0 = {(ep-e0)*1e6:+9.2f} uHa, "
          f"E(-d)-E0 = {(em-e0)*1e6:+9.2f} uHa → {tag}")
print("\n[2] Hessian 步长敏感性 (TS 处虚频数):")
for h in (1e-3, 3e-3, 5e-3, 1e-2):
    f, fi = harmonic_frequencies(calc, ["H"] * 3, ts, h=h)
    print(f"  h={h:.0e}: 虚频数={fi['n_imag']}, 频率(cmm1)="
          f"{' '.join(f'{x:9.1f}' for x in f)}")
print("\n[3] 反应物 (H + H2, R=5 Bohr) 频率:")
f2, fi2 = harmonic_frequencies(calc, ["H"] * 3, A, h=3e-3)
print(f"  虚频数={fi2['n_imag']}, 频率={[f'{x:.1f}' for x in f2]}")
