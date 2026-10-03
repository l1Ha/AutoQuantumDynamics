#!/usr/bin/env python3
"""阱底单点高精度计算 (自建增广 cc-pV5Z 级基组) — 确定收敛阱深。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from large_basis_ion import make_basis, e_ccsdt, HA_EV

R = 3.69
B = make_basis()
t0 = time.time()
a, nao = e_ccsdt(B, f"He 0 0 0; Li 0 0 {R}", charge=1)
print(f"超分子 done (nao={nao}, {time.time()-t0:.0f}s)", flush=True)
h, _ = e_ccsdt(B, "He 0 0 0");      print(f"He done ({time.time()-t0:.0f}s)", flush=True)
l, _ = e_ccsdt(B, "Li 0 0 0", 1);   print(f"Li+ done ({time.time()-t0:.0f}s)", flush=True)
hg, _ = e_ccsdt(B, f"He 0 0 0; ghost:Li 0 0 {R}");  print(f"He(ghost) done ({time.time()-t0:.0f}s)", flush=True)
lg, _ = e_ccsdt(B, f"ghost:He 0 0 0; Li 0 0 {R}", 1); print(f"Li+(ghost) done ({time.time()-t0:.0f}s)", flush=True)
vu = (a - h - l) * HA_EV * 1e3
vc = (a - hg - lg) * HA_EV * 1e3
print(f"\n未校正阱深 = {-vu:.3f} meV | CP 阱深 = {-vc:.3f} meV | BSSE = {vu-vc:.3f} meV")
print("对照: 参考(原始) 85.91 | 参考(去BSSE) 75.50 | def2-QZVPPD-CP 75.50 | aVQZ-CP 78.36 | CBS(aVTZ,aVQZ) 81.96")
