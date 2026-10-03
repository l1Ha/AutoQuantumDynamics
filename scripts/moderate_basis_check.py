#!/usr/bin/env python3
"""中等自建基组 (cc-pVQZ + 3 弥散/ℓ) 阱底交叉验证 — 独立确认收敛阱深。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from large_basis_ion import aug_like, e_ccsdt, HA_EV

R = 3.69
B = {"He": aug_like("He", base="cc-pVQZ", n_diffuse=3, ratio=2.8),
     "Li": aug_like("Li", base="cc-pVQZ", n_diffuse=3, ratio=2.8)}
t0 = time.time()
a, nao = e_ccsdt(B, f"He 0 0 0; Li 0 0 {R}", charge=1)
print(f"超分子 (nao={nao}) {time.time()-t0:.0f}s", flush=True)
h, _ = e_ccsdt(B, "He 0 0 0");                        print(f"He {time.time()-t0:.0f}s", flush=True)
l, _ = e_ccsdt(B, "Li 0 0 0", 1);                     print(f"Li+ {time.time()-t0:.0f}s", flush=True)
hg, _ = e_ccsdt(B, f"He 0 0 0; ghost:Li 0 0 {R}");    print(f"He(ghost) {time.time()-t0:.0f}s", flush=True)
lg, _ = e_ccsdt(B, f"ghost:He 0 0 0; Li 0 0 {R}", 1); print(f"Li+(ghost) {time.time()-t0:.0f}s", flush=True)
vu = (a - h - l) * HA_EV * 1e3
vc = (a - hg - lg) * HA_EV * 1e3
print(f"\n未校正 {-vu:.3f} meV | CP {-vc:.3f} meV | BSSE {vu-vc:.3f} meV")
print("对照: 参考(去BSSE) 75.50 | def2-QZVPPD-CP 75.50 | aVQZ-CP 78.36 | aVTZ-CP 73.42")
