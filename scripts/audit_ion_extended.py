#!/usr/bin/env python3
"""He+Li⁺ 离子出射通道 V⁺(R) 扩展范围 CCSD(T)+counterpoise 审计。

背景: 0.20.2/0.20.3 的离子曲线审计网格为 R ∈ [3.2, 12] bohr (19 点), 而参考
HeLip.txt (数字化样条) 覆盖 R ∈ [2.55, 15.05] bohr, 其中短程排斥壁
(R < 3.2 bohr, 参考值升至 +375 meV @ 2.55 bohr) 此前完全缺失 — 本脚本
将网格扩展为 R ∈ [2.3, 20] bohr 并在短程加密, 同时把长程尾补到 20 bohr
(超出参考数据范围的点仅作 −C₄/R⁴ 诱导尾检验, 不与参考样条比较)。

方法与约定与 target_1pct_audit.py 完全一致:
  V⁺(R) = E(HeLi⁺) − E_He(ghost Li) − E_Li⁺(ghost He)   (full counterpoise)
  基组默认 aug-cc-pVQZ; 4 电子体系 CCSD(T) 精确可控。

用法 (服务器):  python scripts/audit_ion_extended.py --basis aug-cc-pVQZ
用法 (本地报告): python scripts/audit_ion_extended.py --report <npz> --reference <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HA_EV = 27.211386245988
BOHR_ANG = 0.529177210903


def extended_grid() -> np.ndarray:
    """短程加密、长程补尾的扩展网格 (bohr)。"""
    seg = [
        np.arange(2.3, 5.0 + 1e-9, 0.1),    # 排斥壁 + 阱区 (用户指出 rmin 不足)
        np.arange(5.25, 8.0 + 1e-9, 0.25),
        np.arange(8.5, 12.0 + 1e-9, 0.5),
        np.arange(13.0, 20.0 + 1e-9, 1.0),
    ]
    return np.unique(np.round(np.concatenate(seg), 3))


def compute(basis: str, out: str, method: str = "ccsd(t)"):
    from pyscf import cc, gto, scf

    def e_run(atom: str, charge: int = 0) -> float:
        mol = gto.M(atom=atom, basis=basis, spin=0, charge=charge,
                    unit="Bohr", verbose=0)
        mf = scf.RHF(mol)
        e = mf.kernel()
        if method == "scf":
            return float(e)
        mycc = cc.CCSD(mf)
        e_corr = mycc.kernel()[0]
        return float(e + e_corr + mycc.ccsd_t())

    rg = extended_grid()
    print(f"[ion-extended] basis={basis} method={method} | 网格 {len(rg)} 点, "
          f"R ∈ [{rg[0]}, {rg[-1]}] bohr (短程加密至 ΔR=0.1)")
    e_he = e_run("He 0 0 0")
    e_lip = e_run("Li 0 0 0", charge=1)
    v_unc = np.zeros(len(rg))
    v_cp = np.zeros(len(rg))
    t0 = time.time()
    for i, R in enumerate(rg):
        e_sup = e_run(f"He 0 0 0; Li 0 0 {R}", charge=1)
        e_he_g = e_run(f"He 0 0 0; ghost:Li 0 0 {R}", charge=0)
        e_lip_g = e_run(f"ghost:He 0 0 0; Li 0 0 {R}", charge=1)
        v_unc[i] = (e_sup - e_he - e_lip) * HA_EV * 1e3
        v_cp[i] = (e_sup - e_he_g - e_lip_g) * HA_EV * 1e3
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(rg)}] R={R:5.2f}: "
                  f"V_unc={v_unc[i]:+9.2f} | V_CP={v_cp[i]:+9.2f} meV "
                  f"({time.time()-t0:.0f}s)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, r_grid=rg, v_unc=v_unc, v_cp=v_cp,
                        basis=basis, method=method,
                        provenance=json.dumps(dict(
                            note="extended range [2.3,20] bohr, short-range dense",
                            cp="full counterpoise"), ensure_ascii=False))
    print(f"  saved → {out}  (耗时 {time.time()-t0:.0f}s)")


def report(npz: str, ref_dir: str | None):
    d = np.load(npz, allow_pickle=True)
    rg, v_cp = d["r_grid"], d["v_cp"]
    print(f"[report] {npz}: {d['basis']} / {d['method']}, {len(rg)} 点, "
          f"R ∈ [{rg[0]}, {rg[-1]}] bohr")
    i_min = int(np.argmin(v_cp))
    print(f"  CP 阱深 = {(-v_cp[i_min]):.2f} meV @ R = {rg[i_min]:.2f} bohr "
          f"({rg[i_min]*BOHR_ANG:.3f} Å)")
    if not ref_dir:
        return
    import sys
    sys.path.insert(0, ref_dir)
    from src.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
    from src.potentials import IonicPotential

    ref = IonicPotential()
    # 参考数据有效范围 (样条在此区间外 clip)
    r_lo, r_hi = ref.rmin_bohr, ref.rmax_bohr
    m = (rg >= r_lo) & (rg <= r_hi)
    vref = np.array([ref(float(x)) for x in rg]) * 1e3 * HARTREE_TO_EV
    # 参考阱深与其位置
    j_min = int(np.argmin(vref))
    print(f"  参考阱深 = {(-vref[j_min]):.2f} meV @ R = {rg[j_min]:.2f} bohr "
          f"(样条; 数据范围 [{r_lo:.2f}, {r_hi:.2f}] bohr)")
    bands = [("排斥壁", 2.3, 3.2), ("内阱", 3.2, 4.5), ("阱谷", 4.5, 6.0),
             ("尾部", 6.0, r_hi)]
    print(f"  {'区段':>6} {'n':>3} {'MAE(meV)':>10} {'max|Δ|':>9} {'%阱深':>7}")
    for name, lo, hi in bands:
        k = m & (rg >= lo) & (rg <= hi)
        if not k.any():
            continue
        dm = v_cp[k] - vref[k]
        print(f"  {name:>6} {int(k.sum()):3d} {np.abs(dm).mean():10.3f} "
              f"{np.abs(dm).max():9.3f} "
              f"{np.abs(dm).mean()/87*100:6.2f}%")
    print(f"  {'逐点':>6}: R(bohr) 参考 CP Δ")
    step = max(1, int(np.ceil(m.sum() / 20)))
    idxs = np.where(m)[0][::step]
    for i in idxs:
        print(f"    {rg[i]:7.2f} {vref[i]:+9.2f} {v_cp[i]:+9.2f} "
              f"{v_cp[i]-vref[i]:+8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="aug-cc-pVQZ")
    ap.add_argument("--method", default="ccsd(t)", choices=["scf", "ccsd(t)"])
    ap.add_argument("--out", default="results/audit_ion_extended.npz")
    ap.add_argument("--report", metavar="NPZ", help="仅报告模式 (本地)")
    ap.add_argument("--reference", default=None,
                    help="Pro_HeLi_Enhanced 目录 (报告模式)")
    a = ap.parse_args()
    if a.report:
        report(a.report, a.reference)
    else:
        compute(a.basis, a.out, a.method)


if __name__ == "__main__":
    main()
