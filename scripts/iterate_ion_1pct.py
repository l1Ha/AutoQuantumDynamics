#!/usr/bin/env python3
"""离子参考势能面 1% 最大误差迭代: 方法阶梯复现。

目标 (会话 goal): 与参考势能面最大误差 ≤ 1% (阱深 87 meV 的 1% = 0.87 meV)。
v0.20.3 已识别参考 HeLip.txt ≈ cc-pVQZ 级未校正 CCSD(T) (19 点 max 2.4 meV)。
本脚本在参考自身的 78 个数字化 R 点上运行方法阶梯, 逐配置报告
max/Δ| 与 RMS|Δ| (对样条和对原始点两种口径), 迭代至噪声地板。

方法假说 (广泛文献依据):
- W. Meyer 的 MRSCEP = 多参考 CI; 对 HeLi⁺ (4 电子) MRCI(SD) ≈ FCI
  → FCI/cc-pVQZ 与 FCI/cc-pVTZ 是"方法+基组完全复现"的检验;
- 未校正 (无 counterpoise) — 参考含 ~10 meV BSSE (v0.20.3);
- 基组阶梯: cc-pVQZ / cc-pVTZ / cc-pV5Z / def2-QZVPPD; CCSD vs CCSD(T)。

用法 (服务器):
  python scripts/iterate_ion_1pct.py --configs ccsdt_qz,fci_qz,fci_tz,ccsd_qz
  python scripts/iterate_ion_1pct.py --configs fci_qz --stride 3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HA_EV = 27.211386245988
BOHR_ANG = 0.529177210903
WELL_MEV = 86.75  # 参考样条阱深 (meV); 1% = 0.8675 meV

CONFIGS = {
    "ccsdt_qz": dict(basis="cc-pVQZ", method="ccsd(t)"),
    "ccsdt_tz": dict(basis="cc-pVTZ", method="ccsd(t)"),
    "ccsdt_5z": dict(basis="cc-pV5Z", method="ccsd(t)"),
    "ccsdt_dz": dict(basis="def2-QZVPPD", method="ccsd(t)"),
    "ccsd_qz": dict(basis="cc-pVQZ", method="ccsd"),
    "fci_qz": dict(basis="cc-pVQZ", method="fci"),
    "fci_tz": dict(basis="cc-pVTZ", method="fci"),
    "ccsdt_utmm": dict(basis="mm-uncontracted", method="ccsd(t)"),
}


def build_mm_uncontracted_basis() -> dict:
    """非收缩 cc-pVTZ (收缩误差消除) + Movre–Thiel–Meyer (2000) 第 III 节
    已发表弥散/价层指数 — 文献可溯源的基组变体, 检验其相关-BSSE 指纹
    是否更接近原作者大基组 (合法收敛步骤, 非对参考拟合)。"""
    from pyscf import gto

    def uncontract(el: str) -> list:
        base = gto.basis.load("cc-pVTZ", el)
        out = []
        for sh in base:
            l = sh[0]
            for block in sh[1:]:        # 每段 [exp, c1, c2, ...]
                out.append([l, [block[0], 1.0]])
        return out

    out = {el: uncontract(el) for el in ("He", "Li")}
    # MM (2000) 已发表弥散/价层指数 (去重后追加)
    mm = {"He": [[0, [0.04, 1.0]], [0, [0.016, 1.0]], [0, [0.0064, 1.0]],
                 [1, [0.025, 1.0]]],
          "Li": [[0, [0.13, 1.0]], [0, [0.055, 1.0]], [0, [0.022, 1.0]],
                 [1, [0.1, 1.0]], [1, [0.04, 1.0]], [1, [0.016, 1.0]]]}
    for el, shells in mm.items():
        for sh in shells:
            l, e = sh[0], sh[1][0]
            if any(sh2[0] == l and abs(np.log(e / sh2[1][0])) < np.log(1.4)
                   for sh2 in out[el]):
                continue
            out[el].append(list(sh))
    return out


def load_reference(ref_dir: str):
    """返回 (78 个 R/bohr, 样条值 meV, 原始数字化值 meV)。

    优先读本地预生成的 results/ref_ion_points.npz (服务器无参考目录);
    参考数据仅在比对阶段进入, 计算本身不读参考。"""
    pre = os.path.join("results", "ref_ion_points.npz")
    if os.path.exists(pre):
        d = np.load(pre)
        return d["r_bohr"], d["spl_mev"], d["raw_mev"]
    import sys
    sys.path.insert(0, ref_dir)
    from src.potentials import IonicPotential

    V = IonicPotential()
    d = np.loadtxt(os.path.join(ref_dir, "data_raw", "HeLip.txt"),
                   delimiter=",")
    r_ang, y = d[:, 0], d[:, 1]
    i = np.argsort(r_ang)
    r_ang, y = r_ang[i], y[i]
    v_inf = float(V.spline(float(r_ang.max())))
    r_bohr = r_ang / BOHR_ANG
    spl = np.array([float(V.spline(float(x))) for x in r_ang]) - v_inf  # eV
    raw = y - v_inf
    return r_bohr, spl * 1e3, raw * 1e3


def compute_points(basis: str, method: str, r_bohr: np.ndarray,
                   stride: int = 1):
    """未校正单点能曲线 (与识别出的参考方法一致, 无 counterpoise)。

    碎片 (He, Li⁺) 与二聚体必须同方法 — 否则相关能差污染渐近零点
    (RHF 碎片 + CCSD(T) 二聚体 → −1.55 eV 假偏移, 已在服务器实测)。"""
    from pyscf import cc, fci, gto, scf, ao2mo

    b = build_mm_uncontracted_basis() if basis == "mm-uncontracted" else basis
    if basis == "mm-uncontracted":
        print("    基组: 非收缩 cc-pVTZ + MM(2000) 弥散指数", flush=True)

    def e_frag(atom: str, charge: int = 0) -> float:
        m = gto.M(atom=atom, basis=b, spin=0, charge=charge,
                  unit="Bohr", verbose=0)
        mf = scf.RHF(m)
        e = mf.kernel()
        if method == "fci":
            hh = m.intor("int1e_kin") + m.intor("int1e_nuc")
            h1 = mf.mo_coeff.T @ hh @ mf.mo_coeff
            er = ao2mo.restore(1, ao2mo.kernel(m, mf.mo_coeff), m.nao)
            cs = fci.direct_spin1.FCI()
            return float(np.asarray(
                cs.kernel(h1, er, m.nao, (1, 1) if charge == 0 else (1, 1)
                          )[0]).ravel()[0]) + m.energy_nuc()
        mycc = cc.CCSD(mf)
        ec = mycc.kernel()[0]
        return float(e + ec + (mycc.ccsd_t() if method == "ccsd(t)" else 0.0))

    e_he = e_frag("He 0 0 0")
    e_lip = e_frag("Li 0 0 0", charge=1)
    print(f"    碎片: E(He)={e_he:.6f}  E(Li⁺)={e_lip:.6f} "
          f"(同方法 {method})", flush=True)
    idx = np.arange(0, len(r_bohr), stride)
    v = np.full(len(r_bohr), np.nan)
    t0 = time.time()
    for n, k in enumerate(idx):
        R = float(r_bohr[k])
        b = build_mm_uncontracted_basis() if basis == "mm-uncontracted"             else basis
        mol = gto.M(atom=f"He 0 0 0; Li 0 0 {R}", basis=b, spin=0,
                    charge=1, unit="Bohr", verbose=0)
        mf = scf.RHF(mol)
        e = mf.kernel()
        if method == "fci":
            h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            h1e = mf.mo_coeff.T @ h @ mf.mo_coeff
            eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
            cs = fci.direct_spin1.FCI()
            e = float(np.asarray(
                cs.kernel(h1e, eri, mol.nao, (2, 2))[0]).ravel()[0]) \
                + mol.energy_nuc()
        else:
            mycc = cc.CCSD(mf)
            e_corr = mycc.kernel()[0]
            e = float(e + e_corr + (mycc.ccsd_t() if method == "ccsd(t)"
                                    else 0.0))
        v[k] = (e - (e_he + e_lip)) * HA_EV * 1e3
        if (n + 1) % 10 == 0 or n == 0:
            print(f"    [{n+1}/{len(idx)}] R={R:6.2f} V={v[k]:+9.2f} meV "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="ccsdt_qz,fci_tz,ccsd_qz")
    ap.add_argument("--stride", type=int, default=1,
                    help="参考点抽样步长 (FCI 大基组降耗用)")
    ap.add_argument("--reference",
                    default="/Users/lihao/Library/CloudStorage/SynologyDrive-"
                            "aecho/doc/Pennningionization/Pro_HeLi_Enhanced")
    ap.add_argument("--out", default="results/iterate_ion_1pct.npz")
    a = ap.parse_args()

    ref_dir = a.reference
    if not os.path.isdir(ref_dir):
        ref_dir = os.environ.get("AQD_REF_DIR", ref_dir)
    r_bohr, spl_mev, raw_mev = load_reference(ref_dir)
    print(f"参考点: {len(r_bohr)} 个, R ∈ [{r_bohr[0]:.2f}, "
          f"{r_bohr[-1]:.2f}] bohr; 1% 目标 = {WELL_MEV/100:.2f} meV", flush=True)

    # 全局原子渐近 (每配置相同)
    results = {}
    for name in a.configs.split(","):
        cfg = CONFIGS[name]
        basis, method = cfg["basis"], cfg["method"]
        print(f"\n=== 配置 {name}: {method}/{basis} (未校正) ===", flush=True)
        v = compute_points(basis, method, r_bohr, a.stride)
        m = ~np.isnan(v)
        d_spl = v[m] - spl_mev[m]
        d_raw = v[m] - raw_mev[m]
        results[name] = dict(
            r=r_bohr[m], v=v[m], method=method, basis=basis,
            max_spl=float(np.abs(d_spl).max()), rms_spl=float(
                np.sqrt((d_spl ** 2).mean())),
            max_raw=float(np.abs(d_raw).max()), rms_raw=float(
                np.sqrt((d_raw ** 2).mean())),
        )
        r_ = results[name]
        print(f"  → vs 样条: max {r_['max_spl']:.2f} meV "
              f"({r_['max_spl']/WELL_MEV*100:.2f}%) RMS {r_['rms_spl']:.2f}"
              f" | vs 原始点: max {r_['max_raw']:.2f} RMS {r_['rms_raw']:.2f}",
              flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, **{f"v_{k}": results[k]["v"]
                                      for k in results},
                            **{f"r_{k}": results[k]["r"] for k in results},
                            meta=json.dumps({k: {kk: vv for kk, vv in
                                                 results[k].items()
                                                 if kk not in ("r", "v")}
                                             for k in results},
                                            ensure_ascii=False))
    print("\n=== 阶梯汇总 (max|Δ| meV, /87 = %阱深) ===", flush=True)
    for k, r_ in results.items():
        print(f"  {k:10s}: 样条 max {r_['max_spl']:6.2f} "
              f"({r_['max_spl']/WELL_MEV*100:5.2f}%)  原始 max {r_['max_raw']:6.2f}")


if __name__ == "__main__":
    main()
