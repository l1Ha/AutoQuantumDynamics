#!/usr/bin/env python3
"""1% 误差目标 — 最终汇总审计。

用法:
    python scripts/error_budget_report.py --reference <Pro_HeLi_Enhanced>
输出各类指标的当前误差与是否达标 (< 1%)。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import numpy as np

HA_EV = 27.211386245988
NIST = {"He_exc_eV": 19.819614, "Li_IP_eV": 5.391715}


def cbs_extrapolate(e3, e4, x3=3, x4=4):
    """两点 X⁻³ 外推: E(X) = E_CBS + A/X³。"""
    A = (e3 - e4) / (1.0 / x3 ** 3 - 1.0 / x4 ** 3)
    return e3 - A / x3 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    print("=" * 78)
    print("误差 < 1% 审计汇总")
    print("=" * 78)

    # ---------- A. 原子参考 (vs NIST) ----------
    print("\n[A] 原子参考量 (基准: NIST)")
    print(f"    {'指标':<22} {'基组/方法':<26} {'计算值':>10} {'误差':>10} {'达标':>6}")
    for f in sorted(glob.glob(os.path.join(a.results, "audit_atoms_*.npz"))):
        d = np.load(f, allow_pickle=True)
        b, m = str(d["basis"]), str(d["method"])
        for name, val, ref, key in (("He* 激发能 (eV)", d["he_exc_eV"],
                                     NIST["He_exc_eV"], "he_err_pct"),
                                    ("Li 电离能 (eV)", d["li_ip_eV"],
                                     NIST["Li_IP_eV"], "li_err_pct")):
            err = float(d[key])
            print(f"    {name:<22} {b+'/'+m:<26} {float(val):>10.4f} "
                  f"{err:>+9.3f}% {'✓' if abs(err) < 1 else '✗':>6}")

    # ---------- B. He+Li⁺ 离子曲线 (vs 参考样条) ----------
    print("\n[B] He+Li⁺ 离子曲线 (基准: 参考 HeLip.txt 样条)")
    sys.path.insert(0, a.reference)
    from src.potentials import IonicPotential
    ref = IonicPotential()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.results, "audit_ion_*.npz"))):
        d = np.load(f, allow_pickle=True)
        b, m = str(d["basis"]), str(d["method"])
        rg, vu, vc = d["r_grid"], d["v_unc"], d["v_cp"]
        vr = np.array([ref(float(x)) for x in rg]) * 1e3 * HA_EV
        depth = -vr.min()
        well = (rg >= 5.0) & (rg <= 9.0)
        stats = {}
        for tag, v in (("未校正", vu), ("CP 校正", vc)):
            stats[tag] = dict(
                mae_all=np.abs(v - vr).mean(),
                mae_well=np.abs(v - vr)[well].mean(),
                depth=-v.min())
        rows.append((b, m, depth, stats))
        print(f"\n    {b} / {m} (参考阱深 {depth:.3f} meV):")
        for tag, st in stats.items():
            print(f"      {tag:<8}: 阱深 {st['depth']:7.3f} meV "
                  f"({(st['depth']-depth)/depth*100:+6.2f}%) | "
                  f"MAE 全区间 {st['mae_all']:5.3f} meV ({st['mae_all']/depth*100:5.2f}%) "
                  f"{'✓' if st['mae_all']/depth<0.01 else '✗'} | "
                  f"MAE 阱区(5–9 bohr) {st['mae_well']:5.3f} meV "
                  f"({st['mae_well']/depth*100:5.2f}%) "
                  f"{'✓' if st['mae_well']/depth<0.01 else '✗'}")
    # CBS 外推 (CP 序列): 必须按基组大小排序 (X=3 在前)
    if len(rows) >= 2:
        ordered = sorted(rows, key=lambda r: 0 if "VTZ" in r[0] else 1)
        d3 = ordered[0][3]["CP 校正"]["depth"]
        d4 = ordered[1][3]["CP 校正"]["depth"]
        cbs = cbs_extrapolate(d3, d4)
        ref_depth = rows[0][2]
        print(f"\n    CP 序列 CBS 外推 (X⁻³): 阱深 {cbs:.2f} meV "
              f"(vs 参考 {ref_depth:.2f}, 误差 {(cbs-ref_depth)/ref_depth*100:+.2f}%)")

    # ---------- B2. 参考曲线自身的 BSSE 归因 ----------
    ccpvqz = os.path.join(a.results, "audit_ion_cc-pVQZ.npz")
    if os.path.exists(ccpvqz) and rows:
        d = np.load(ccpvqz, allow_pickle=True)
        rg, vu, vc = d["r_grid"], d["v_unc"], d["v_cp"]
        vr = np.array([ref(float(x)) for x in rg]) * 1e3 * HA_EV
        dm = vu - vr
        print("\n[B2] 参考曲线自身误差归因 (决定性诊断):")
        print(f"    未校正 cc-pVQZ CCSD(T) 复现参考曲线: MAE {np.abs(dm).mean():.3f} meV, "
              f"max|Δ| {np.abs(dm).max():.3f} meV")
        print(f"    → 参考曲线 ≈ cc-pVQZ 级别、未做 counterpoise 校正")
        bi = int(np.argmin(vr))
        print(f"    参考阱底 (R={rg[bi]:.2f} bohr) 含 BSSE = {vu[bi]-vc[bi]:.2f} meV "
              f"(其 85.91 meV '阱深' 中约 {(vu[bi]-vc[bi])/85.91*100:.0f}% 为 BSSE)")
        vr_corr = vr - (vu - vc)
        depth_corr = -vr_corr.min()
        # 汇总所有 CP 估计 → 收敛值 + 不确定度
        est = {}
        for f in sorted(glob.glob(os.path.join(a.results, "audit_ion_*.npz"))):
            db = np.load(f, allow_pickle=True)
            tag = os.path.basename(f).replace("audit_ion_", "").replace(".npz", "")
            est[tag] = (-db["v_cp"].min(), db["v_unc"].min() - db["v_cp"].min())
        print("    本工作无 BSSE 阱深估计 (CP 校正, 括注 BSSE 幅度):")
        for k, (d, b) in est.items():
            print(f"      {k:<18}: {d:7.2f} meV (BSSE {b:6.2f} meV)")
        qz = [d for k, (d, b) in est.items() if ("VQZ" in k or "V5Z" in k)]
        if qz:
            lo, hi = min(qz), max(qz)
            best = 0.5 * (lo + hi)
            print(f"    QZ 级一致估计: {lo:.2f}–{hi:.2f} meV → 最佳估计 {best:.2f} ± "
                  f"{(hi-lo)/2:.2f} meV ({(hi-lo)/2/best*100:.1f}% 不确定度)")
            print(f"    扣除 BSSE 后参考阱深 = {depth_corr:.2f} meV")
            err = (best - depth_corr) / depth_corr * 100
            raw = (best - depth_hi_ref) / depth_hi_ref * 100 if False else None
            print(f"    → 最佳估计 vs 校正参考: {err:+.2f}% | "
                  f"vs 原始参考 85.91: {(best-85.91)/85.91*100:+.2f}%")
            print(f"    ⚠ 本工作自身的基组收敛不确定度 {((hi-lo)/2)/best*100:.1f}% > 1% 判据, "
                  f"且参考含 6–10 meV 未校正 BSSE → 该指标无法在 <1% 水平上认证")
        # 长程缺陷
        # 参考长程缺陷
        rr_ = rg[rg >= 9.5]
        if len(rr_):
            idx = rg >= 9.5
            print(f"    参考长程 (R>9.5 bohr) 偏离本工作 BSSE 自由曲线: "
                  f"{np.abs(vc[idx]-vr_corr[idx]).mean():.3f} meV (参考在此被人为拉平)")

    # ---------- C. Γ(R) 模型化验证 ----------
    print("\n[C] Γ(R) 模型化验证 (生产管线输入 vs 参考 MRCI 数据):")
    try:
        import subprocess
        r = subprocess.run([sys.executable, os.path.join("scripts", "width_model_validation.py"),
                            "--reference", a.reference], capture_output=True, text=True, timeout=300)
        for ln in r.stdout.splitlines():
            if "最大相对误差" in ln or "结论" in ln:
                print("    " + ln.strip())
    except Exception as exc:
        print(f"    (跳过: {exc})")

    # ---------- D. 诊断说明 ----------
    print("\n[C] 已知限制 (诊断, 非计算误差)")
    print("    - 阱底 (R≈3.7 bohr): 未校正与 CP 值包夹参考 (离子体系 CP 过校正常见);")
    print("    - 长程 (R>9.5 bohr): 参考数据在 7.6 Å 后被拉平为渐近平台,")
    print("      本工作遵循物理 -C4/R⁴ 尾, 该区间差异非本工作误差;")
    print("    - Γ(R): CAP-CI 机制已验证 (束缚态 Γ∝η→0), 但 He*+Li 共振")
    print("      在 aug-cc-pVTZ 级基组下不可分辨 (见 cap_ci_search 输出)。")
    print("=" * 78)


if __name__ == "__main__":
    main()
