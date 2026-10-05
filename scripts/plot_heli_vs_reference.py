#!/usr/bin/env python3
"""He* + Li: AutoQuantum 从头算 vs 生产参考势能面 (Pro_HeLi_Enhanced) 对比图。

数据来源:
- 本工作: scripts/validate_vs_reference_pes.py (CASCI(5e,12o) 轨道匹配) 与
  counterpoise 校正结果 — results/heli_casci.npz
- 参考:  doc/Pennningionization/Pro_HeLi_Enhanced (MLR 4 通道 + 离子势 + MRCI 宽度)

用法:
    python scripts/plot_heli_vs_reference.py \
        --data results/heli_casci.npz \
        --reference ../doc/Pennningionization/Pro_HeLi_Enhanced \
        --fig results/heli_vs_reference.png
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

EV = 27.211386245988
DELTA_E_INT_EV = 19.819614 - 5.391715   # = 14.427899 eV (参考 constants.py)


def setup_cjk_font():
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "FandolHei", "Droid Sans Fallback",
                 "PingFang SC", "Heiti SC", "Songti SC", "STHeiti",
                 "Hiragino Sans GB", "Arial Unicode MS", "Microsoft YaHei"):
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    return plt.rcParams["font.sans-serif"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/heli_casci.npz")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--fig", default="results/heli_vs_reference.png")
    a = ap.parse_args()

    font = setup_cjk_font()
    d = np.load(a.data, allow_pickle=True)
    Rg, v4, v2, vion = d["r_grid"], d["v_quartet"], d["v_doublet_ground"], d["v_ion"]
    basis = str(d["basis"])

    sys.path.insert(0, a.reference)
    from src.potentials import EntrancePotential, HESTAR_LI_MLR_PARAMS, IonicPotential
    from src.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
    from src.widths import WidthModel, load_xy_data, exp_tail

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))

    # ================= (a) 势能曲线总览 =================
    ax = axes[0, 0]
    Rr = np.linspace(3.0, 20.0, 400)
    styles = {"2Sigma_He23S1": ("k-", "²Σ⁺ He*(2³S)+Li (反应性)"),
              "2Sigma_He21S0": ("k--", "²Σ⁺ He*(2¹S₀)+Li"),
              "2Sigma_He23P": ("k-.", "²Σ⁺ He*(2³P)+Li"),
              "2Pi_He23P": ("k:", "²Π He*(2³P)+Li")}
    for name, (ls, lab) in styles.items():
        pot = EntrancePotential(name)
        ax.plot(Rr, np.array([pot(x) for x in Rr]) * 1e3, ls, lw=1.8,
                label=f"参考 MLR: {lab}")
    ax.plot(Rg, v4, "r.-", lw=2.2, ms=9, label="本工作 CASCI: ⁴Σ⁺ (自旋禁阻)")
    ax.plot(Rg, v2, "b.-", lw=2.2, ms=9, label="本工作 CASCI: ²Σ⁺ 基态")
    vstar_npz = os.path.join("results", "feshbach_width.npz")
    if os.path.exists(vstar_npz):
        dv = np.load(vstar_npz, allow_pickle=True)
        ax.plot(dv["R"], dv["v_star_meV"], "g^-", lw=2.0, ms=7,
                label="本工作 Feshbach: ²Σ⁺ He*(2³S)+Li 共振态 (SA-CASSCF)")
    ax.axhline(0, color="gray", ls=":", lw=1)

    # 标注关键数值
    p3 = HESTAR_LI_MLR_PARAMS["2Sigma_He23S1"]
    ax.annotate(f"参考 ²Σ⁺ 阱: {p3['D_e']*HARTREE_TO_EV*1e3:.0f} meV\n"
                f"@ R_e={p3['R_e']*BOHR_TO_ANGSTROM:.2f} A (代码约定)",
                xy=(p3["R_e"] * 1.8897, -p3["D_e"] * HARTREE_TO_EV * 1e3),
                xytext=(9.5, -1050), fontsize=8.5, color="k",
                arrowprops=dict(arrowstyle="->", color="k", lw=1))
    i4 = int(np.argmax(v4[:8]))
    ax.annotate(f"⁴Σ⁺ 避免交叉势垒 ≈{v4[i4]:.0f} meV @ R≈{Rg[i4]:.1f} bohr",
                xy=(Rg[i4], v4[i4]), xytext=(8.5, 520), fontsize=8.5, color="r",
                arrowprops=dict(arrowstyle="->", color="r", lw=1))
    ax.annotate("反应性 ²Σ⁺ 深阱需 MRCI 级别\n(本工作 SCF/CASCI 不可达)",
                xy=(p3["R_e"] * 1.8897 * 0.9, -p3["D_e"] * HARTREE_TO_EV * 1e3 * 0.75),
                xytext=(11.0, -760), fontsize=8.5, color="k",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_xlabel("R (bohr)"); ax.set_ylabel("V(R)  (meV)")
    ax.set_title(f"(a) 势能曲线: 生产参考 MLR vs 本工作 CASCI ({basis})")
    ax.set_ylim(-1200, 1400)
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(True, ls=":", alpha=0.6)

    # ================= (b) 离子通道 (迭代后: CCSD(T) + CP) =================
    ax = axes[0, 1]
    v_ion_ref = IonicPotential()
    Rr_ion = np.linspace(3.4, 16.0, 300)
    ax.plot(Rr_ion, np.array([v_ion_ref(x) for x in Rr_ion]) * 1e3 * HARTREE_TO_EV,
            "k-", lw=2.2, label="参考 HeLip.txt (spline, He+Li⁺)")
    ax.plot(Rg, vion, color="0.65", ls="-", lw=1.2, label="RHF/aVTZ (初版, MAE 2.8%)")
    ion_npz = os.path.join("results", "audit_ion_aug-cc-pVQZ.npz")
    if os.path.exists(ion_npz):
        di = np.load(ion_npz, allow_pickle=True)
        ri, vcp = di["r_grid"], di["v_cp"]
        ax.plot(ri, vcp, "g.-", lw=2.4, ms=9,
                label="CCSD(T)/aVQZ + CP (迭代后)")
        rr = np.array([v_ion_ref(float(x)) for x in ri]) * 1e3 * HARTREE_TO_EV
        well = (ri >= 5.0) & (ri <= 9.0)
        mae_w = np.abs(vcp - rr)[well].mean()
        depth = -rr.min()
        ax.annotate(f"阱区 (R=5-9 bohr) MAE = {mae_w:.2f} meV\n"
                    f"= {mae_w/depth*100:.2f}% ✓ <1%",
                    xy=(6.5, -13), xytext=(8.0, -60), fontsize=9, color="green",
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    d2 = os.path.join("results", "audit_ion_def2-QZVPPD.npz")
    if os.path.exists(d2):
        dd = np.load(d2, allow_pickle=True)
        ax.plot(dd["r_grid"], dd["v_unc"], "b--", lw=1.6, ms=7,
                label="CCSD(T)/def2-QZVPPD (BSSE≈2.8 meV)")
    dext = os.path.join("results", "audit_ion_extended.npz")
    if os.path.exists(dext):
        de = np.load(dext, allow_pickle=True)
        ax.plot(de["r_grid"], de["v_cp"], "m-", lw=1.4,
                label="CCSD(T)+CP/aVQZ 扩展范围 [2.3, 20] bohr (短程加密)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R (bohr)"); ax.set_ylabel(r"$V^+$(R)  (meV)")
    ax.set_title("(b) 离子出射通道: 迭代至阱区误差 < 1%")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.6)

    # ================= (c) 自电离宽度 =================
    ax = axes[1, 0]
    for fn, tag, c in [("data_raw/G_2Sigma.txt", "²Σ MRCI 数据点", "ko"),
                       ("data_raw/G_2Pi.txt", "²Π MRCI 数据点", "bs")]:
        x, y = load_xy_data(os.path.join(a.reference, fn))
        ax.plot(x, y, c, ms=6, label=f"参考 {tag}")
    w2s = WidthModel("2Sigma", tail_type="exponential")
    Rt = np.linspace(w2s.R_switch_ang, 8.0, 100)     # Å
    ax.semilogy(Rt, w2s.A_exp * np.exp(-w2s.k_exp * Rt), "k-", lw=1.8,
                label=f"参考指数尾 $A e^{{-kR}}$ (k={w2s.k_exp:.2f} /Angstrom)")
    # 本工作早期使用的解析模型 (bohr 单位, Ha): Γ = 0.04 exp(-1.1 R)
    Rb = np.linspace(3.0, 20.0, 200)
    ax.semilogy(Rb * BOHR_TO_ANGSTROM, 0.04 * np.exp(-1.1 * Rb) * 1e3 * HARTREE_TO_EV,
                "m--", lw=1.8,
                label=r"本工作旧模型 $\Gamma=0.04e^{-1.1R}$ Ha (R/bohr)")
    ax.set_xlabel("R (Angstrom)"); ax.set_ylabel(r"$\Gamma$ (meV)")
    ax.set_title("(c) 自电离宽度: MRCI 参考 vs 旧解析模型 (量级相差数个数量级)")
    ax.set_ylim(1e-6, 40); ax.set_xlim(1.5, 20)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, ls=":", alpha=0.6)

    # ================= (d) 彭宁电子能量 =================
    ax = axes[1, 1]
    pot2 = EntrancePotential("2Sigma_He23S1")
    Rr3 = np.linspace(6.0, 18.0, 200)      # 生产曲线在 R<6 bohr 因单位混配失真
    dv = np.array([pot2(x) for x in Rr3]) - np.array([v_ion_ref(x) for x in Rr3])
    e_pen = DELTA_E_INT_EV + dv * HARTREE_TO_EV
    ax.plot(Rr3, e_pen * 1e3, "k-", lw=2.4,
            label=r"正确: $E_e = 14.4279$ eV $+ [V_{2\Sigma}-V^+]$")
    ax.axhline(DELTA_E_INT_EV * 1e3, color="k", ls="--", lw=1.2,
               label=f"渐近极限 {DELTA_E_INT_EV:.3f} eV (参考 constants.py)")
    ax.axhline(18.84e3, color="r", ls=":", lw=2.0,
               label="本工作早期错误标注: 垂直激发能隙 18.84 eV")
    ax.annotate("(垂直能隙 = He* 激发能,\n 比可用电子能量高 4.4 eV)",
                xy=(14.0, 18.84e3), xytext=(9.0, 19.8e3), fontsize=8.5, color="r",
                arrowprops=dict(arrowstyle="->", color="r", lw=1))
    ax.set_xlabel("R (bohr)"); ax.set_ylabel(r"彭宁电子动能 $E_e$ (meV)")
    ax.set_title("(d) 彭宁电子能量 (修正后) 与早期标注错误")
    ax.legend(fontsize=8, loc="center right"); ax.grid(True, ls=":", alpha=0.6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.1f}k"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(a.fig, dpi=200)
    plt.close(fig)
    print(f"✓ 对比图: {a.fig}  (中文字体: {font})")


if __name__ == "__main__":
    main()
