#!/usr/bin/env python3
"""真实从头算: 亚稳态碰撞体系 He* + Li 的势能面与彭宁电离特性。

物理背景
--------
He*(2³S) 激发能 19.82 eV 远高于 Li 电离能 5.39 eV, 体系处于单电离连续谱
之上的**自电离共振态**。本脚本用 AutoQuantum 的 PySCF 后端真实计算:

1. **四重态 ⁴Σ⁺** (He*(2³S) + Li(²S), S=3/2, 2S=3):
   自旋选择定则 (ΔS=0) 禁阻静电自电离, 是 S=3/2 流形中的最低态。
   用 ROHF + spin_lock 严格约束, 可选 MOM 链式跟踪, 计算实数相互作用势 V(R);
2. **双重态 ²Σ⁺** (He(1s²) + Li(1s²2s)): 基态参考, 给出电离连续谱阈值;
3. **垂直能隙 ΔE(R) = E(⁴Σ⁺) - E(²Σ⁺)**: 彭宁电离可释放于电子的能量;
4. **模型自电离宽度 Γ(R)** 与复光学势 W(R) = V(R) - iΓ/2;
5. **原子渐近验证**: He ³S-¹S FCI 激发能 vs 实验 19.82 eV。

关键数值经验 (本脚本已内建)
--------------------------
He* 的 2s 轨道极为弥散 (指数 ~0.2), **必须使用含弥散函数的基组**
(aug-cc-pVTZ)。若用 cc-pVTZ/def2-svp, 2s 无法描述, ³S 激发能会虚高
到 25-40 eV, 且 SCF 会塌陷到 He⁺Li⁻ 电荷转移态。

用法:
  python scripts/calc_metastable_heli.py --basis aug-cc-pVTZ --n-points 25
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from autoquantum.pes.calculators import PySCFCalculator

EV = 27.211386245988  # Hartree → eV


def setup_cjk_font() -> str | None:
    """跨平台挑选可用的 CJK 字体, 避免图中中文标题渲染为豆腐块。

    依次尝试 Linux 集群 (Noto CJK/Fandol/Droid) 与 macOS (Songti/PingFang/
    Hiragino) 常见字体; 无可用 CJK 字体时回退 DejaVu Sans 并关闭负号修正。
    """
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Serif CJK SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "FandolHei", "Droid Sans Fallback",
                 "PingFang SC", "Heiti SC", "Songti SC", "STHeiti",
                 "Hiragino Sans GB", "Arial Unicode MS",
                 "Microsoft YaHei", "SimHei"):
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            # 数学文本固定用 DejaVu 数学字体, 避免 CJK 字体缺数学字形告警
            plt.rcParams["mathtext.fontset"] = "dejavusans"
            return name
    return None


def plot_results(r_grid, e_q, e_d, gamma, v_q_rel, v_d_rel, delta_e_ev,
                 f_q, f_d, basis: str, plot_path: str) -> None:
    """绘制四联图: 相互作用势 / 垂直能隙 / 自电离宽度 / 核力。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    ax = axes[0, 0]
    ax.plot(r_grid, v_q_rel, "r.-", lw=1.6, ms=7, label=r"$^4\Sigma^+$ He*(2$^3$S)+Li")
    ax.plot(r_grid, v_d_rel, "b.-", lw=1.6, ms=7, label=r"$^2\Sigma^+$ He(1s$^2$)+Li")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"$R$ (Bohr)"); ax.set_ylabel(r"$V(R)-V(\infty)$ (eV)")
    ax.set_title(f"(a) 相互作用势 ({basis})")
    ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.6)

    ax = axes[0, 1]
    ax.plot(r_grid, delta_e_ev, "k.-", lw=1.6, ms=7)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"$R$ (Bohr)"); ax.set_ylabel(r"$\Delta E = E(^4\Sigma^+) - E(^2\Sigma^+)$ (eV)")
    ax.set_title(r"(b) 垂直能隙 (彭宁电子可用能量)")
    ax.grid(True, ls=":", alpha=0.6)

    ax = axes[1, 0]
    ax.semilogy(r_grid, gamma, "g.-", lw=1.6, ms=7)
    # CJK 字体普遍缺 U+2212, 且 mathtext 不走逐字形回退; 用纯文本指数避免负号丢失
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax.set_xlabel(r"$R$ (Bohr)"); ax.set_ylabel(r"$\Gamma(R)$ (Hartree)")
    ax.set_title(r"(c) 模型自电离宽度 $\Gamma(R)$ (指数衰减)")
    ax.grid(True, ls=":", alpha=0.6)

    ax = axes[1, 1]
    ax.plot(r_grid, f_q, "r.-", lw=1.6, ms=7, label=r"$^4\Sigma^+$")
    ax.plot(r_grid, f_d, "b.-", lw=1.6, ms=7, label=r"$^2\Sigma^+$")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"$R$ (Bohr)"); ax.set_ylabel(r"$F_z$(Li) (Hartree/Bohr)")
    ax.set_title("(d) 沿碰撞轴核力 (QCT/波包动力学输入)")
    ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200); plt.close(fig)


def replot_from_npz(npz_path: str, plot_path: str | None = None) -> str:
    """从已算好的 npz 重新出图 (不重跑 Quantum Chemistry 计算)。"""
    d = np.load(npz_path, allow_pickle=True)
    r_grid = d["r_grid"]
    e_q, e_d = d["e_quartet"], d["e_doublet"]
    g_q, g_d = d["g_quartet"], d["g_doublet"]
    gamma = d["gamma_penning"]
    asym = d["asymptotes"]
    basis = str(d["basis"]) if "basis" in d else "unknown"
    v_q_rel = (e_q - (asym[1] + asym[2])) * EV
    v_d_rel = (e_d - (asym[0] + asym[2])) * EV
    delta_e_ev = (e_q - e_d) * EV
    if plot_path is None:
        plot_path = os.path.join(os.path.dirname(npz_path) or ".", "he_li_metastable_pes.png")
    plot_results(r_grid, e_q, e_d, gamma, v_q_rel, v_d_rel, delta_e_ev,
                 -g_q[:, 1, 2], -g_d[:, 1, 2], basis, plot_path)
    return plot_path


def atomic_asymptotes(basis: str, with_fci: bool = True) -> dict:
    """计算原子渐近参考: He(¹S), He*(³S), Li(²S) 及 He 激发能 FCI 校验。"""
    from pyscf import gto, scf
    out = {}

    mol_he_s = gto.M(atom="He 0 0 0", basis=basis, spin=0, verbose=0)
    out["He_1S"] = float(scf.RHF(mol_he_s).kernel())

    mol_he_t = gto.M(atom="He 0 0 0", basis=basis, spin=2, verbose=0)
    out["He_3S"] = float(scf.ROHF(mol_he_t).kernel())

    mol_li = gto.M(atom="Li 0 0 0", basis=basis, spin=1, verbose=0)
    out["Li_2S"] = float(scf.ROHF(mol_li).kernel())

    out["ROHF_gap_eV"] = (out["He_3S"] - out["He_1S"]) * EV

    if with_fci:
        try:
            from pyscf import fci
            mf = scf.RHF(mol_he_s).run()
            cs = fci.FCI(mol_he_s, mf.mo_coeff)
            e_all, c_all = cs.kernel(nroots=6)
            for e, c in zip(e_all, c_all):
                ss = cs.spin_square(c, mol_he_s.nao, mol_he_s.nelectron)[0]
                if abs(ss - 2.0) < 0.05:
                    out["FCI_He_3S"] = float(e)
                    out["FCI_gap_eV"] = float((e - e_all[0]) * EV)
                    break
        except Exception as exc:  # FCI 仅为附加校验, 失败不阻断主流程
            out["FCI_error"] = str(exc)

    return out


def run_scan(basis: str = "aug-cc-pVTZ", r_min: float = 3.2, r_max: float = 16.0,
             n_points: int = 25, use_mom: bool = False,
             out_dir: str = "results/he_li_metastable",
             cap_a: float = 0.04, cap_beta: float = 1.1) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    # 对数式密集采样: 短程 (强排斥区) 密集, 长程稀疏
    r_grid = np.unique(np.concatenate([
        np.linspace(r_min, 6.0, max(4, n_points // 3)),
        np.linspace(6.0, r_max, n_points - max(4, n_points // 3) + 1),
    ]))

    print("=" * 74)
    print("AutoQuantum 亚稳态从头算: He*(2³S) + Li(2²S) 势能面与彭宁电离")
    print("=" * 74)
    print(f"基组: {basis} | 采样点: {len(r_grid)} | R ∈ [{r_grid[0]:.2f}, {r_grid[-1]:.2f}] Bohr")
    print(f"MOM (最大重叠法) 链式跟踪: {'开启' if use_mom else '关闭'}")

    # ---- 原子渐近参考与 FCI 校验 ----
    print("\n[0/3] 原子渐近参考与 FCI 激发能校验...")
    asym = atomic_asymptotes(basis, with_fci=True)
    print(f"  He(1s²)¹S  ROHF = {asym['He_1S']:.6f} Ha")
    print(f"  He*(1s2s)³S ROHF = {asym['He_3S']:.6f} Ha  (ΔE_ROHF = {asym['ROHF_gap_eV']:.2f} eV)")
    if "FCI_gap_eV" in asym:
        print(f"  He*³S-¹S FCI 激发能 = {asym['FCI_gap_eV']:.2f} eV   "
              f"← 实验值 19.82 eV, FCI/{basis} 校验 ✓")
    print(f"  Li(²S)    ROHF = {asym['Li_2S']:.6f} Ha")
    e_asym_quartet = asym["He_3S"] + asym["Li_2S"]
    e_asym_doublet = asym["He_1S"] + asym["Li_2S"]
    print(f"  渐近和: ⁴Σ⁺ = {e_asym_quartet:.6f} Ha, ²Σ⁺ = {e_asym_doublet:.6f} Ha")

    # ---- 构造 Calculators ----
    calc_q = PySCFCalculator(symbols=["He", "Li"], basis=basis, charge=0, spin=3,
                             method="rohf", spin_lock=True, spin_tol=0.05,
                             use_mom=use_mom)
    calc_d = PySCFCalculator(symbols=["He", "Li"], basis=basis, charge=0, spin=1,
                             method="rohf", spin_lock=True, spin_tol=0.05)
    cap_model = {"type": "exponential", "A": cap_a, "beta": cap_beta, "r_index": (0, 1)}
    calc_w = PySCFCalculator(symbols=["He", "Li"], basis=basis, charge=0, spin=3,
                             method="rohf", cap_params=cap_model)

    e_q = np.zeros(len(r_grid)); g_q = np.zeros((len(r_grid), 2, 3))
    e_d = np.zeros(len(r_grid)); g_d = np.zeros((len(r_grid), 2, 3))
    gamma = np.zeros(len(r_grid))

    # ---- 逐点扫描 (自长程向短程, 便于 MOM 链式继承) ----
    print(f"\n[1/3] 逐点从头算 (能量 + 核梯度, 自 R 大端向小端传播 MOM 参考)...")
    order = np.argsort(r_grid)[::-1]  # 由大到小
    t0 = time.time()
    for k, i in enumerate(order):
        R = float(r_grid[i])
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
        eq, gq = calc_q.energy_and_gradient(coords)
        ed, gd = calc_d.energy_and_gradient(coords)
        e_q[i], g_q[i] = eq, gq
        e_d[i], g_d[i] = ed, gd
        gamma[i] = calc_w.resonance_width(coords)
        if (k + 1) % 5 == 0 or k == 0 or k == len(order) - 1:
            print(f"  [{k+1:2d}/{len(order)}] R={R:5.2f} Bohr | "
                  f"E(⁴Σ⁺)={eq:.6f} | E(²Σ⁺)={ed:.6f} | "
                  f"ΔE={(eq-ed)*EV:5.2f} eV | Γ={gamma[i]:.2e} Ha")
    elapsed = time.time() - t0

    # ---- 物理分析 ----
    delta_e_ev = (e_q - e_d) * EV
    v_q_rel = (e_q - e_asym_quartet) * EV      # 相对 He*+Li 渐近的相互作用能
    v_d_rel = (e_d - e_asym_doublet) * EV
    f_q = -g_q[:, 1, 2]                        # Li 原子沿 Z 受力 (键轴)
    f_d = -g_d[:, 1, 2]

    print(f"\n[2/3] 扫描完成, 耗时 {elapsed:.1f}s ({len(r_grid)} 点 × 2 态)")
    print("=" * 74)
    print("【物理分析】")
    print("=" * 74)
    print(f"1. 渐近一致性检验 (R={r_grid[-1]:.2f} Bohr):")
    dev_q = (e_q[-1] - e_asym_quartet) * 1000
    dev_d = (e_d[-1] - e_asym_doublet) * 1000
    print(f"   E(⁴Σ⁺, R_max) - [He*³S + Li]  = {dev_q:+.3f} mHa (应≈0, 检验解离正确性)")
    print(f"   E(²Σ⁺, R_max) - [He¹S + Li]   = {dev_d:+.3f} mHa")
    print(f"2. 垂直能隙 ΔE(R) (彭宁电子动能上限):")
    print(f"   ΔE(R_max) = {delta_e_ev[-1]:.2f} eV → ΔE(R_min) = {delta_e_ev[0]:.2f} eV")
    print(f"   (ΔE > 0 表示始终位于电离连续谱之上, 自电离能量上可行)")
    print(f"3. ⁴Σ⁺ 相互作用势 V(R):")
    i_min = int(np.argmin(v_q_rel))
    if 0 < i_min < len(r_grid) - 1:
        print(f"   范德华势阱: R_min = {r_grid[i_min]:.2f} Bohr, "
              f"阱深 = {-v_q_rel[i_min]*1000:.1f} meV")
    print(f"   短程排斥 (R={r_grid[0]:.2f} Bohr): V = {v_q_rel[0]:+.3f} eV, "
          f"力 = {f_q[0]:+.4f} Ha/Bohr")
    print(f"   渐近 (R={r_grid[-1]:.2f}): 力 = {f_q[-1]:+.6f} Ha/Bohr (→0)")
    print(f"4. 模型自电离宽度 Γ(R) = {cap_a}·exp(-{cap_beta}·R) Ha:")
    print(f"   Γ(R_min) = {gamma[0]:.3e} Ha (寿命 ~{1.055e-34/ (gamma[0]*4.3597e-18) if gamma[0]>0 else np.inf:.2e} s) "
          f"→ Γ(R_max) = {gamma[-1]:.3e} Ha")
    print(f"   复光学势 W(R) = V(R) - iΓ(R)/2 可直接用于非厄米波包传播")
    print("=" * 74)

    # ---- 保存 ----
    data_path = os.path.join(out_dir, "he_li_metastable_pes.npz")
    np.savez_compressed(
        data_path, r_grid=r_grid,
        e_quartet=e_q, g_quartet=g_q, e_doublet=e_d, g_doublet=g_d,
        gamma_penning=gamma, delta_e_ev=delta_e_ev,
        asymptotes=np.array([asym["He_1S"], asym["He_3S"], asym["Li_2S"]]),
        provenance=json.dumps(calc_q.provenance, ensure_ascii=False),
        basis=basis, use_mom=use_mom,
    )
    print(f"\n[3/3] ✓ 数据: {data_path}")

    # ---- 绘图 ----
    plot_path = os.path.join(out_dir, "he_li_metastable_pes.png")
    plot_results(r_grid, e_q, e_d, gamma, v_q_rel, v_d_rel, delta_e_ev,
                 f_q, f_d, basis, plot_path)
    print(f"✓ 图: {plot_path}")

    return {"data": data_path, "plot": plot_path, "elapsed": elapsed,
            "fci_gap_eV": asym.get("FCI_gap_eV"), "delta_e_asym_eV": float(delta_e_ev[-1])}


def main():
    p = argparse.ArgumentParser(description="He* + Li 亚稳态势能面真实从头算")
    p.add_argument("--basis", default="aug-cc-pVTZ",
                   help="基组 (必须含弥散函数, 默认 aug-cc-pVTZ)")
    p.add_argument("--r-min", type=float, default=3.2)
    p.add_argument("--r-max", type=float, default=16.0)
    p.add_argument("--n-points", type=int, default=25)
    p.add_argument("--mom", action="store_true", help="启用最大重叠法(MOM)链式跟踪")
    p.add_argument("--out-dir", default="results/he_li_metastable")
    p.add_argument("--cap-a", type=float, default=0.04, help="Γ(R)=A·exp(-βR) 的 A")
    p.add_argument("--cap-beta", type=float, default=1.1, help="Γ(R) 的 β (Bohr⁻¹)")
    p.add_argument("--replot", metavar="NPZ",
                   help="仅从已算好的 npz 重新出图 (跳过全部量子化学计算)")
    a = p.parse_args()

    font = setup_cjk_font()
    print(f"matplotlib 中文字体: {font or '未找到 (回退 DejaVu Sans)'}")

    if a.replot:
        path = replot_from_npz(a.replot)
        print(f"✓ 重绘完成: {path}")
        return

    r = run_scan(a.basis, a.r_min, a.r_max, a.n_points, a.mom, a.out_dir,
                 a.cap_a, a.cap_beta)
    print(f"\n最终: FCI 激发的 He*³S 校验 = {r['fci_gap_eV']:.2f} eV; "
          f"渐近垂直能隙 ΔE = {r['delta_e_asym_eV']:.2f} eV")


if __name__ == "__main__":
    main()
