#!/usr/bin/env python3
"""用生产参考势能面数据校核 AutoQuantum 从头算曲线 (He* + Li)。

对每个 R 计算并输出:
- ⁴Σ⁺ (He*(2³S)+Li, 自旋禁阻通道) 相互作用能, 含 counterpoise (CP) 校正
- ²Σ⁺ 基态 (He(1s²)+Li) 相互作用能, 含 CP 校正
- He + Li⁺ 离子势 V⁺(R) (彭宁电离出射通道)
- 原子渐近参考 (He*³S / He¹S / Li / Li⁺)

并与参考数据 (MLR 4 通道 + 离子势 + MRCI 宽度) 对比绘图:
    python scripts/validate_vs_reference_pes.py --compute --out results/heli_validation.npz
    python scripts/validate_vs_reference_pes.py --plot --data results/heli_validation.npz \
        --reference /path/to/Pro_HeLi_Enhanced --fig results/heli_vs_reference.png

物理要点 (本脚本的核心发现, 详见 --plot 输出):
1. ⁴Σ⁺ 的"范德华阱"在 CP 校正后消失: 未校正曲线在 R≈10.5 bohr 的 ~50 meV
   阱完全来自基组叠加误差 (BSSE); 该通道在 SCF 层面是纯排斥的。
2. 反应性通道 ²Σ⁺(He*(2³S)+Li) 是**多参考**态 (两个 2s 电子在成键轨道配对),
   PySCF 的 ROHF 对同一空间占据的双重态/四重态给出相同能量 (能量泛函不区分
   自旋耦合), 因此 SCF 层面无法复现参考的 0.855 eV 深阱 — 需 CASSCF/MRCI。
3. 彭宁电子动能 = 19.819614 eV (He* 激发能) - 5.391715 eV (Li 电离能)
   = 14.4279 eV (减去势能面差值修正), 而非垂直激发能隙。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

HA_EV = 27.211386245988


# ---------------------------------------------------------------------------
# 从头算部分 (需 pyscf; 在集群上运行)
# ---------------------------------------------------------------------------

def compute_curves(basis: str = "aug-cc-pVTZ", r_min: float = 3.2,
                   r_max: float = 20.0, n_points: int = 17,
                   out_path: str = "results/heli_validation.npz") -> str:
    """计算 ⁴Σ⁺/²Σ⁺(基态)/离子通道曲线, 含 counterpoise 校正。

    注意: 所有 gto.M 必须显式 unit="Bohr" — PySCF 默认单位是 Ångström,
    混淆会造成 1.89 倍的距离尺度错误 (本项目调试中真实踩过)。
    超分子四重态用 MOM 链式跟踪, 防止短程塌陷到其他四重态解。
    """
    from pyscf import gto, scf
    from autoquantum.pes.calculators import PySCFCalculator

    def e_scf(atom: str, spin: int, charge: int = 0) -> float:
        mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge,
                    unit="Bohr", verbose=0)
        mf = scf.ROHF(mol) if spin > 0 else scf.RHF(mol)
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"SCF 未收敛: {atom} spin={spin}")
        return float(e)

    print("=" * 78)
    print(f"AutoQuantum 参考势校核计算 (basis={basis}, 全部 R 以 Bohr 计)")
    print("=" * 78)

    e_he_star = e_scf("He 0 0 0", 2)      # He*(1s2s) ³S
    e_he_1s = e_scf("He 0 0 0", 0)        # He(1s²)
    e_li = e_scf("Li 0 0 0", 1)           # Li(2s)
    e_li_plus = e_scf("Li 0 0 0", 0, charge=1)
    asym = dict(He_1S=e_he_1s, He_3S=e_he_star, Li=e_li, Li_plus=e_li_plus)
    print(f"原子参考: He*³S={e_he_star:.6f}  He¹S={e_he_1s:.6f}  "
          f"Li={e_li:.6f}  Li⁺={e_li_plus:.6f} Ha")
    print(f"  He* 激发能 = {(e_he_star-e_he_1s)*HA_EV:.4f} eV (NIST 19.8196); "
          f"Li IP = {(e_li_plus-e_li)*HA_EV:.4f} eV (NIST 5.3917)")

    r_grid = np.linspace(r_min, r_max, n_points)
    v_q_unc = np.zeros(n_points); v_q_cp = np.zeros(n_points)
    v_d_unc = np.zeros(n_points); v_d_cp = np.zeros(n_points)
    v_ion = np.zeros(n_points)
    bsse_q = np.zeros(n_points)

    # 四重态 Calculator (MOM 链式; unit 默认 Bohr)
    calc_q = PySCFCalculator(symbols=["He", "Li"], basis=basis, charge=0,
                             spin=3, method="rohf", use_mom=True, spin_lock=True)

    t0 = time.time()
    order = np.argsort(r_grid)[::-1]   # 由大 R 向小 R 传播 MOM 参考
    for k, i in enumerate(order):
        R = float(r_grid[i])
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
        # --- 四重态超分子 (MOM) ---
        e_sup_q, _ = calc_q.energy_and_gradient(coords)
        # --- 四重态 CP 片段 (dimer 基组) ---
        e_he_g = e_scf(f"He 0 0 0; ghost:Li 0 0 {R}", 2)
        e_li_g = e_scf(f"ghost:He 0 0 0; Li 0 0 {R}", 1)
        v_q_unc[i] = (e_sup_q - e_he_star - e_li) * HA_EV
        v_q_cp[i] = (e_sup_q - e_he_g - e_li_g) * HA_EV
        bsse_q[i] = v_q_unc[i] - v_q_cp[i]
        # --- 基态双重态 (He+Li) ---
        e_sup_d = e_scf(f"He 0 0 0; Li 0 0 {R}", 1)
        e_he_g0 = e_scf(f"He 0 0 0; ghost:Li 0 0 {R}", 0)
        v_d_unc[i] = (e_sup_d - e_he_1s - e_li) * HA_EV
        v_d_cp[i] = (e_sup_d - e_he_g0 - e_li_g) * HA_EV
        # --- 离子通道 (He + Li⁺) ---
        e_sup_i = e_scf(f"He 0 0 0; Li 0 0 {R}", 0, charge=1)
        v_ion[i] = (e_sup_i - e_he_1s - e_li_plus) * HA_EV
        if (k + 1) % 4 == 0 or k == 0:
            print(f"  [{k+1:2d}/{n_points}] R={R:5.2f} | V(⁴Σ⁺)_CP={v_q_cp[i]:+9.4f} meV "
                  f"(BSSE={bsse_q[i]:+8.2f}) | V(²Σ⁺)_CP={v_d_cp[i]:+9.4f} | "
                  f"V_ion={v_ion[i]*1e3:+9.3f} meV")

    # 状态一致性自检: 片段在 dimer 基组中的能量不应比自由单体低太多 (除非真 BSSE)
    print(f"\n片段状态自检 (He* 在 ghost-Li 基组中): max BSSE = "
          f"{np.max(np.abs(bsse_q)):.2f} meV "
          f"({'合理' if np.max(np.abs(bsse_q)) < 200 else '⚠ 可能有态塌陷'})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, r_grid=r_grid, basis=basis,
                        v_quartet_unc=v_q_unc, v_quartet_cp=v_q_cp,
                        v_doublet_unc=v_d_unc, v_doublet_cp=v_d_cp,
                        v_ion=v_ion, bsse_quartet=bsse_q,
                        asymptotes=json.dumps(asym),
                        elapsed=time.time() - t0)
    print(f"\n✓ 计算完成 ({time.time()-t0:.1f}s) → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 对比绘图 (本地; 需要参考数据目录)
# ---------------------------------------------------------------------------

def plot_comparison(data_path: str, ref_dir: str | None, fig_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # CJK 字体
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

    d = np.load(data_path, allow_pickle=True)
    R = d["r_grid"]
    asym = json.loads(str(d["asymptotes"]))
    basis = str(d["basis"])
    R_ang = R * 0.529177210903

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))

    # ---- (a) 势能曲线总览 ----
    ax = axes[0, 0]
    ax.plot(R, d["v_quartet_unc"] * 1e3, "r--", lw=1.4, label="⁴Σ⁺ 未校正 (含 BSSE)")
    ax.plot(R, d["v_quartet_cp"] * 1e3, "r.-", lw=2.0, ms=8, label="⁴Σ⁺ CP 校正 (本工作)")
    ax.plot(R, d["v_doublet_cp"] * 1e3, "b.-", lw=2.0, ms=8, label="²Σ⁺ 基态 CP 校正")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R (bohr)"); ax.set_ylabel("V(R) - V(∞)  (meV)")
    ax.set_title(f"(a) AutoQuantum 从头算曲线 ({basis})")
    ax.set_ylim(-80, 60); ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.6)

    # ---- (b) 与参考 MLR 对比 (含两种单位解读) ----
    ax = axes[0, 1]
    ref_ok = False
    if ref_dir and os.path.isdir(ref_dir):
        sys.path.insert(0, ref_dir)
        try:
            from src.potentials import EntrancePotential, HESTAR_LI_MLR_PARAMS, mlr_potential
            from src.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
            ref_ok = True

            Rr = np.linspace(3.0, 20.0, 400)
            pot = EntrancePotential("2Sigma_He23S1")
            ax.plot(Rr, np.array([pot(x) for x in Rr]) * 1e3, "k-", lw=2.0,
                    label="参考 ²Σ⁺(He*2³S) 生产版 (代码原样)")

            # 单位自洽版: C 视为 eV·Å⁶, φ_inf 用一致单位
            p = HESTAR_LI_MLR_PARAMS["2Sigma_He23S1"]
            De_ev = p["D_e"] * HARTREE_TO_EV

            def mlr_consistent(Rb):
                Ra = Rb * BOHR_TO_ANGSTROM
                Re = p["R_e"]
                u = p["C6"]/Ra**6 + p["C8"]/Ra**8 + p["C10"]/Ra**10
                uRe = p["C6"]/Re**6 + p["C8"]/Re**8 + p["C10"]/Re**10
                yp = (Ra**4 - Re**4) / (Ra**4 + Re**4)
                phi_inf = np.log(2*De_ev/uRe)          # eV 与 eV → 量纲一致
                phi_poly = sum(pj*yp**j for j, pj in enumerate(p["phi"]))
                phi = (1-yp)*phi_poly + phi_inf*yp
                v_ev = De_ev*(1 - (u/uRe)*np.exp(-phi*yp))**2 - De_ev
                return v_ev / HARTREE_TO_EV

            ax.plot(Rr, np.array([mlr_consistent(x) for x in Rr]) * 1e3, "k:",
                    lw=1.8, label=r"参考 ²Σ⁺ 单位自洽版 (C 视为 eV·$\AA^6$)")
            ax.plot(R, d["v_quartet_cp"] * 1e3, "r.-", lw=2.0, ms=8,
                    label="⁴Σ⁺ CP 校正 (本工作)")
        except Exception as exc:
            print(f"  [warn] 参考数据加载失败: {exc}")
    if not ref_ok:
        ax.text(0.5, 0.5, "参考数据不可用\n(需 --reference 指向 Pro_HeLi_Enhanced)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("R (bohr)"); ax.set_ylabel("V(R)  (meV)")
    ax.set_title("(b) 与生产参考势对比 (注意 ²Σ⁺ 深阱 vs ⁴Σ⁺ 无阱)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.6)

    # ---- (c) 长程尾巴 ----
    ax = axes[1, 0]
    ax.plot(R, d["v_quartet_unc"] * 1e3, "r--", lw=1.4, label="⁴Σ⁺ 未校正 (BSSE)")
    ax.plot(R, d["v_quartet_cp"] * 1e3, "r.-", lw=2.0, ms=8, label="⁴Σ⁺ CP 校正")
    Rr = np.linspace(max(6, R.min()), R.max(), 300)
    if ref_ok:
        ax.plot(Rr, np.array([pot(x) for x in Rr]) * 1e3, "k-", lw=2.0,
                label="参考 ²Σ⁺ 生产版尾巴")
        ax.plot(Rr, np.array([mlr_consistent(x) for x in Rr]) * 1e3, "k:", lw=1.8,
                label="参考 ²Σ⁺ 单位自洽版尾巴")
        # 物理 C6 尾巴参考线 (C6 = 2090 eV·Å⁶ → 3498 a.u.)
        C6_au = 2090.0 / HARTREE_TO_EV * (1.0/0.529177210903)**6
        ax.plot(Rr, -C6_au/Rr**6 * 1e3 * HA_EV, "g-.", lw=1.4,
                label=f"$-C_6/R^6$ (C₆={C6_au:.0f} a.u., 若 C 为 eV·Å⁶)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R (bohr)"); ax.set_ylabel("V(R)  (meV)")
    ax.set_title("(c) 长程行为: BSSE 与参考尾巴 (对数纵轴见下)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.6)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    # ---- (d) 自电离宽度 ----
    ax = axes[1, 1]
    if ref_ok:
        try:
            from src.widths import load_xy_data
            base = ref_dir
            for fn, tag, c in [("data_raw/G_2Sigma.txt", "²Σ MRCI (参考)", "k"),
                               ("data_raw/G_2Pi.txt", "²Π MRCI (参考)", "b")]:
                x, y = load_xy_data(os.path.join(base, fn))
                ax.plot(x, y, c + "o-", ms=5, lw=1.4, label=f"{tag} Γ (meV @ R/Å)")
            # 我的旧模型 (bohr 单位, Ha): 0.04 exp(-1.1 R)
            Rb = np.linspace(3.0, 20.0, 200)
            gamma_old = 0.04 * np.exp(-1.1 * Rb) * 1e3 * HA_EV
            ax.semilogy(Rb, gamma_old, "m--", lw=1.8,
                        label=r"旧模型 $\Gamma=0.04e^{-1.1R}$ Ha (R/bohr)")
            ax.set_xlabel("R  (Å for 参考数据 / bohr for 旧模型)")
            ax.set_ylabel(r"$\Gamma$ (meV)")
            ax.set_title("(d) 自电离宽度: MRCI 参考数据 vs 旧解析模型")
            ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.6)
        except Exception as exc:
            print(f"  [warn] 宽度数据加载失败: {exc}")
    else:
        ax.text(0.5, 0.5, "参考数据不可用", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"✓ 对比图: {fig_path}")
    return fig_path


def main():
    ap = argparse.ArgumentParser(description="AutoQuantum vs 参考势能面校核")
    ap.add_argument("--compute", action="store_true", help="运行从头算 (需 pyscf)")
    ap.add_argument("--plot", action="store_true", help="绘制对比图")
    ap.add_argument("--basis", default="aug-cc-pVTZ")
    ap.add_argument("--r-min", type=float, default=3.2)
    ap.add_argument("--r-max", type=float, default=20.0)
    ap.add_argument("--n-points", type=int, default=17)
    ap.add_argument("--data", default="results/heli_validation.npz")
    ap.add_argument("--reference", default=None, help="Pro_HeLi_Enhanced 目录")
    ap.add_argument("--fig", default="results/heli_vs_reference.png")
    a = ap.parse_args()

    if a.compute:
        compute_curves(a.basis, a.r_min, a.r_max, a.n_points, a.data)
    if a.plot:
        plot_comparison(a.data, a.reference, a.fig)
    if not (a.compute or a.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
