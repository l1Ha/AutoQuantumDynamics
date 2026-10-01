#!/usr/bin/env python3
"""量子波包 (QM) 与准经典轨线 (QCT) 对比分析与可视化。

该脚本在同一势能面与运动学参数下, 对比全量子含时波包与准经典轨线 (QCT) 的反应概率:
- 低能端 (E < V_barrier): 量子隧穿导致 P_QM > P_QCT (经典禁阻区 QCT=0)
- 高能端 (E >> V_barrier): 经典极限对应原理 P_QM ≈ P_QCT
- 波动结构: 量子干涉与几何相导致 P_QM 展现丰富结构, QCT 展现平滑经典阈值行为

用法:
    python scripts/compare_qct_quantum.py --pes leps --e-min 0.12 --e-max 0.28 --points 6 --n-traj 200
    python scripts/compare_qct_quantum.py --smoke  # 冒烟测试快速跑
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoquantum.dynamics import (
    WavePacket2D, WavePacket2DPropagator, WavePacket2DScan,
    QCTEnsemble, h3_reduced_masses, DEFAULT_MASS_H,
    leps_jacobi_pes, leps_exchange_mask, morse_ground_width,
    harmonic_ground_width,
)
from autoquantum.pes.leps import LEPSBuilder
from autoquantum.pes.eckart import EckartBuilder


def build_system(pes_name: str = "leps", grid_r: int = 192, grid_rv: int = 144):
    """构建势能面、质量与几何分界面。"""
    mass_R, mass_r = h3_reduced_masses(DEFAULT_MASS_H)
    R_grid = np.linspace(0.5, 10.0, grid_r)
    r_grid = np.linspace(0.2, 9.5, grid_rv)

    if pes_name == "leps":
        builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028, "r0": 1.401, "sato": 0.05})
        pes = leps_jacobi_pes(builder)
        R0 = 6.7
        r_mean = 1.401
        sigma_r = morse_ground_width(0.1744, 1.028, mass_r)
        crit = lambda R, r: R < 1.5 * r
        prod_mask = leps_exchange_mask()
        v_barrier = 0.134  # LEPS sato=0.05 教学参数势垒高度 (au)
    elif pes_name == "eckart":
        builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5, "r0": 1.401, "coupling": 0.08})
        pes = builder.evaluate_2d
        R0 = 6.0
        r_mean = 1.401
        sigma_r = harmonic_ground_width(0.5, mass_r)
        crit = lambda R, r: R < 2.0
        prod_mask = lambda Rr, rr: np.asarray(Rr) <= 2.0
        v_barrier = 0.015
    else:
        raise ValueError(f"未知势能面: {pes_name}")

    return {
        "pes": pes,
        "mass_R": mass_R,
        "mass_r": mass_r,
        "R_grid": R_grid,
        "r_grid": r_grid,
        "R0": R0,
        "r_mean": r_mean,
        "sigma_r": sigma_r,
        "crit": crit,
        "prod_mask": prod_mask,
        "v_barrier": v_barrier,
    }


def run_comparison(pes_name: str = "leps",
                   energies: np.ndarray = None,
                   n_traj: int = 200,
                   qm_steps: int = 2500,
                   qct_steps: int = 2500,
                   dt: float = 0.5,
                   grid_r: int = 192,
                   grid_rv: int = 144) -> dict:
    """运行 QM 与 QCT 并行计算并返回比较结果。"""
    sys_info = build_system(pes_name, grid_r, grid_rv)
    if energies is None:
        energies = np.linspace(0.12, 0.28, 6)

    print(f"=== QM vs QCT 动力学对比计算 ({pes_name.upper()}) ===")
    print(f"能量点数: {len(energies)} [{energies[0]:.4f} ~ {energies[-1]:.4f} au]")
    print(f"QM 网格: {grid_r}x{grid_rv}, 步长 dt={dt}, 步数={qm_steps}")
    print(f"QCT 轨迹数: {n_traj}/点, 步数={qct_steps}")

    # 1. 准经典轨线 (QCT)
    print("\n[1/2] 正在运行 QCT 轨迹系综 ...")
    ensemble = QCTEnsemble(sys_info["pes"], sys_info["mass_R"], sys_info["mass_r"],
                           dt=dt, max_steps=qct_steps)
    qct_res = ensemble.run(energies, R0=sys_info["R0"], r_mean=sys_info["r_mean"],
                           r_sigma=sys_info["sigma_r"], reaction_criterion=sys_info["crit"],
                           n_traj=n_traj, seed=42)
    print(f"  QCT 完成! 最大能量漂移: {qct_res.energy_drift_max:.2e}")

    # 2. 量子波包 (QM)
    print("\n[2/2] 正在运行 2D 量子波包传播 ...")
    prop = WavePacket2DPropagator(
        sys_info["pes"], sys_info["R_grid"], sys_info["r_grid"],
        sys_info["mass_R"], sys_info["mass_r"], dt=dt,
        cap_edges=("R_min", "R_max", "r_min", "r_max"),
        cap_width_frac=0.05, cap_height=0.15)
    packet = WavePacket2D(
        R0=sys_info["R0"], r0=sys_info["r_mean"], sigma_R=0.5,
        sigma_r=sys_info["sigma_r"])
    scan = WavePacket2DScan(
        prop, packet, sys_info["prod_mask"], ("r_max",),
        reactant_mask=lambda Rr, rr: np.asarray(Rr) >= 1.5 * np.asarray(rr) if pes_name == "leps" else np.asarray(Rr) >= 2.0,
        n_steps=qm_steps)
    qm_res = scan.run(energies[0], energies[-1], len(energies))
    print(f"  QM 完成!")

    qm_probs = qm_res.reaction_prob
    qct_probs = qct_res.reaction_probs

    print("\n--- 计算结果对比表 ---")
    print(f"{'E (au)':<10} | {'P_QM':<10} | {'P_QCT':<10} | {'QM - QCT':<10} | {'物理效应特征'}")
    print("-" * 65)
    for e, pqm, pqct in zip(energies, qm_probs, qct_probs):
        diff = pqm - pqct
        if e < sys_info["v_barrier"]:
            feature = "量子隧穿占优 (E < V_barrier)"
        elif pqct > 0.8 and abs(diff) < 0.2:
            feature = "高能经典对应收敛 (E >> V_barrier)"
        else:
            feature = "过渡区 / 干涉效应"
        print(f"{e:<10.4f} | {pqm:<10.4f} | {pqct:<10.4f} | {diff:<+10.4f} | {feature}")

    return {
        "energies": energies,
        "qm_probs": qm_probs,
        "qct_probs": qct_probs,
        "energy_drift_max": qct_res.energy_drift_max,
        "v_barrier": sys_info["v_barrier"],
        "pes_name": pes_name,
    }


def plot_comparison(res: dict, save_path: str = "results/qct_vs_quantum.png"):
    """绘制出版级 QM vs QCT 动力学对比图 (双子图: 反应几率对比 + 量子修正增量)。"""
    energies = res["energies"]
    qm_probs = res["qm_probs"]
    qct_probs = res["qct_probs"]
    v_barrier = res["v_barrier"]
    pes_name = res["pes_name"].upper()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # 子图 1: 绝对反应概率对比
    ax1.plot(energies, qm_probs, "o-", color="#1f77b4", lw=2.2, ms=6,
             label=r"Quantum Wavepacket (QM, $\hbar=1$)")
    ax1.plot(energies, qct_probs, "s--", color="#d62728", lw=2.0, ms=6,
             label=r"Quasi-Classical Trajectory (QCT)")

    ax1.axvline(v_barrier, color="gray", linestyle=":", lw=1.5,
                label=f"Classical Barrier $V^{{\\ddagger}} \\approx {v_barrier:.3f}$ au")

    # 阴影标注隧穿区与经典区
    ax1.axvspan(energies[0], v_barrier, color="#3498db", alpha=0.1, label="Quantum Tunneling Regime")
    ax1.axvspan(v_barrier, energies[-1], color="#e67e22", alpha=0.08, label="Classical Allowed Regime")

    ax1.set_ylabel("Reaction Probability $P(E)$", fontsize=11)
    ax1.set_title(f"{pes_name} System: Quantum Wavepacket vs. Quasi-Classical Trajectory (QCT)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 子图 2: 量子效应差值 Delta P = P_QM - P_QCT
    delta_p = qm_probs - qct_probs
    ax2.plot(energies, delta_p, "^-", color="#2ca02c", lw=1.8, ms=5,
             label=r"$\Delta P = P_{\mathrm{QM}} - P_{\mathrm{QCT}}$ (Quantum Correction)")
    ax2.axhline(0.0, color="black", linestyle="-", lw=0.8, alpha=0.7)
    ax2.axvline(v_barrier, color="gray", linestyle=":", lw=1.5)

    ax2.set_xlabel("Collision Energy $E_{\\mathrm{coll}}$ (Hartree)", fontsize=11)
    ax2.set_ylabel(r"$\Delta P$", fontsize=11)
    ax2.set_ylim(-0.35, 0.45)
    ax2.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n对比图已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="QM 与 QCT 反应动力学对比")
    parser.add_argument("--pes", default="leps", choices=["leps", "eckart"],
                        help="势能面类型")
    parser.add_argument("--e-min", type=float, default=0.12, help="最小碰撞能量 (au)")
    parser.add_argument("--e-max", type=float, default=0.28, help="最大碰撞能量 (au)")
    parser.add_argument("--points", type=int, default=6, help="能量扫描点数")
    parser.add_argument("--n-traj", type=int, default=200, help="QCT 每能量点轨迹数")
    parser.add_argument("--qm-steps", type=int, default=2500, help="QM 传播步数")
    parser.add_argument("--qct-steps", type=int, default=2500, help="QCT 最大积分步数")
    parser.add_argument("--dt", type=float, default=0.5, help="时间步长 (au)")
    parser.add_argument("--grid-r", type=int, default=192, help="R 轴网格数")
    parser.add_argument("--grid-rv", type=int, default=144, help="r 轴网格数")
    parser.add_argument("--out", default="results/qct_vs_quantum", help="输出路径前缀")
    parser.add_argument("--smoke", action="store_true", help="快速冒烟测试 (小网格少量点)")
    args = parser.parse_args()

    if args.smoke:
        args.points = 3
        args.n_traj = 30
        args.qm_steps = 800
        args.qct_steps = 800
        args.grid_r = 96
        args.grid_rv = 64
        args.e_min = 0.15
        args.e_max = 0.25

    energies = np.linspace(args.e_min, args.e_max, args.points)
    res = run_comparison(
        pes_name=args.pes,
        energies=energies,
        n_traj=args.n_traj,
        qm_steps=args.qm_steps,
        qct_steps=args.qct_steps,
        dt=args.dt,
        grid_r=args.grid_r,
        grid_rv=args.grid_rv,
    )

    out_prefix = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    np.savez(
        out_prefix + ".npz",
        energies=res["energies"],
        qm_probs=res["qm_probs"],
        qct_probs=res["qct_probs"],
        v_barrier=res["v_barrier"],
        pes=res["pes_name"],
        energy_drift_max=res["energy_drift_max"],
    )
    print(f"数据已保存至: {out_prefix}.npz")

    plot_comparison(res, save_path=out_prefix + ".png")


if __name__ == "__main__":
    main()
