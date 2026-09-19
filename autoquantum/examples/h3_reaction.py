#!/usr/bin/env python3
"""H + H₂ → H₂ + H 共线反应 — Eckart 势垒 + 谐振耦合势能面。"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from autoquantum.pes.eckart import EckartBuilder
from autoquantum.dynamics.quantum_1d import QuantumScattering1D
from autoquantum.visualization.contour import PESContourPlotter, ReactionPathPlotter
from autoquantum.visualization.plot import DynamicsPlotter


def run_h3_reaction(output_dir="output_h3_2d"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("H + H₂ → H₂ + H 共线反应 (Eckart 垒)")

    builder = EckartBuilder({
        "V0": 0.015, "beta": 1.5, "k_r": 0.5,
        "r0": 1.401, "coupling": 0.08,
    })

    R_range = (0.5, 6.0)
    r_range = (0.5, 3.0)

    print("[1/4] 构建势能面 ...")
    R_grid, r_grid, V_grid = builder.generate_grid(R_range, r_range, 200, 200)
    PESContourPlotter.plot_contour(
        R_grid, r_grid, V_grid,
        save_path=os.path.join(output_dir, "pes_contour.png"),
        title="H + H₂ Eckart Barrier PES",
    )
    PESContourPlotter.plot_3d(
        R_grid, r_grid, V_grid,
        save_path=os.path.join(output_dir, "pes_3d.png"),
    )

    print("[2/4] 提取最小能量路径 (MEP) ...")
    mep_V = np.min(V_grid, axis=1)
    V0 = mep_V.min()
    mep_V_rel = mep_V - V0

    ReactionPathPlotter.plot_reaction_profile(
        R_grid, mep_V_rel,
        save_path=os.path.join(output_dir, "reaction_profile.png"),
    )
    print(f"      势垒高度: {mep_V_rel.max():.4f} au ({mep_V_rel.max() * 27.2114:.2f} eV)")

    print("[3/4] 一维有效势量子散射 ...")
    def mep_potential(x):
        return np.interp(x, R_grid, mep_V_rel)

    solver = QuantumScattering1D(
        mass=1.0, pes=mep_potential,
        n_grid=800, grid_min=R_range[0], grid_max=R_range[1],
    )
    result = solver.solve(energy_min=0.001, energy_max=0.05, n_points=100)
    print(f"      最大反应概率: {result.transmission.max():.4f}")
    if np.any(result.transmission > 0.01):
        thresh_idx = np.argmax(result.transmission > 0.01)
        print(f"      阈值能量: {result.energy[thresh_idx]:.4f} au")
    else:
        print("      未达到反应阈值")

    DynamicsPlotter.plot_transmission(
        result.energy, result.transmission,
        save_path=os.path.join(output_dir, "reaction_probability.png"),
    )

    print("[4/4] 生成报告 ...")
    from autoquantum.visualization.dashboard import DashboardGenerator
    class Cfg: pass
    cfg = Cfg()
    cfg.system_name = "H3_2D_Eckart"
    cfg.mass = 1.0
    cfg.pes_type = "eckart"
    cfg.output_dir = output_dir
    DashboardGenerator({"pes_2d": builder, "dynamics_result": result, "dynamics_dim": "2d"}, cfg).generate_all()

    print(f"\n输出目录: {output_dir}/")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_h3_reaction()
