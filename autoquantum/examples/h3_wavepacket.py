#!/usr/bin/env python3
"""H + H₂ 二维含时波包示例 — Eckart 垒 (默认) 或 LEPS 交换面。

物理单位: ħ = 1, 能量 Hartree, 长度 Bohr, 质量 = 电子质量。
演示: 网格/初态构建 → CAP 检查 → 碰撞能扫描 → 单能量含时演化
→ 快照/通道概率/GIF 可视化。

用法:
    python autoquantum/examples/h3_wavepacket.py                # Eckart 垒
    python autoquantum/examples/h3_wavepacket.py --pes leps     # LEPS 交换
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from autoquantum.pes.eckart import EckartBuilder
from autoquantum.pes.leps import LEPSBuilder
from autoquantum.dynamics import (
    WavePacket2D, WavePacket2DPropagator, WavePacket2DScan,
    h3_reduced_masses, leps_jacobi_pes, leps_exchange_mask,
    eckart_product_mask, morse_ground_width, harmonic_ground_width,
    exchange_dividing_surface_line,
)
from autoquantum.visualization import WavePacket2DPlotter, PESContourPlotter


def run(pes_name: str = "eckart", output_dir: str = "output_h3_wavepacket",
        n_scan: int = 5, n_steps: int = 3000):
    os.makedirs(output_dir, exist_ok=True)
    mass_R, mass_r = h3_reduced_masses()

    # ---- PES 与体系配置 ----
    if pes_name == "eckart":
        builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                                 "r0": 1.401, "coupling": 0.08})
        pes = builder.evaluate_2d
        R_range, r_range = (0.5, 9.0), (0.5, 3.5)
        R0, r0 = 6.0, 1.401 - 0.08 / 0.5 * np.tanh(1.5 * (6.0 - 3.0))
        sigma_r = harmonic_ground_width(0.5, mass_r)
        product_mask, product_edges = eckart_product_mask(2.0), ("R_min",)
        reactant_mask = lambda R, r: np.asarray(R) >= 2.0
        ds_line = ([2.0, 2.0], [r_range[0], r_range[1]])
        e_min, e_max = 0.005, 0.05       # Eckart 垒高 0.015 au (0.41 eV)
        cap_edges = ("R_min", "R_max")
    else:
        builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028,
                               "r0": 1.401, "sato": 0.05})
        pes = leps_jacobi_pes(builder)
        R_range, r_range = (0.5, 10.0), (0.2, 9.5)
        R0, r0 = 6.7, 1.401
        sigma_r = morse_ground_width(0.1744, 1.028, mass_r)
        product_mask, product_edges = leps_exchange_mask(), ("r_max",)
        reactant_mask = lambda R, r: np.asarray(R) >= 1.5 * np.asarray(r)
        ds_line = exchange_dividing_surface_line(np.linspace(0.5, 10.0, 2))
        e_min, e_max = 0.10, 0.35        # 该 LEPS 参数化 MEP 垒 ~0.14 au
        cap_edges = ("R_min", "R_max", "r_min", "r_max")

    print("=" * 60)
    print(f"H + H₂ 二维含时波包 ({'Eckart 垒' if pes_name == 'eckart' else 'LEPS 交换'})")

    # ---- 网格与传播子 ----
    print("[1/4] 构建网格与传播子 ...")
    R_grid = np.linspace(*R_range, 192)
    r_grid = np.linspace(*r_range, 144)
    prop = WavePacket2DPropagator(
        pes, R_grid, r_grid, mass_R, mass_r, dt=0.5,
        cap_edges=cap_edges, cap_width_frac=0.05 if pes_name == "leps" else 0.15,
        cap_height=0.15,
    )
    packet = WavePacket2D(R0=R0, r0=r0, sigma_R=0.5, sigma_r=sigma_r)

    overlap = prop.check_initial_overlap(packet.initialize(R_grid, r_grid))
    print(f"      初始波包-CAP 重叠: {overlap:.2e} (应 <1e-4)")
    if overlap > 1e-4:
        raise RuntimeError("CAP 与初态重叠过大, 请调整网格")

    PESContourPlotter.plot_contour(
        R_grid, r_grid, prop.V,
        save_path=os.path.join(output_dir, "pes_contour.png"),
        title=f"H + H₂ {'Eckart' if pes_name == 'eckart' else 'LEPS'} PES (Jacobi)",
    )

    # ---- 碰撞能扫描 ----
    print(f"[2/4] 碰撞能扫描 ({e_min}-{e_max} au, {n_scan} 点) ...")
    scan = WavePacket2DScan(prop, packet, product_mask, product_edges,
                            reactant_mask=reactant_mask, n_steps=n_steps)
    scan_result = scan.run(e_min, e_max, n_scan)
    for E, P in zip(scan_result.energy, scan_result.reaction_prob):
        print(f"      E_coll = {E:.4f} au ({E * 27.211:5.2f} eV): "
              f"P_react = {P:.4f}")

    # ---- 单能量含时演化演示 ----
    E_demo = float(e_min + 0.75 * (e_max - e_min))
    print(f"[3/4] 含时演化演示 (E_coll = {E_demo:.4f} au) ...")
    packet.p_R0 = -np.sqrt(2.0 * mass_R * E_demo)
    wp = prop.propagate(packet.initialize(R_grid, r_grid), n_steps,
                        save_every=n_steps // 12,
                        product_mask=product_mask, product_edges=product_edges,
                        reactant_mask=reactant_mask, save_density=True)
    print(f"      P_react = {wp.reaction_prob[-1]:.4f}, "
          f"P_refl = {wp.reflection_prob[-1]:.4f}, "
          f"存活 = {wp.norm_t[-1]:.4f}")
    identity = wp.norm_t[-1] + sum(v[-1] for v in wp.absorbed.values())
    print(f"      恒等式 存活+吸收 = {identity:.10f} (应为 1)")

    # ---- 可视化 ----
    print("[4/4] 生成可视化 ...")
    WavePacket2DPlotter.plot_snapshots(
        wp, save_path=os.path.join(output_dir, "wavepacket_snapshots.png"),
        n_show=4, ds_line=ds_line)
    WavePacket2DPlotter.plot_probabilities(
        wp, save_path=os.path.join(output_dir, "wavepacket_probability.png"))
    WavePacket2DPlotter.save_animation(
        wp, save_path=os.path.join(output_dir, "wavepacket.gif"), ds_line=ds_line)

    print(f"\n输出目录: {output_dir}/")
    print("=" * 60)
    return scan_result, wp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pes", choices=["eckart", "leps"], default="eckart")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--scan-points", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    out = args.output or f"output_h3_wavepacket_{args.pes}"
    run(args.pes, out, args.scan_points, args.steps)
