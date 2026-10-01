#!/usr/bin/env python3
"""集群单分片能量扫描 (Slurm array 任务内运行)。

    python scripts/cluster_scan.py --e-min 0.10 --e-max 0.20 \
        --points 3 --part 0 --out scan_part0.npz

LEPS Jacobi 面 + 二维含时波包 (与 engine._step_dynamics_2d_wavepacket
的 LEPS 分支同参数), 结果存 npz 供 merge_scan.py 合并。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autoquantum.dynamics import (
    WavePacket2D, WavePacket2DPropagator, WavePacket2DScan,
    h3_reduced_masses, leps_jacobi_pes, leps_exchange_mask,
    morse_ground_width, harmonic_ground_width, DEFAULT_MASS_H,
    QCTEnsemble,
)
from autoquantum.pes.leps import LEPSBuilder
from autoquantum.pes.eckart import EckartBuilder



def _build_pes(pes_name: str):
    """按名称构建势能面 (返回 V(R,r) 闭包与初始参数)。"""
    if pes_name == "leps":
        builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028, "r0": 1.401,
                               "sato": 0.05})
        return leps_jacobi_pes(builder)
    elif pes_name == "eckart":
        builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                                 "r0": 1.401, "coupling": 0.08})
        return builder.evaluate_2d
    elif pes_name == "morse":
        from autoquantum.pes.analytic import MorsePES
        m = MorsePES({"D": 0.1744, "alpha": 1.028, "r0": 0.7416})
        return lambda R, r: m.evaluate(r)  # 仅 r 方向
    raise ValueError(f"未知势能面: {pes_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-min", type=float, required=True)
    parser.add_argument("--e-max", type=float, required=True)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--grid-r", type=int, default=192)
    parser.add_argument("--grid-rv", type=int, default=144)
    parser.add_argument("--pes", default="leps",
                        choices=["leps", "eckart", "morse"])
    parser.add_argument("--method", default="wavepacket",
                        choices=["wavepacket", "qct"],
                        help="动力学计算方法 (wavepacket: 量子波包; qct: 准经典轨线)")
    parser.add_argument("--n-traj", type=int, default=200,
                        help="QCT 采样每能量点轨迹数")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="时间步长 (au)")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--torch", action="store_true",
                        help="使用 PyTorch 后端 (GPU: 加 dtype=float32)")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mass_R, mass_r = h3_reduced_masses(DEFAULT_MASS_H)
    pes = _build_pes(args.pes)

    R = np.linspace(0.5, 10.0, args.grid_r)
    r = np.linspace(0.2, 9.5, args.grid_rv)

    if args.method == "qct":
        sigma_r = morse_ground_width(0.1744, 1.028, mass_r)
        R0 = 6.7
        r_mean = 1.401
        crit = lambda R_val, r_val: R_val < 1.5 * r_val
        if args.pes == "eckart":
            R0 = 6.0
            crit = lambda R_val, r_val: R_val < 2.0
            sigma_r = harmonic_ground_width(0.5, mass_r)
        elif args.pes == "morse":
            crit = lambda R_val, r_val: r_val > 4.0

        ensemble = QCTEnsemble(pes, mass_R, mass_r, dt=args.dt, max_steps=args.steps)
        energies = np.linspace(args.e_min, args.e_max, args.points)
        result = ensemble.run(energies, R0=R0, r_mean=r_mean, r_sigma=sigma_r,
                              reaction_criterion=crit, n_traj=args.n_traj,
                              seed=args.part * 1000 + 42)
        out = os.path.abspath(args.out)
        np.savez(out, energy=result.energies, reaction=result.reaction_probs,
                 method="qct", n_traj=args.n_traj,
                 energy_drift_max=result.energy_drift_max,
                 R_grid=R, r_grid=r)
        print(f"[part {args.part} (QCT)] E {args.e_min}-{args.e_max} au, "
              f"{args.points} 点 ({args.n_traj} 轨/点, 最大漂移 {result.energy_drift_max:.2e}) → {out}")
        for e, p in zip(result.energies, result.reaction_probs):
            print(f"  E={e:.4f}: P={p:.4f}")
        return

    Prop = WavePacket2DPropagator
    kw = {}
    if args.torch:
        from autoquantum.dynamics.wavepacket_2d_torch import (
            TorchWavePacket2DPropagator)
        Prop = TorchWavePacket2DPropagator
        kw = {"dtype": args.dtype}
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        kw["device"] = device
        print(f"[backend] torch {args.dtype} on {device}")
    prop = Prop(
        pes, R, r, mass_R, mass_r, dt=0.5,
        cap_edges=("R_min", "R_max", "r_min", "r_max"),
        cap_width_frac=0.05, cap_height=0.15, **kw)
    packet = WavePacket2D(
        R0=6.7, r0=1.401, sigma_R=0.5,
        sigma_r=morse_ground_width(0.1744, 1.028, mass_r))

    scan = WavePacket2DScan(
        prop, packet, leps_exchange_mask(), ("r_max",),
        reactant_mask=lambda Rr, rr: np.asarray(Rr) >= 1.5 * np.asarray(rr),
        n_steps=args.steps)
    result = scan.run(args.e_min, args.e_max, args.points)

    out = os.path.abspath(args.out)
    np.savez(out, energy=result.energy, reaction=result.reaction_prob,
             R_grid=R, r_grid=r)
    print(f"[part {args.part}] E {args.e_min}-{args.e_max} au, "
          f"{args.points} 点 → {out}")
    for e, p in zip(result.energy, result.reaction_prob):
        print(f"  E={e:.4f}: P={p:.4f}")


if __name__ == "__main__":
    main()
