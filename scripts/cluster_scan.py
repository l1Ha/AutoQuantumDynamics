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
    morse_ground_width, DEFAULT_MASS_H,
)
from autoquantum.pes.leps import LEPSBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-min", type=float, required=True)
    parser.add_argument("--e-max", type=float, required=True)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--grid-r", type=int, default=192)
    parser.add_argument("--grid-rv", type=int, default=144)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mass_R, mass_r = h3_reduced_masses(DEFAULT_MASS_H)
    builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028, "r0": 1.401,
                           "sato": 0.05})
    pes = leps_jacobi_pes(builder)

    R = np.linspace(0.5, 10.0, args.grid_r)
    r = np.linspace(0.2, 9.5, args.grid_rv)
    prop = WavePacket2DPropagator(
        pes, R, r, mass_R, mass_r, dt=0.5,
        cap_edges=("R_min", "R_max", "r_min", "r_max"),
        cap_width_frac=0.05, cap_height=0.15)
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
