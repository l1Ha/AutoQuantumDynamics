#!/usr/bin/env python3
"""可复现微基准 — 科学计算软件应当诚实报告性能基线。

    python scripts/benchmark.py            # 完整 (约 1-2 分钟)
    python scripts/benchmark.py --quick    # 冒烟 (约 15 秒)
    python scripts/benchmark.py --json out.json

测量: 二维波包单步吞吐、能量扫描单点耗时、NN 训练每轮耗时、1D
散射扫描耗时。数字随机器变化 — 关注的是同一台机器上的版本间对比。
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from autoquantum import __version__
from autoquantum.dynamics.wavepacket_2d import (
    WavePacket2D, WavePacket2DPropagator, WavePacket2DScan, eckart_product_mask,
    harmonic_ground_width, h3_reduced_masses,
)
from autoquantum.pes.eckart import EckartBuilder
from autoquantum.nn.train import NNTrainer, TrainingConfig


def bench_wavepacket_step(quick: bool):
    mass_R, mass_r = h3_reduced_masses()
    builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                             "r0": 1.401, "coupling": 0.08})
    n = (96, 64) if quick else (192, 144)
    R = np.linspace(0.5, 9.0, n[0])
    r = np.linspace(0.5, 3.5, n[1])
    prop = WavePacket2DPropagator(builder.evaluate_2d, R, r, mass_R, mass_r,
                                 0.5, cap_edges=("R_min", "R_max"))
    psi = WavePacket2D(6.0, 1.241, 0.5,
                       harmonic_ground_width(0.5, mass_r)).initialize(R, r)
    steps = 30 if quick else 120
    prop.step(psi)  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        psi = prop.step(psi)
    dt = (time.perf_counter() - t0) / steps
    return {
        "grid": f"{n[0]}x{n[1]}",
        "seconds_per_step": round(dt, 5),
        "steps_per_second": round(1 / dt, 1),
    }


def bench_energy_scan(quick: bool):
    mass_R, mass_r = h3_reduced_masses()
    builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                             "r0": 1.401, "coupling": 0.0})
    n = (96, 64) if quick else (160, 96)
    R = np.linspace(0.5, 9.0, n[0])
    r = np.linspace(0.5, 3.5, n[1])
    prop = WavePacket2DPropagator(builder.evaluate_2d, R, r, mass_R, mass_r,
                                 0.5, cap_edges=("R_min", "R_max"))
    packet = WavePacket2D(6.0, 1.401, 0.5,
                          harmonic_ground_width(0.5, mass_r))
    steps = 400 if quick else 1200
    scan = WavePacket2DScan(prop, packet, eckart_product_mask(2.0),
                            ("R_min",),
                            reactant_mask=lambda Rr, rr: np.asarray(Rr) >= 2.0,
                            n_steps=steps)
    t0 = time.perf_counter()
    scan.run(0.01, 0.04, 2)
    total = time.perf_counter() - t0
    return {
        "grid": f"{n[0]}x{n[1]}, {steps} steps/point",
        "seconds_per_point": round(total / 2, 4),
    }


def bench_nn_epoch(quick: bool):
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41),
                         indexing="ij")
    X = np.column_stack([g1.ravel(), g2.ravel()])
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] ** 2
    dY = np.column_stack([2 * np.cos(2 * X[:, 0]), 0.6 * X[:, 1]])
    epochs = 20 if quick else 100
    t0 = time.perf_counter()
    NNTrainer(TrainingConfig(hidden_layers=[64, 64, 32], epochs=epochs,
                             lr=0.005, force_weight=1.0, seed=0)
              ).train(X, y, dY=dY)
    total = time.perf_counter() - t0
    return {
        "config": "2D 1681 点, [64,64,32], force training",
        "seconds_per_epoch": round(total / epochs, 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    results = {
        "autoquantum": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "wavepacket_step": bench_wavepacket_step(args.quick),
        "energy_scan": bench_energy_scan(args.quick),
        "nn_epoch": bench_nn_epoch(args.quick),
    }
    for name in ("wavepacket_step", "energy_scan", "nn_epoch"):
        print(f"{name}: {json.dumps(results[name], ensure_ascii=False)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
