"""二维量子反应散射框架 — 基于 R 和 r 坐标的 H + H₂ 共线反应。

.. warning:: 实验性求解器

    本模块沿 R 方向做单通道中心差分递推，无横向动能算符、无通道
    耦合，概率提取也未做流归一化。输出只反映势垒形状的定性趋势，
    不能作为定量散射结果使用。定量计算请用含时波包
    ``wavepacket_2d.WavePacket2DPropagator``。

时间无关的定能量扫描: V[i, j] = pes(R_i, r_j)。
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from autoquantum.core.base import DynamicsResult


@dataclass
class ScatteringResult2D:
    energy: np.ndarray
    transmission: np.ndarray
    reaction_prob: np.ndarray
    R_grid: np.ndarray
    r_grid: np.ndarray


class QuantumReaction2D:
    def __init__(self, mass_H: float, pes: Callable,
                 n_R: int = 200, n_r: int = 200,
                 R_range: Tuple[float, float] = (0.5, 6.0),
                 r_range: Tuple[float, float] = (0.5, 6.0)):
        self.mass = mass_H
        self.mass_H2 = 2 * mass_H
        self.reduced_mass_R = 2 * mass_H / 3
        self.reduced_mass_r = mass_H / 2
        self.pes = pes

        self.n_R = n_R
        self.n_r = n_r
        self.R_grid = np.linspace(*R_range, n_R)
        self.r_grid = np.linspace(*r_range, n_r)
        self.dR = self.R_grid[1] - self.R_grid[0]
        self.dr = self.r_grid[1] - self.r_grid[0]

        RR, rr = np.meshgrid(self.R_grid, self.r_grid, indexing="ij")
        try:
            V = np.asarray(self.pes(RR, rr), dtype=float)
            if V.shape != RR.shape:
                raise ValueError
        except Exception:
            V = np.array([[float(self.pes(R, r)) for r in self.r_grid]
                          for R in self.R_grid])
        self.V = V

    def solve(self, energy_min: float, energy_max: float,
              n_points: int = 50) -> ScatteringResult2D:
        energies = np.linspace(energy_min, energy_max, n_points)

        reaction_prob = np.zeros(n_points)
        transmission = np.zeros(n_points)

        for i, E in enumerate(energies):
            prob = self._solve_energy(E)
            reaction_prob[i] = prob
            transmission[i] = prob

        return ScatteringResult2D(
            energy=energies,
            transmission=transmission,
            reaction_prob=reaction_prob,
            R_grid=self.R_grid,
            r_grid=self.r_grid,
        )

    def _solve_energy(self, E: float) -> float:
        """单能量解 (实验性)。

        沿 R 的三点中心差分递推 ψ_{i+1} = (2 - h²k²)ψ_i - ψ_{i-1}，
        k² = 2μ(E-V) 允许为负 (虚指数 → 隧穿趋势的衰减解)。
        概率由入射/出射参考截面的平均振幅比估计 — 未做流归一化，
        结果是定性/演示量级，不宜作定量结论 (见模块 docstring)。
        """
        n_R, n_r = self.n_R, self.n_r
        V = self.V

        k2_R = 2 * self.reduced_mass_R * (E - V)

        psi = np.zeros((n_R, n_r), dtype=complex)
        psi[0, :] = 1.0
        psi[1, :] = np.exp(1j * np.sqrt(k2_R[1, :].astype(complex)) * self.dR)

        h2 = self.dR ** 2
        for i in range(1, n_R - 1):
            T = 2.0 - h2 * k2_R[i, :]
            psi[i + 1, 1:-1] = T[1:-1] * psi[i, 1:-1] - psi[i - 1, 1:-1]

        n_react = n_R // 4
        k_out2 = np.mean(k2_R[-n_react:, :])
        if k_out2 <= 0:
            return 0.0

        A_inc = np.mean(np.abs(psi[n_react, :]))
        A_trans = np.mean(np.abs(psi[-1, :]))
        if A_inc <= 0:
            return 0.0
        prob = (A_trans / A_inc) ** 2
        if not np.isfinite(prob):
            return 0.0
        return float(prob)
