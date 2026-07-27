"""二维量子反应散射框架 — 基于 R 和 r 坐标的 H + H₂ 共线反应。"""

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
        n_R, n_r = self.n_R, self.n_r
        V = np.zeros((n_R, n_r))

        for i in range(n_R):
            for j in range(n_r):
                r_AB = self.r_grid[j]
                r_BC = self.R_grid[i]
                V[i, j] = self.pes(r_AB, r_BC)

        min_V = V.min()
        E_eff = E - min_V

        if E_eff <= 0:
            return 0.0

        k2_R = 2 * self.reduced_mass_R * np.maximum(E_eff - V, 0)

        psi = np.zeros((n_R, n_r), dtype=complex)
        psi[0, :] = 1.0
        psi[1, :] = np.exp(1j * np.sqrt(np.maximum(k2_R[1, :], 0)) * self.dR)

        for i in range(1, n_R - 1):
            for j in range(1, n_r - 1):
                T = 2.0 + self.dR ** 2 * k2_R[i, j]
                psi[i + 1, j] = T * psi[i, j] - psi[i - 1, j]

        n_react = n_R // 4
        k_R = np.sqrt(np.mean(k2_R[-n_react:, :]))

        if k_R > 0:
            A_inc = np.mean(np.abs(psi[n_react, :]))
            A_trans = np.mean(np.abs(psi[-1, :]))
            prob = (A_trans / A_inc) ** 2 if A_inc > 0 else 0.0
        else:
            prob = 0.0

        return float(np.clip(prob, 0, 1))
