import numpy as np
from typing import Callable

from autoquantum.core.base import DynamicsResult


class QuantumScattering1D:
    def __init__(self, mass: float, pes: Callable,
                 n_grid: int = 1000, grid_min: float = 0.3,
                 grid_max: float = 5.0):
        self.mass = mass
        self.pes = pes
        self.n_grid = n_grid
        self.grid_min = grid_min
        self.grid_max = grid_max

    def solve(self, energy_min: float, energy_max: float,
              n_points: int = 100) -> DynamicsResult:
        energies = np.linspace(energy_min, energy_max, n_points)
        grid = np.linspace(self.grid_min, self.grid_max, self.n_grid)
        dx = grid[1] - grid[0]
        V = np.asarray(self.pes(grid), dtype=float)

        n_left = max(10, self.n_grid // 12)
        n_right = max(10, self.n_grid // 12)

        v_left = np.mean(V[:n_left])
        v_right = np.mean(V[-n_right:])

        transmission = np.zeros(n_points)
        reflection = np.zeros(n_points)

        for i, E in enumerate(energies):
            T, R = self._solve_energy(E, V, dx, n_left, n_right, v_left, v_right)
            transmission[i] = T
            reflection[i] = R

        return DynamicsResult(
            energy=energies,
            transmission=transmission,
            reflection=reflection,
            grid=grid,
        )

    def _solve_energy(self, E, V, dx, n_left, n_right, v_left, v_right):
        N = self.n_grid
        ek_left = max(E - v_left, 1e-30)
        ek_right = max(E - v_right, 1e-30)
        k_left = np.sqrt(2.0 * self.mass * ek_left)
        k_right = np.sqrt(2.0 * self.mass * ek_right)

        k2 = 2.0 * self.mass * (E - V)
        k2_pad = np.pad(k2, (0, 1), mode="edge")

        psi = np.zeros(N + 1, dtype=complex)
        psi[N] = 1.0
        psi[N - 1] = np.exp(-1j * k_right * dx)

        h2 = dx * dx
        for j in range(N - 1, 1, -1):
            denom = 1.0 + h2 * k2_pad[j - 1] / 12.0
            if abs(denom) < 1e-30:
                continue
            psi[j - 1] = ((2.0 - 5.0 * h2 * k2_pad[j] / 6.0) * psi[j]
                          - (1.0 + h2 * k2_pad[j + 1] / 12.0) * psi[j + 1]) / denom

        psi_left = psi[n_left]
        psi_left_next = psi[n_left + 1]

        exp_ikdx = np.exp(1j * k_left * dx)
        exp_mikdx = np.exp(-1j * k_left * dx)
        sin_kdx = np.sin(k_left * dx)

        if abs(sin_kdx) < 1e-15 or k_left < 1e-15:
            return 0.0, 1.0

        A = (psi_left_next - psi_left * exp_mikdx) / (exp_ikdx - exp_mikdx)
        B = psi_left - A

        if abs(A) < 1e-30:
            return 0.0, 1.0

        T = (k_right / k_left) * (abs(psi[N]) / abs(A)) ** 2
        R = abs(B / A) ** 2

        total = T + R
        if total > 1e-15:
            T /= total
            R /= total
        else:
            T, R = 0.0, 1.0

        return float(np.clip(T, 0, 1)), float(np.clip(R, 0, 1))
