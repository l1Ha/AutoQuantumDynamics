import numpy as np
from typing import Optional, Callable, Tuple


class WavePacket1D:
    def __init__(self, x0: float, p0: float, sigma: float):
        self.x0 = x0
        self.p0 = p0
        self.sigma = sigma

    def initialize(self, grid: np.ndarray) -> np.ndarray:
        psi = np.exp(-0.5 * ((grid - self.x0) / self.sigma) ** 2)
        psi *= np.exp(1j * self.p0 * (grid - self.x0))
        psi /= np.sqrt(np.trapz(np.abs(psi) ** 2, grid))
        return psi


class SplitOperatorPropagator:
    def __init__(self, mass: float, pes: Callable,
                 grid: np.ndarray, dt: float):
        self.mass = mass
        self.pes = pes
        self.grid = grid
        self.dt = dt
        self.dx = grid[1] - grid[0]

        V = pes(grid)
        self.T_half = np.exp(-0.5j * dt * V)
        self.V_half = None

    def _compute_kinetic(self, psi: np.ndarray) -> np.ndarray:
        k = 2 * np.pi * np.fft.fftfreq(self.grid.size, self.dx)
        T = 0.5 * (k ** 2) / self.mass
        return np.fft.ifft(np.exp(-1j * self.dt * T) * np.fft.fft(psi))

    def step(self, psi: np.ndarray) -> np.ndarray:
        psi = self.T_half * psi
        psi = self._compute_kinetic(psi)
        psi = self.T_half * psi
        return psi

    def propagate(self, psi: np.ndarray, n_steps: int,
                  save_every: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        times = np.arange(0, n_steps * self.dt, self.dt * save_every)
        n_save = len(times)
        psi_all = np.zeros((n_save, self.grid.size), dtype=complex)

        psi_all[0] = psi
        for i in range(1, n_steps):
            psi = self.step(psi)
            if i % save_every == 0:
                psi_all[i // save_every] = psi

        return times, psi_all
