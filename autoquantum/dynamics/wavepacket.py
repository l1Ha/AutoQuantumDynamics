import numpy as np
from typing import Optional, Callable, Tuple


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """numpy 2.x 移除了 np.trapz，此处做版本兼容。"""
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = np.trapz
    return trap(y, x)


class WavePacket1D:
    def __init__(self, x0: float, p0: float, sigma: float):
        self.x0 = x0
        self.p0 = p0
        self.sigma = sigma

    def initialize(self, grid: np.ndarray) -> np.ndarray:
        psi = np.exp(-0.5 * ((grid - self.x0) / self.sigma) ** 2).astype(complex)
        psi *= np.exp(1j * self.p0 * (grid - self.x0))
        psi /= np.sqrt(trapezoid(np.abs(psi) ** 2, grid))
        return psi


class SplitOperatorPropagator:
    """一维 Split-Operator (FFT) 传播子。

    ψ(t+dt) = e^{-iV·dt/2} · F⁻¹[e^{-iT·dt} · F(e^{-iV·dt/2} ψ)]

    可选网格边缘二次型吸收势 (CAP)，防止 FFT 周期回绕，
    并按左右边归因吸收量 (self.last_absorbed)。
    """

    def __init__(self, mass: float, pes: Callable,
                 grid: np.ndarray, dt: float,
                 cap_width_frac: float = 0.15,
                 cap_height: float = 0.1,
                 cap_edges: Tuple[str, ...] = ("left", "right")):
        self.mass = mass
        self.pes = pes
        self.grid = grid
        self.dt = dt
        self.dx = grid[1] - grid[0]

        V = np.asarray(pes(grid), dtype=float)
        self.exp_V_half = np.exp(-0.5j * dt * V)

        k = 2 * np.pi * np.fft.fftfreq(self.grid.size, self.dx)
        T = 0.5 * (k ** 2) / self.mass
        self.exp_T = np.exp(-1j * dt * T)

        n = self.grid.size
        n_cap = int(cap_width_frac * n)
        ramps = {
            "left": cap_height * _low_ramp_1d(n, n_cap),
            "right": cap_height * _high_ramp_1d(n, n_cap),
        }
        self.cap_factors = {
            edge: np.exp(-ramps[edge] * dt) for edge in cap_edges
        }
        self.last_absorbed: dict = {}

    def step(self, psi: np.ndarray) -> np.ndarray:
        psi = self.exp_V_half * psi
        psi = np.fft.ifft(self.exp_T * np.fft.fft(psi))
        psi = self.exp_V_half * psi

        self.last_absorbed = {}
        for edge, factor in self.cap_factors.items():
            damped = psi * factor
            self.last_absorbed[edge] = (
                np.sum(np.abs(psi) ** 2) - np.sum(np.abs(damped) ** 2)
            ) * self.dx
            psi = damped
        return psi

    def propagate(self, psi: np.ndarray, n_steps: int,
                  save_every: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        save_steps = list(range(0, n_steps + 1, save_every))
        if save_steps[-1] != n_steps:
            save_steps.append(n_steps)
        save_index = {s: k for k, s in enumerate(save_steps)}
        times = np.array(save_steps, dtype=float) * self.dt
        psi_all = np.zeros((len(save_steps), self.grid.size), dtype=complex)

        psi_all[0] = psi
        for i in range(1, n_steps + 1):
            psi = self.step(psi)
            if i in save_index:
                psi_all[save_index[i]] = psi

        return times, psi_all


def _low_ramp_1d(n: int, n_cap: int) -> np.ndarray:
    ramp = np.zeros(n)
    if n_cap <= 0:
        return ramp
    n_cap = min(n_cap, n // 2)
    idx = np.arange(n_cap)
    ramp[:n_cap] = ((n_cap - idx) / n_cap) ** 2
    return ramp


def _high_ramp_1d(n: int, n_cap: int) -> np.ndarray:
    ramp = np.zeros(n)
    if n_cap <= 0:
        return ramp
    n_cap = min(n_cap, n // 2)
    idx = np.arange(n_cap)
    ramp[-n_cap:] = ((idx + 1) / n_cap) ** 2
    return ramp
