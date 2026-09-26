"""二维波包传播的 PyTorch (GPU) 后端 — 与 NumPy 版逐项对拍的可选加速。

何时使用: 网格 ≥ 256×256 或需要大批量能量扫描时, A100 等数据中心
GPU 对 FFT 主导的传播子有显著加速 (float32 下通常 5-20 倍)。
正确性: 与 NumPy 版在同一基准上透射率一致 (float64 ~1e-6, 
float32 ~1e-4, 见 tests/test_gpu_parity.py), 概率记账恒等式同满足。

依赖: torch (可选)。CPU 上的 torch 也可运行 (小规模测试用)。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

from autoquantum.dynamics.wavepacket_2d import (
    WavePacket2D, WavePacket2DResult)


class TorchWavePacket2DPropagator:
    """WavePacket2DPropagator 的 torch 后端 (API 兼容子集)。

    复势对称分裂 W = V − iη: W/2 → T → W/2, T 在 k 空间对角,
    CAP 逐边归因吸收 (与 NumPy 版相同的确定性方案)。
    """

    EDGES = ("R_min", "R_max", "r_min", "r_max")

    def __init__(self, pes: Callable, R_grid: np.ndarray, r_grid: np.ndarray,
                 mass_R: float, mass_r: float, dt: float,
                 cap_edges: Tuple[str, ...] = EDGES,
                 cap_width_frac: float = 0.15, cap_height: float = 0.1,
                 dtype: str = "float64", device: Optional[str] = None):
        if not HAS_TORCH:
            raise RuntimeError("需要安装 torch: pip install torch")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                              else "cpu"))
        self.tdtype = {"float32": torch.complex64,
                       "float64": torch.complex128}[dtype]
        self.real_t = {"float32": torch.float32,
                       "float64": torch.float64}[dtype]

        self.R_grid = np.asarray(R_grid, dtype=float)
        self.r_grid = np.asarray(r_grid, dtype=float)
        self.n_R, self.n_r = self.R_grid.size, self.r_grid.size
        self.dR = float(self.R_grid[1] - self.R_grid[0])
        self.dr = float(self.r_grid[1] - self.r_grid[0])
        self.dV = self.dR * self.dr
        self.dt = float(dt)
        self.mass_R, self.mass_r = float(mass_R), float(mass_r)

        RR, rr = np.meshgrid(self.R_grid, self.r_grid, indexing="ij")
        self._mesh = (RR, rr)
        self.V = np.asarray(pes(RR, rr), dtype=float)

        # CAP η (与 NumPy 版同一二次 ramp); 归因表在 _eta_maps 构建后生成
        self._eta_maps = self._eta_maps_for_edges(cap_edges, cap_width_frac,
                                                  cap_height)
        self.eta_total = sum(self._eta_maps[e] for e in cap_edges)
        self._assign = np.argmax(np.stack([self._eta_maps[e] for e in cap_edges]),
                                 axis=0)

        def T(a):
            return torch.as_tensor(a, device=self.device)

        V_eff = T(self.V) - 1j * T(self.eta_total)
        self.exp_W_half = torch.exp(-0.5j * T(dt) * V_eff).to(self.tdtype)

        kR = 2 * np.pi * np.fft.fftfreq(self.n_R, self.dR)[:, None]
        kr = 2 * np.pi * np.fft.fftfreq(self.n_r, self.dr)[None, :]
        t_k = 0.5 * (kR ** 2 / mass_R + kr ** 2 / mass_r)
        self.exp_T = torch.exp(-1j * T(dt) * T(t_k)).to(self.tdtype)

        # 逐格点 CAP 归因 (与 NumPy 版相同的 argmax 方案; bincount 用 CPU)
        self._eta_maps = self._eta_maps_for_edges(cap_edges, cap_width_frac,
                                                 cap_height)
        self._assign_flat = np.asarray(self._assign, dtype=np.int64).ravel()
        self.cap_edges = tuple(cap_edges)
        self.last_absorbed: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _eta_maps_for_edges(self, cap_edges, frac, height) -> Dict[str, np.ndarray]:
        maps = {}
        for edge in cap_edges:
            axis, side = edge.split("_")
            n = self.n_R if axis == "R" else self.n_r
            n_cap = int(frac * n)
            idx = np.arange(min(n_cap, n // 2))
            ramp = np.zeros(n)
            if side == "min":
                ramp[:len(idx)] = ((len(idx) - idx) / len(idx)) ** 2
            else:
                ramp[-len(idx):] = ((idx + 1) / len(idx)) ** 2
            if axis == "R":
                maps[edge] = (height * ramp)[:, None] * np.ones((1, self.n_r))
            else:
                maps[edge] = (height * ramp)[None, :] * np.ones((self.n_R, 1))
        return maps

    # ------------------------------------------------------------------
    def initialize(self, packet: WavePacket2D) -> "torch.Tensor":
        psi_np = packet.initialize(self.R_grid, self.r_grid)
        return torch.as_tensor(psi_np, device=self.device).to(self.tdtype)

    def _losses(self, p2_before, p2_after) -> np.ndarray:
        d = (p2_before - p2_after) * self.dV
        return np.bincount(self._assign_flat, weights=d.ravel(),
                           minlength=len(self.cap_edges))

    def step(self, psi: "torch.Tensor") -> "torch.Tensor":
        losses = np.zeros(len(self.cap_edges))
        with torch.no_grad():
            if self.cap_edges:
                p2 = (psi.abs() ** 2).cpu().numpy()
            psi = self.exp_W_half * psi
            psi = torch.fft.ifft2(self.exp_T * torch.fft.fft2(psi))
            if self.cap_edges:
                p2n = (psi.abs() ** 2).cpu().numpy()
                losses += self._losses(p2, p2n)
                p2 = p2n
            psi = self.exp_W_half * psi
            if self.cap_edges:
                p2n = (psi.abs() ** 2).cpu().numpy()
                losses += self._losses(p2, p2n)
        self.last_absorbed = {e: float(v)
                              for e, v in zip(self.cap_edges, losses)}
        return psi

    # ------------------------------------------------------------------
    def propagate(self, psi0: "torch.Tensor", n_steps: int,
                  save_every: int = 10,
                  product_mask: Optional[Callable] = None,
                  product_edges: Tuple[str, ...] = (),
                  reactant_mask: Optional[Callable] = None,
                  save_density: bool = True,
                  track_energy: bool = False) -> WavePacket2DResult:
        for e in product_edges:
            if e not in self.cap_edges:
                raise ValueError(f"product_edges 的 {e} 未启用 CAP")

        save_steps = list(range(0, n_steps + 1, save_every))
        if save_steps[-1] != n_steps:
            save_steps.append(n_steps)
        save_index = {s: k for k, s in enumerate(save_steps)}
        times = np.array(save_steps, dtype=float) * self.dt
        n_save = len(save_steps)

        RR, rr = self._mesh
        prod_mask = product_mask(RR, rr) if product_mask is not None else None
        react_mask = reactant_mask(RR, rr) if reactant_mask is not None else None

        snapshots = (np.zeros((n_save, self.n_R, self.n_r))
                     if save_density else None)
        norm_t = np.zeros(n_save)
        prod_pop = np.zeros(n_save)
        react_pop = np.zeros(n_save)
        energies = np.zeros(n_save) if track_energy else None
        absorbed = {e: np.zeros(n_save) for e in self.cap_edges}
        cumulative = {e: 0.0 for e in self.cap_edges}

        def to_np(psi):
            return (psi.abs() ** 2).cpu().numpy() * self.dV

        def record(k, psi):
            p2 = to_np(psi)
            norm_t[k] = p2.sum()
            if prod_mask is not None:
                prod_pop[k] = p2[prod_mask].sum()
            if react_mask is not None:
                react_pop[k] = p2[react_mask].sum()
            for e in absorbed:
                absorbed[e][k] = cumulative[e]
            if save_density:
                snapshots[k] = p2 / self.dV
            if track_energy:
                energies[k] = self.energy_expectation(psi)

        psi = psi0.clone()
        record(0, psi)
        for i in range(1, n_steps + 1):
            psi = self.step(psi)
            for e, d in self.last_absorbed.items():
                cumulative[e] += d
            if i in save_index:
                record(save_index[i], psi)

        reaction = prod_pop + sum(absorbed[e] for e in product_edges
                                  if e in absorbed)
        reflection = react_pop + sum(absorbed[e] for e in self.cap_edges
                                     if e not in product_edges)
        drift = (float(np.max(np.abs(energies - energies[0])) / abs(energies[0]))
                 if track_energy and energies[0] != 0 else None)
        return WavePacket2DResult(
            R_grid=self.R_grid, r_grid=self.r_grid, V_grid=self.V,
            times=times, snapshots=snapshots, norm_t=norm_t, absorbed=absorbed,
            product_population=prod_pop, reactant_population=react_pop,
            reaction_prob=reaction, reflection_prob=reflection,
            energy_track=energies, energy_drift_rel=drift,
            energy_reference=float(energies[0]) if track_energy else None)

    # ------------------------------------------------------------------
    def energy_expectation(self, psi: "torch.Tensor") -> float:
        with torch.no_grad():
            kR = 2 * np.pi * np.fft.fftfreq(self.n_R, self.dR)[:, None]
            kr = 2 * np.pi * np.fft.fftfreq(self.n_r, self.dr)[None, :]
            t_k = 0.5 * (kR ** 2 / self.mass_R + kr ** 2 / self.mass_r)
            t_t = torch.as_tensor(t_k, device=self.device).to(self.tdtype)
            v_t = torch.as_tensor(self.V, device=self.device).to(self.tdtype)
            rho_k = (torch.fft.fft2(psi).abs() ** 2)
            rho_x = (psi.abs() ** 2)
            t_mean = float((t_t * rho_k).sum().cpu() / rho_k.sum().cpu())
            v_mean = float((v_t * rho_x).sum().cpu() / rho_x.sum().cpu())
        return t_mean + v_mean

    def check_initial_overlap(self, psi0: "torch.Tensor") -> float:
        if not self.cap_edges:
            return 0.0
        cap = self.eta_total > 0
        p2 = (psi0.abs() ** 2).cpu().numpy() * self.dV
        return float(p2[cap].sum())

    def expectation_R(self, psi: "torch.Tensor") -> float:
        p2 = (psi.abs() ** 2).cpu().numpy() * self.dV
        return float(np.sum(p2 * self._mesh[0]))
