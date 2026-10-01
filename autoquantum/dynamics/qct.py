"""准经典轨线 (QCT) — 经典轨迹在 Born-Oppenheimer 势能面上的传播。

QCT 是含时量子波包的经典极限对照:

- **高能端** (E >> E_barrier): QCT 与量子结果一致 — 核运动近似经典;
- **低能端** (E ~ E_barrier): 量子效应 (隧穿、零点能、干涉) 使量子
  P_react **高于** QCT — 这是 QCT 方法的系统偏差;
- **计算代价**: O(N_traj × N_steps) 而非 O(N_grid² × N_steps)，
  对高维体系 (N_dim > 2) 尤其有优势。

物理模型: 共线 H + H₂ (2D Jacobi 坐标)。
坐标 (R, r) 对应波包代码的 (R, r)，质量 (μ_R, μ_r) = (2m/3, m/2)。
Hamilton 方程用 Velocity Verlet (辛算法) 积分。

单位约定: ħ = 1, Hartree, Bohr, 质量以 mₑ 为单位。

参考文献:
  - Porter & Karplus, J. Chem. Phys. 40, 1105 (1964)
  - Truhlar & Muckerman, in "Atom-Molecule Collision Theory" (1979)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def wigner_sample(mu: float, omega: float, n: int,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """谐振子基态 Wigner 分布采样 (位置 q 和动量 p)。

    W(q,p) ∝ exp(−q²/σ_q²) · exp(−p²/σ_p²), 其中
    σ_q = 1/√(2μω), σ_p = √(μω/2); q·p 满足最小不确定性。
    """
    rng = np.random.RandomState(seed)
    sigma_q = 1.0 / np.sqrt(2.0 * mu * omega)
    sigma_p = np.sqrt(mu * omega / 2.0)
    q = rng.normal(0.0, sigma_q, n)
    p = rng.normal(0.0, sigma_p, n)
    return q, p


# ---------------------------------------------------------------------------
# 单条轨迹 (2D 共线: 坐标 (R, r), 动量 (P_R, p_r))
# ---------------------------------------------------------------------------

class QCTTrajectory:
    """单条经典轨迹: Velocity Verlet 积分在 V(R, r) 上。"""

    def __init__(self, pes: Callable, mass_R: float, mass_r: float,
                 dt: float = 0.5):
        """
        Parameters
        ----------
        pes : Callable(R, r) → float (Hartree)
        mass_R, mass_r : 约化质量 (mₑ)
        dt : 时间步长 (au)
        """
        self.pes = pes
        self.mass_R = mass_R
        self.mass_r = mass_r
        self.dt = dt

    def _grad(self, R: float, r: float) -> Tuple[float, float]:
        """中心差分 ∂V/∂R, ∂V/∂r。"""
        h = 1e-5
        dV_dR = (self.pes(R + h, r) - self.pes(R - h, r)) / (2 * h)
        dV_dr = (self.pes(R, r + h) - self.pes(R, r - h)) / (2 * h)
        return dV_dR, dV_dr

    def propagate(self, R: float, r: float,
                  P_R: float, p_r: float,
                  n_steps: int,
                  reaction_criterion: Optional[Callable] = None,
                  R_refl_threshold: Optional[float] = None,
                  ) -> dict:
        """传播轨迹。

        Parameters
        ----------
        R, r : 初始坐标 (Bohr)
        P_R, p_r : 初始动量 (au)
        n_steps : 最大步数
        reaction_criterion : Callable(R, r) -> bool, 判定是否进入反应产物区
        R_refl_threshold : float, 可选反弹阈值 (当 R > R_refl 且 P_R > 0 时提早终止未反应轨迹)

        Returns
        -------
        dict with keys: reacted, R_final, r_final, energy_drift, steps_used
        """
        m_R, m_r = self.mass_R, self.mass_r
        V0 = self.pes(R, r)
        e_init = V0 + 0.5 * P_R ** 2 / m_R + 0.5 * p_r ** 2 / m_r
        dV_dR, dV_dr = self._grad(R, r)
        reacted = False
        steps_used = n_steps

        for step in range(n_steps):
            # Velocity Verlet: F = -∂V/∂R, 动量更新用力 (非梯度)
            P_R -= 0.5 * self.dt * dV_dR
            p_r -= 0.5 * self.dt * dV_dr
            R += self.dt * P_R / m_R
            r += self.dt * p_r / m_r
            dV_dR, dV_dr = self._grad(R, r)
            P_R -= 0.5 * self.dt * dV_dR
            p_r -= 0.5 * self.dt * dV_dr

            if reaction_criterion is not None and reaction_criterion(R, r):
                reacted = True
                steps_used = step + 1
                break

            # 若已反弹并离开相互作用区, 提早退出
            if R_refl_threshold is not None and R > R_refl_threshold and P_R > 0:
                steps_used = step + 1
                break

        e_final = self.pes(R, r) + 0.5 * P_R ** 2 / m_R + 0.5 * p_r ** 2 / m_r
        drift = abs(e_final - e_init) / max(abs(e_init), 1e-30)
        return {"reacted": reacted, "R_final": R, "r_final": r,
                "energy_drift": drift, "steps_used": steps_used}


# ---------------------------------------------------------------------------
# 轨迹系综
# ---------------------------------------------------------------------------

@dataclass
class QCTResult:
    """QCT 轨迹系综结果 (与 ScatteringResult2D 接口兼容)。"""
    energies: np.ndarray
    reaction_probs: np.ndarray
    n_traj_per_energy: int
    energy_drift_max: float

    @property
    def transmission(self) -> np.ndarray:
        return self.reaction_probs


class QCTEnsemble:
    """QCT 轨迹系综 — 每个碰撞能运行 N_traj 条轨迹计算 P_react。

    初始条件: R = R0 (渐近区), P_R = −√(2μ_R·E_coll) (碰撞动量),
    r 和 p_r 从谐振子基态 Wigner 分布采样 (振动态量子期望)。
    """

    def __init__(self, pes: Callable, mass_R: float, mass_r: float,
                 dt: float = 0.5, max_steps: int = 5000):
        self.pes = pes
        self.mass_R = mass_R
        self.mass_r = mass_r
        self.dt = dt
        self.max_steps = max_steps

    def run(self, E_grid: np.ndarray, R0: float,
            r_mean: float, r_sigma: float,
            reaction_criterion: Callable,
            n_traj: int = 200, seed: int = 0,
            omega: Optional[float] = None) -> QCTResult:
        """对每个碰撞能运行轨迹系综。

        Parameters
        ----------
        E_grid : 碰撞能网格 (Hartree)
        R0 : 反应物初始中心距 (Bohr)
        r_mean : 双原子键长平衡位置 (Bohr)
        r_sigma : 双原子振幅高斯宽度 σ_r (Bohr)
        reaction_criterion : Callable(R, r) → bool
            返回 True 时标记为反应 (如 R < 1.5·r 交换分界面)。
        n_traj : 每个能量点的轨迹数
        seed : 随机数种子
        omega : 可选振动频率 (若未指定, 则由 σ_r = 1/√(μ_r·ω) 推导)
        """
        mass_R = self.mass_R
        mass_r = self.mass_r
        traj = QCTTrajectory(self.pes, mass_R, mass_r, dt=self.dt)
        energies = np.asarray(E_grid, dtype=float)
        reaction_probs = np.zeros(energies.size)
        drift_max = 0.0
        n_reacted_all = []

        if omega is None:
            omega = 1.0 / (mass_r * r_sigma ** 2)

        for ie, E in enumerate(energies):
            p_R0 = -np.sqrt(2.0 * mass_R * E)
            rng = np.random.RandomState(seed + ie * 1000)
            n_react = 0
            for _ in range(n_traj):
                # 物理 Wigner 分布采样: q 为相对平衡位置位移, p 为物理共轭动量
                r0_arr, pr0_arr = wigner_sample(mass_r, omega, 1, seed=rng.randint(1 << 30))
                r0 = float(r0_arr[0]) + r_mean
                pr0 = float(pr0_arr[0])
                result = traj.propagate(
                    float(R0), r0, float(p_R0), pr0, self.max_steps,
                    reaction_criterion=reaction_criterion,
                    R_refl_threshold=float(R0 + 0.5))
                if result["reacted"]:
                    n_react += 1
                drift_max = max(drift_max, float(result["energy_drift"]))
            reaction_probs[ie] = n_react / n_traj
            n_reacted_all.append(n_react)

        return QCTResult(energies=energies, reaction_probs=reaction_probs,
                         n_traj_per_energy=n_traj, energy_drift_max=float(drift_max))
