"""二维含时波包传播 — Split-Operator (FFT) 传播子。

在 Jacobi 坐标 (R, r) 下求解二维含时薛定谔方程:

    iħ ∂ψ/∂t = [T̂ + V(R, r)] ψ
    T̂ = -(ħ²/2μ_R) ∂²/∂R² - (ħ²/2μ_r) ∂²/∂r²

分步傅里叶方法 (Strang 分裂, 二阶精度):

    ψ(t+dt) = e^{-iW·dt/2} · F⁻¹[ e^{-iT·dt} · F( e^{-iW·dt/2} ψ ) ]

其中 W(R,r) = V(R,r) - iη(R,r) 为复势: η ≥ 0 是网格边缘的二次型
复吸收势 (CAP), 以对半分裂的方式同时作用于每步前后两半, 保持
整体二阶精度。吸收量按格点确定性归因到单一 CAP 边 (重叠角点
归 η 最大的边), 与应用顺序无关。

通道记账恒等式 (精确):

    P_grid(t) + Σ_edges L_edge(t) = 1

其中 P_grid 为网格存活概率, L_edge 为各边累计吸收量。
反应/反射通道概率定义为有限时间估计:

    P_react(t) = P_product_mask(t) + Σ_{product edges} L_e(t)
    P_refl(t)  = P_reactant_mask(t) + Σ_{other edges} L_e(t)

若 product/reactant 掩码恰好划分整个网格 (无重叠、无空隙),
则 P_react + P_refl = 1 精确成立; 产物区布居可能再穿越分界面,
故这是有限时间通道估计而非渐近反应产额。

宽度约定: ``WavePacket2D`` 的 σ 为振幅高斯宽度,
|ψ|² ∝ exp(-(Δx/σ)²), 即位置标准差为 σ/√2, 动量标准差为 ħ/(√2σ)。

单位约定: ħ = 1，能量 Hartree，长度 Bohr，质量以电子质量为单位
(m_H ≈ 1836.15)。通过 ``g_matrix`` 参数也支持非对角动能项
(共线三原子键长坐标)。

数值验证 (见 RELEASE_VALIDATION.md):
  - 1D Eckart 垒: 与解析谱透射率加权平均误差 < 4×10⁻⁴;
  - 可分离 2D Eckart + 谐振子: 与解析解误差 1.2×10⁻⁵;
  - 概率恒等式在所有算例中满足至 ~10⁻⁹。

参考文献:
  - Kosloff, R. J. Phys. Chem. 92, 2087 (1988)
  - Neuhauser, D.; Baer, M. J. Chem. Phys. 90, 4351 (1989)
  - Schatz, G. C. Rev. Mod. Phys. 61, 669 (1989)
"""

import numpy as np
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field

from autoquantum.dynamics.quantum_2d import ScatteringResult2D

# 氢原子质量 (电子质量单位)
DEFAULT_MASS_H = 1836.15


# ---------------------------------------------------------------------------
# 初始波包
# ---------------------------------------------------------------------------

@dataclass
class WavePacket2D:
    """二维高斯波包初始条件。

    psi ∝ exp(-0.5·((R-R0)/σ_R)² - 0.5·((r-r0)/σ_r)²) · exp(i(p_R·ΔR + p_r·Δr))

    σ 为振幅宽度: 位置标准差 σ/√2，动量标准差 1/(√2σ)，
    纵向能量分辨率 ΔE ≈ 1/(4μσ²) (σ→σ_R)。
    """

    R0: float
    r0: float
    sigma_R: float
    sigma_r: float
    p_R0: float = 0.0
    p_r0: float = 0.0

    def initialize(self, R_grid: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
        RR, rr = np.meshgrid(R_grid, r_grid, indexing="ij")
        psi = np.exp(-0.5 * ((RR - self.R0) / self.sigma_R) ** 2
                     - 0.5 * ((rr - self.r0) / self.sigma_r) ** 2)
        psi = psi.astype(complex)
        psi *= np.exp(1j * (self.p_R0 * (RR - self.R0)
                            + self.p_r0 * (rr - self.r0)))
        dV = (R_grid[1] - R_grid[0]) * (r_grid[1] - r_grid[0])
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dV)
        return psi


# ---------------------------------------------------------------------------
# 传播结果
# ---------------------------------------------------------------------------

@dataclass
class WavePacket2DResult:
    """含时波包传播结果。

    恒等式: norm_t + Σ absorbed = 1 (精确)。
    reaction_prob + reflection_prob = 1 当且仅当 product/reactant
    掩码划分整个网格; 二者为有限时间通道估计 (产物布居可再穿越)。
    """

    R_grid: np.ndarray
    r_grid: np.ndarray
    V_grid: np.ndarray
    times: np.ndarray
    snapshots: Optional[np.ndarray]          # (n_save, n_R, n_r) |ψ|²
    norm_t: np.ndarray                       # 时刻 t 网格上的存活概率
    absorbed: Dict[str, np.ndarray]          # 各 CAP 边的累计吸收概率
    product_population: np.ndarray           # 产物区布居 P_prod(t)
    reactant_population: np.ndarray          # 反应物区布居 P_react_ch(t)
    reaction_prob: np.ndarray                # 通道估计 P_react(t)
    reflection_prob: np.ndarray              # 通道估计 P_refl(t)
    energy_track: Optional[np.ndarray] = None   # track_energy=True 时为 ⟨H⟩(t)
    energy_drift_rel: Optional[float] = None    # |⟨H⟩-⟨H⟩₀|/|⟨H⟩₀|
    energy_reference: Optional[float] = None
    metadata: Optional[Dict[str, object]] = None

    @property
    def transmission(self) -> np.ndarray:
        """与 Dashboard / 一维扫描接口兼容的透射概率。"""
        return self.reaction_prob

    @property
    def final_reaction_probability(self) -> float:
        return float(self.reaction_prob[-1])

    @property
    def snapshot_times(self) -> np.ndarray:
        return self.times


# ---------------------------------------------------------------------------
# 吸收边界辅助
# ---------------------------------------------------------------------------

def _low_ramp(n: int, n_cap: int) -> np.ndarray:
    """低指数端二次吸收 ramp: 边界处为 1，向内平滑降为 0。"""
    ramp = np.zeros(n)
    if n_cap <= 0:
        return ramp
    n_cap = min(n_cap, n // 2)
    idx = np.arange(n_cap)
    ramp[:n_cap] = ((n_cap - idx) / n_cap) ** 2
    return ramp


def _high_ramp(n: int, n_cap: int) -> np.ndarray:
    """高指数端二次吸收 ramp: 边界处为 1，向内平滑降为 0。"""
    ramp = np.zeros(n)
    if n_cap <= 0:
        return ramp
    n_cap = min(n_cap, n // 2)
    idx = np.arange(n_cap)
    ramp[-n_cap:] = ((idx + 1) / n_cap) ** 2
    return ramp


# ---------------------------------------------------------------------------
# 二维 Split-Operator 传播子
# ---------------------------------------------------------------------------

class WavePacket2DPropagator:
    """二维含时波包 Split-Operator (FFT) 传播子 (复势对称分裂)。

    Parameters
    ----------
    pes : Callable
        势能面函数 V(R, r)，支持广播 (输入二维网格返回同形数组)。
    R_grid, r_grid : np.ndarray
        两个坐标方向的网格 (等间距)。
    mass_R, mass_r : float
        两个方向的约化质量 (电子质量单位)。若提供 ``g_matrix`` 则忽略。
    dt : float
        时间步长。
    g_matrix : (2, 2) array, optional
        动能 g 矩阵 T = (ħ²/2)[g_RR k_R² + 2 g_Rr k_R k_r + g_rr k_r²]，
        默认对角 [[1/μ_R, 0], [0, 1/μ_r]]。注意: 非对角动能时
        ``WavePacket2DScan`` 用 mass_R 设定初始动量的做法并不严格，
        需自行保证能量语义 (见其文档)。
    cap_edges : sequence of str
        启用吸收势的边，取值 "R_min"/"R_max"/"r_min"/"r_max"。
        束缚振动坐标不应启用对应边的 CAP。
    cap_width_frac : float
        每条 CAP 占该方向网格的比例。
    cap_height : float
        CAP 强度 η_max (Hartree)。
    """

    EDGES = ("R_min", "R_max", "r_min", "r_max")

    def __init__(self, pes: Callable,
                 R_grid: np.ndarray, r_grid: np.ndarray,
                 mass_R: float, mass_r: float, dt: float,
                 g_matrix: Optional[np.ndarray] = None,
                 cap_edges: Tuple[str, ...] = EDGES,
                 cap_width_frac: float = 0.15,
                 cap_height: float = 0.1):
        self.pes = pes
        self.R_grid = np.asarray(R_grid, dtype=float)
        self.r_grid = np.asarray(r_grid, dtype=float)
        self.n_R = self.R_grid.size
        self.n_r = self.r_grid.size
        self.dR = self.R_grid[1] - self.R_grid[0]
        self.dr = self.r_grid[1] - self.r_grid[0]
        self.dV = self.dR * self.dr
        self.dt = dt
        self.mass_R = mass_R
        self.mass_r = mass_r

        RR, rr = np.meshgrid(self.R_grid, self.r_grid, indexing="ij")
        self._mesh = (RR, rr)
        self.V = np.asarray(pes(RR, rr), dtype=float)

        if g_matrix is None:
            g = np.array([[1.0 / mass_R, 0.0],
                          [0.0, 1.0 / mass_r]])
        else:
            g = np.asarray(g_matrix, dtype=float)
        self.g_matrix = g

        kR = 2 * np.pi * np.fft.fftfreq(self.n_R, self.dR)[:, None]
        kr = 2 * np.pi * np.fft.fftfreq(self.n_r, self.dr)[None, :]
        T_k = 0.5 * (g[0, 0] * kR ** 2
                     + 2.0 * g[0, 1] * kR * kr
                     + g[1, 1] * kr ** 2)
        self.exp_T = np.exp(-1j * dt * T_k)

        # CAP: 各边 η 图 + 合并复势 + 按格点确定性归因
        for edge in cap_edges:
            if edge not in self.EDGES:
                raise ValueError(f"未知 CAP 边: {edge}，可选 {self.EDGES}")
        self.cap_edges = tuple(cap_edges)
        n_cap_R = int(cap_width_frac * self.n_R)
        n_cap_r = int(cap_width_frac * self.n_r)
        ramps_R = {"min": _low_ramp(self.n_R, n_cap_R)[:, None],
                   "max": _high_ramp(self.n_R, n_cap_R)[:, None]}
        ramps_r = {"min": _low_ramp(self.n_r, n_cap_r)[None, :],
                   "max": _high_ramp(self.n_r, n_cap_r)[None, :]}

        eta_total = np.zeros((self.n_R, self.n_r))
        if self.cap_edges:
            eta_maps = []
            for edge in self.cap_edges:
                axis, side = edge.split("_")
                eta = cap_height * (ramps_R[side] if axis == "R"
                                    else ramps_r[side])
                eta_full = np.broadcast_to(eta, (self.n_R, self.n_r)).copy()
                eta_maps.append(eta_full)
                eta_total = eta_total + eta_full
            # 角点重叠: 归 η 最大的边; 并列取边清单顺序靠前者 (argmax 稳定)
            self._assign = np.argmax(np.stack(eta_maps), axis=0)
            self._assign_flat = self._assign.ravel()
        else:
            eta_maps = []
        self._eta_maps = eta_maps
        self.eta_total = eta_total

        # 复势 W = V - iη 的对半分裂因子 (含每半步 e^{-η·dt/2} 阻尼)
        self.exp_W_half = np.exp(-0.5j * dt * (self.V - 1j * eta_total))

        self.last_absorbed: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def step(self, psi: np.ndarray) -> np.ndarray:
        """推进一个时间步: W/2 → T → W/2 (W = V - iη)。

        吸收量在每半步按格点变化确定性归因到对应 CAP 边，
        与边清单顺序无关 (每格点只归其 assigned 边)。
        """
        losses = np.zeros(len(self.cap_edges))
        if self.cap_edges:
            p2 = np.abs(psi) ** 2

        psi = self.exp_W_half * psi
        if self.cap_edges:
            d = (p2 - np.abs(psi) ** 2) * self.dV
            losses += np.bincount(self._assign_flat, weights=d.ravel(),
                                  minlength=len(self.cap_edges))

        psi = np.fft.ifft2(self.exp_T * np.fft.fft2(psi))

        p2 = np.abs(psi) ** 2 if self.cap_edges else None
        psi = self.exp_W_half * psi
        if self.cap_edges:
            d = (p2 - np.abs(psi) ** 2) * self.dV
            losses += np.bincount(self._assign_flat, weights=d.ravel(),
                                  minlength=len(self.cap_edges))

        self.last_absorbed = {e: float(v)
                              for e, v in zip(self.cap_edges, losses)}
        return psi

    # ------------------------------------------------------------------
    def check_initial_overlap(self, psi0: np.ndarray) -> float:
        """初始波包落入 CAP 区域的概率 (应远小于 1，如 < 1e-4)。"""
        if not self.cap_edges:
            return 0.0
        cap_cells = self.eta_total > 0
        p2 = np.abs(psi0) ** 2 * self.dV
        return float(p2[cap_cells].sum())

    # ------------------------------------------------------------------
    def _k_arrays(self):
        kR = 2 * np.pi * np.fft.fftfreq(self.n_R, self.dR)[:, None]
        kr = 2 * np.pi * np.fft.fftfreq(self.n_r, self.dr)[None, :]
        return kR, kr

    def energy_expectation(self, psi: np.ndarray) -> float:
        """⟨H⟩ = ⟨T⟩ + ⟨V⟩ (Hartree), k 空间动能 + 实空间势能。

        T 与 V 不对易, split-operator 传播中 ⟨H⟩ 会有 O(dt²) 级漂移;
        该量作为传播可信度诊断 (见 core.validation.check_propagation)。
        """
        psi = np.asarray(psi, dtype=complex)
        kR, kr = self._k_arrays()
        g = self.g_matrix
        psi_k = np.fft.fft2(psi)
        t_k = 0.5 * (g[0, 0] * kR ** 2
                     + 2.0 * g[0, 1] * kR * kr
                     + g[1, 1] * kr ** 2)
        rho_k = np.abs(psi_k) ** 2
        rho_x = np.abs(psi) ** 2
        if rho_k.sum() <= 0 or rho_x.sum() <= 0:
            return float("nan")
        t_mean = float(np.sum(t_k * rho_k) / np.sum(rho_k))   # Parseval: dV 抵消
        v_mean = float(np.sum(self.V * rho_x) / np.sum(rho_x))
        return t_mean + v_mean

    # ------------------------------------------------------------------
    def propagate(self, psi0: np.ndarray, n_steps: int,
                  save_every: int = 10,
                  product_mask: Optional[Callable] = None,
                  product_edges: Tuple[str, ...] = (),
                  reactant_mask: Optional[Callable] = None,
                  save_density: bool = True,
                  track_energy: bool = False) -> WavePacket2DResult:
        """传播波包并记录通道概率随时间的演化。

        Parameters
        ----------
        product_mask : Callable(R, r) -> bool array
            产物区掩码 (几何分域)。
        reactant_mask : Callable(R, r) -> bool array
            反应物区掩码。若两掩码均提供则必须互斥 (重叠时抛错)；
            二者恰好划分网格时 P_react + P_refl = 1 精确成立。
        product_edges : sequence of str
            视为产物吸收通道的 CAP 边 (如 LEPS 交换反应取 "r_max"，
            Eckart 模型取 "R_min")；其余启用边计入反射通道。
        """
        for e in product_edges:
            if e not in self.cap_edges:
                raise ValueError(f"product_edges 中的 {e} 未启用 CAP")

        psi = np.array(psi0, dtype=complex, copy=True)

        save_steps = list(range(0, n_steps + 1, save_every))
        if save_steps[-1] != n_steps:
            save_steps.append(n_steps)
        save_index = {s: k for k, s in enumerate(save_steps)}
        times = np.array(save_steps, dtype=float) * self.dt
        n_save = len(save_steps)

        RR, rr = self._mesh
        prod_mask = product_mask(RR, rr) if product_mask is not None else None
        react_mask = reactant_mask(RR, rr) if reactant_mask is not None else None
        if prod_mask is not None and react_mask is not None:
            if np.any(prod_mask & react_mask):
                raise ValueError("product_mask 与 reactant_mask 重叠, "
                                 "通道记账要求二者互斥")

        snapshots = np.zeros((n_save, self.n_R, self.n_r)) if save_density else None
        energies = np.zeros(n_save) if track_energy else None
        norm_t = np.zeros(n_save)
        prod_pop = np.zeros(n_save)
        react_pop = np.zeros(n_save)
        absorbed = {e: np.zeros(n_save) for e in self.cap_edges}
        cumulative = {e: 0.0 for e in self.cap_edges}

        def record(k: int, psi: np.ndarray):
            p2 = np.abs(psi) ** 2 * self.dV
            norm_t[k] = p2.sum()
            if prod_mask is not None:
                prod_pop[k] = p2[prod_mask].sum()
            if react_mask is not None:
                react_pop[k] = p2[react_mask].sum()
            for e in absorbed:
                absorbed[e][k] = cumulative[e]
            if save_density:
                snapshots[k] = np.abs(psi) ** 2
            if track_energy:
                energies[k] = self.energy_expectation(psi)

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

        drift = None
        reference = None
        if track_energy and energies[0] != 0:
            reference = float(energies[0])
            drift = float(np.max(np.abs(energies - energies[0]))) / abs(reference)

        return WavePacket2DResult(
            R_grid=self.R_grid,
            r_grid=self.r_grid,
            V_grid=self.V,
            times=times,
            snapshots=snapshots,
            norm_t=norm_t,
            absorbed=absorbed,
            product_population=prod_pop,
            reactant_population=react_pop,
            reaction_prob=reaction,
            reflection_prob=reflection,
            energy_track=energies,
            energy_drift_rel=drift,
            energy_reference=reference,
        )

    # ------------------------------------------------------------------
    def expectation_R(self, psi: np.ndarray) -> float:
        """计算 ⟨R⟩ (用于诊断波包运动)。"""
        p2 = np.abs(psi) ** 2 * self.dV
        RR, _ = self._mesh
        return float(np.sum(p2 * RR))


# ---------------------------------------------------------------------------
# 能量扫描 — 与时间无关扫描接口兼容
# ---------------------------------------------------------------------------

class WavePacket2DScan:
    """用一系列不同初始动量的波包扫描碰撞能，得到反应概率 P(E)。

    ``energy_*`` 为**碰撞能** (初始平动能 E = p_R0²/2μ_R，非总能量)。
    每个能量下波包从反应物渐近区出发 (p_R0 < 0，朝相互作用区传播)，
    传播结束后取累计反应概率。返回与 ``ScatteringResult2D`` 兼容的
    结果，可直接接入既有可视化/汇总流程。

    方法说明: 这是有限能量宽度高斯波包在固定传播时长下的单点估计 —
    未做能量反卷积，也未自动判断传播是否充分。定量使用时应自行做
    网格/时长/能量宽度收敛检查 (见 RELEASE_VALIDATION.md 的流程)。

    注意: 默认 LEPS 参数化 (D=0.1744, alpha=1.028, r0=1.401) 为教学
    模型，其共线交换路径势垒 (~0.14 au MEP) 远高于真实 H+H₂
    (~0.4 eV)；该参数化下的反应阈值不是真实体系的物理量。
    """

    def __init__(self, propagator: WavePacket2DPropagator,
                 packet: WavePacket2D,
                 product_mask: Callable,
                 product_edges: Tuple[str, ...],
                 reactant_mask: Optional[Callable] = None,
                 n_steps: int = 2400,
                 save_every: int = 40):
        self.prop = propagator
        self.packet = packet
        self.product_mask = product_mask
        self.product_edges = product_edges
        self.reactant_mask = reactant_mask
        self.n_steps = n_steps
        self.save_every = save_every

    def run(self, energy_min: float, energy_max: float,
            n_points: int = 8) -> ScatteringResult2D:
        prop = self.prop

        energies = np.linspace(energy_min, energy_max, n_points)
        reaction = np.zeros(n_points)

        for i, E in enumerate(energies):
            if E <= 0:
                reaction[i] = 0.0
                continue
            self.packet.p_R0 = -np.sqrt(2.0 * prop.mass_R * E)
            psi0 = self.packet.initialize(prop.R_grid, prop.r_grid)
            res = prop.propagate(
                psi0, self.n_steps,
                save_every=max(1, self.n_steps // 20),
                product_mask=self.product_mask,
                product_edges=self.product_edges,
                reactant_mask=self.reactant_mask,
                save_density=False,
            )
            reaction[i] = float(np.clip(res.reaction_prob[-1], 0.0, 1.0))

        return ScatteringResult2D(
            energy=energies,
            transmission=reaction.copy(),
            reaction_prob=reaction,
            R_grid=prop.R_grid,
            r_grid=prop.r_grid,
        )


# ---------------------------------------------------------------------------
# 能量分辨率、多宽度扫描与反卷积
# ---------------------------------------------------------------------------

@dataclass
class MultiWidthScanResult:
    """多包宽度扫描结果。

    ``reaction[i, j]`` = 第 i 个宽度 (sigma_R[i]) 在碰撞能
    ``energies[j]`` 处测得的反应概率。有限宽度波包测的是能量平均
    ``∫|ã(E′)|²T(E′)dE′`` 而非点值 T(E); 用多个宽度联合可反卷积出
    T(E) 的近似 (见 ``deconvolve_reaction``)。
    """

    sigmas: np.ndarray              # (n_w,) 包振幅宽度
    energies: np.ndarray            # (m,) 碰撞能 (E = p²/2μ_R)
    reaction: np.ndarray            # (n_w, m) 各宽度的反应概率
    energy_spread: np.ndarray       # (n_w, m) 能量分辨率 σ_E ≈ p·σ_p/μ


def energy_envelope_at(sigma_R: float, mass_R: float,
                       p_center: float, energies: np.ndarray) -> np.ndarray:
    """给定中心动量 p_center 的波包, 其入射能量包络 w(E) (∫w dE = 1)。

    |ã(k)|² ∝ exp(-(k-p₀)²/σ_k²), σ_k = 1/(√2·σ_R); 变量代换
    E = k²/2μ_R (k < 0 向左传播), w(E) = |ã(k(E))|·|dk/dE|。
    """
    E = np.maximum(np.asarray(energies, dtype=float), 1e-14)
    sigma_k = 1.0 / (np.sqrt(2.0) * sigma_R)
    k = -np.sqrt(2.0 * mass_R * E)
    w = np.exp(-((k - p_center) / sigma_k) ** 2)
    w *= np.sqrt(mass_R / (2.0 * E))               # |dk/dE|
    w /= np.trapezoid(w, E)
    return w


def multi_width_scan(prop: WavePacket2DPropagator, packet: WavePacket2D,
                     product_mask: Callable,
                     product_edges: Tuple[str, ...],
                     energy_min: float, energy_max: float, n_points: int,
                     sigmas=(0.3, 0.5, 0.8),
                     reactant_mask: Optional[Callable] = None,
                     n_steps: int = 2400) -> MultiWidthScanResult:
    """用多个包宽度重复扫描, 获得能量平均响应矩阵 (反卷积的输入)。

    注意每个扫描点的波包以该点能量为中心 (p₀ = -√(2μ_R·E)),
    因此反卷积响应包含扫描点平移 (见 ``deconvolve_reaction``)。
    """
    from dataclasses import replace

    sigmas = np.asarray(sigmas, dtype=float)
    energies = np.linspace(energy_min, energy_max, n_points)
    reaction = np.zeros((sigmas.size, n_points))
    spread = np.zeros((sigmas.size, n_points))

    for i, s in enumerate(sigmas):
        scan = WavePacket2DScan(prop, replace(packet, sigma_R=float(s)),
                                product_mask, product_edges,
                                reactant_mask=reactant_mask, n_steps=n_steps)
        res = scan.run(energy_min, energy_max, n_points)
        reaction[i] = res.reaction_prob
        p = np.sqrt(2.0 * prop.mass_R * energies)
        spread[i] = p / (np.sqrt(2.0) * s) / prop.mass_R

    return MultiWidthScanResult(sigmas=sigmas, energies=energies,
                                reaction=reaction, energy_spread=spread)


def deconvolve_reaction(mw: MultiWidthScanResult, mass_R: float,
                        n_fine: int = 200, curvature: float = 0.1,
                        ridge: float = 1e-8
                        ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """多宽度反卷积: 最小二乘拟合 T(E) (分段线性基 + 曲率正则化)。

    扫描点 j 的波包以 E_j 为中心, 测得

        p_ij = ∫ w_i(E; E_j)·T(E) dE ≈ Σ_k M_ijk T_k,
        M_ijk = w_i(E_k; E_j)·ΔE  (细网格上的线性 hat 基近似)

    将所有 (宽度, 扫描点) 行堆叠后解带正则的最小二乘

        (AᵀA + λ_D·D₂ᵀD₂ + λ_r·I) T = Aᵀ p

    其中 D₂ 为二阶差分 (惩罚曲率, 对 T(E) 的台阶形状比 L2 幅值
    正则更自然)。反卷积本质病态 (不同宽度包络在能量轴上高度相关,
    cond(A) 可达 1e15+), 输出是带先验假设的估计量而非点值真解。

    **数据前提**: 测得的 p 必须与包络模型一致 — 有限传播时间下低能
    慢分量未完成会引入系统偏差, 并被病态性放大为振荡。请先用
    ``check_convergence`` 确认传播充分, 并检查返回的相对残差;
    残差大时反卷积曲线不可信 (应视为诊断指标而非结果)。

    返回 (E_fine, T_fine, cond(A), 相对残差 ‖AT-p‖/‖p‖);
    T 裁剪到 [0, 1]。
    """
    E_fine = np.linspace(mw.energies.min(), mw.energies.max(), n_fine)
    dE = E_fine[1] - E_fine[0]

    rows = []
    p_rows = []
    for i, s in enumerate(mw.sigmas):
        for j, E_c in enumerate(mw.energies):
            p_c = -np.sqrt(2.0 * mass_R * E_c)
            w = energy_envelope_at(float(s), mass_R, p_c, E_fine)
            rows.append(w * dE)
            p_rows.append(mw.reaction[i, j])
    A = np.vstack(rows)
    p = np.array(p_rows)

    AtA = A.T @ A
    D2 = np.zeros((n_fine - 2, n_fine))
    for i in range(n_fine - 2):
        D2[i, i:i + 3] = [1.0, -2.0, 1.0]
    # 增广最小二乘 [A; √λ·D₂] 而非正规方程 (AᵀA 会平方条件数,
    # cond(A) ~1e9 时正规方程触及 float64 噪声底并产生 0/1 振荡)
    M = np.vstack([A, np.sqrt(curvature) * D2])
    rhs = np.concatenate([p, np.zeros(D2.shape[0])])
    T, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    T = np.clip(T, 0.0, 1.0)

    residual = float(np.linalg.norm(A @ T - p)
                     / max(np.linalg.norm(p), 1e-12))
    cond = float(np.linalg.cond(A))
    return E_fine, T, cond, residual


def check_convergence(make_propagator: Callable, packet: WavePacket2D,
                      n_R: int, n_r: int, dt: float, n_steps: int,
                      product_mask: Callable, product_edges: Tuple[str, ...],
                      reactant_mask: Optional[Callable] = None,
                      refine=(1.25, 1.25, 1.5, 1.3)) -> Dict[str, float]:
    """传播收敛性检查: 基线设置 vs 加密设置。

    ``make_propagator(n_R, n_r, dt)`` 返回同一体系/初态网格的传播子;
    ``refine = (n_R 因子, n_r 因子, dt 除数, 步数因子)``, 时长保持
    不变 (n_steps 同时放大)。返回两个设置的最终反应概率、存活
    概率及其差值 — 定量使用前应使 ``abs_diff`` 小于目标误差。
    """
    results = {}
    for name, (sR, sr, sdt, ssteps) in (("baseline", (1.0, 1.0, 1.0, 1.0)),
                                         ("refined", refine)):
        prop = make_propagator(int(round(n_R * sR)), int(round(n_r * sr)),
                               dt / sdt)
        steps = int(round(n_steps * ssteps))
        psi0 = packet.initialize(prop.R_grid, prop.r_grid)
        res = prop.propagate(psi0, steps, save_every=max(1, steps // 8),
                             product_mask=product_mask,
                             product_edges=product_edges,
                             reactant_mask=reactant_mask,
                             save_density=False)
        results[f"{name}_p_react"] = float(res.reaction_prob[-1])
        results[f"{name}_norm"] = float(res.norm_t[-1])
    results["abs_diff"] = abs(results["refined_p_react"]
                               - results["baseline_p_react"])
    return results


# ---------------------------------------------------------------------------
# H + H₂ 体系辅助 (物理单位)
# ---------------------------------------------------------------------------

def h3_reduced_masses(mass_H: float = DEFAULT_MASS_H) -> Tuple[float, float]:
    """共线 H + H₂ Jacobi 坐标约化质量 (μ_R, μ_r) = (2m/3, m/2)。

    R = 入射 H 原子到 BC 分子质心的距离 (μ_R = 2m/3)，
    r = BC 键长 (μ_r = m/2)。
    """
    return 2.0 * mass_H / 3.0, mass_H / 2.0


def leps_jacobi_pes(leps_builder) -> Callable:
    """把 LEPSBuilder (键长坐标) 包装为 Jacobi 坐标势函数 V(R, r)。

    共线构型下 r_AB = R - r/2, r_AC = R + r/2。

    注意: 仓库默认 LEPS 参数 (D=0.1744, alpha=1.028, r0=1.401) 为
    教学模型，共线交换势垒偏高 (~0.14 au MEP)，远高于真实 H+H₂；
    需要物理量标时建议使用 Eckart 模型 (势垒 0.015 au)。
    """
    inner = leps_builder._pes

    def V(R, r):
        R = np.asarray(R, dtype=float)
        r = np.asarray(r, dtype=float)
        return inner.evaluate(R - 0.5 * r, r, R + 0.5 * r)

    return V


def leps_exchange_mask() -> Callable:
    """H + H₂ 交换反应产物区掩码: r_AB < r_BC，即 R < 1.5·r。

    与反应物区 {R ≥ 1.5·r} 恰好划分整个网格。
    """
    return lambda R, r: np.asarray(R) < 1.5 * np.asarray(r)


def eckart_product_mask(R_div: float = 2.0) -> Callable:
    """Eckart 模型产物区掩码 R < R_div。

    与反应物区 {R ≥ R_div} 恰好划分整个网格 (配合 engine 的掩码)。
    """
    return lambda R, r: np.asarray(R) < R_div


def morse_ground_width(D: float, alpha: float, mass_r: float) -> float:
    """Morse 振子基态的振幅高斯宽度 σ = 1/√(μω)，ω = α√(2D/μ)。

    谐振近似 (Morse 基态的精确宽度需含非谐修正)。返回值直接用作
    ``WavePacket2D`` 的 sigma 参数 (振幅宽度, 非位置标准差)。
    """
    omega = alpha * np.sqrt(2.0 * D / mass_r)
    return float(1.0 / np.sqrt(mass_r * omega))


def harmonic_ground_width(k: float, mass_r: float) -> float:
    """谐振子基态的振幅高斯宽度 σ = 1/√(μω)，ω = √(k/μ)。

    这是精确基态: |ψ₀|² ∝ exp(-(x-x₀)²/σ²)。返回值直接用作
    ``WavePacket2D`` 的 sigma 参数 (振幅宽度, 非位置标准差)。
    """
    omega = np.sqrt(k / mass_r)
    return float(1.0 / np.sqrt(mass_r * omega))


def exchange_dividing_surface_line(R_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LEPS 交换反应分界面 R = 1.5·r 的绘图线段 (R_pts, r_pts)。"""
    r_pts = np.linspace(0.0, R_grid[-1] / 1.5, 64)
    return 1.5 * r_pts, r_pts
