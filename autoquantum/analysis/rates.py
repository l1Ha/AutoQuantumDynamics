"""热速率常数 — 从微观反应概率 P(E) 计算 k(T) 与 Arrhenius 参数。

对二维 (共线) 反应, 热速率常数的标准表达式:

    k(T) = ∫₀^∞ P(E) · exp(−E/k_BT) · v(E) dE / (k_BT)

其中 v(E) = √(2E/μ) 为相对速度。绝对归一化需要知道初始态配分函数
和碰撞参数; 本模块输出**约化速率常数** (Boltzmann 加权平均 P(E)),
可直接用于势能面间的相对比较和 Arrhenius 分析:

    k(T) = (1/k_BT) ∫₀^∞ P(E) · exp(−E/k_BT) dE

Arrhenius 拟合: ln k(T) = ln A − E_a/(R·T)

参考: P. Pechukas, Ann. Rev. Phys. Chem. 32, 159 (1981);
      Truhlar et al., J. Phys. Chem. A 107, 7068 (2003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 物理常数 (au → SI)
KB_AU = 3.166811563e-6        # Boltzmann (Hartree/K)
AU_TO_KJ_MOL = 2625.500       # Hartree → kJ/mol
AU_TO_KCAL_MOL = 627.509      # Hartree → kcal/mol


@dataclass
class ArrheniusFit:
    """Arrhenius 拟合结果。

    Attributes:
        ea_kj_mol: 活化能 E_a (kJ/mol)。
        log_a: 指前因子常用对数 log₁₀A (化学标准)。
        r_squared: 线性回归决定系数 R²。
        temperatures: 拟合所用温度数组 (K)。
        rates: 对应速率常数数组 (约化单位)。
    """
    ea_kj_mol: float
    log_a: float
    r_squared: float
    temperatures: np.ndarray
    rates: np.ndarray

    def summary(self) -> str:
        """单行摘要: 指前因子 A、活化能 (kJ/mol 与 kcal/mol) 与 R²。"""
        return (f"Arrhenius: A = {10**self.log_a:.3e}, "
                f"Ea = {self.ea_kj_mol:.2f} kJ/mol "
                f"({self.ea_kj_mol / 4.184:.2f} kcal/mol), "
                f"R² = {self.r_squared:.4f}")


def thermal_rate_constant(E_grid: np.ndarray, P_grid: np.ndarray,
                          temperatures: np.ndarray,
                          min_energy: float = 0.0,
                          ) -> np.ndarray:
    """从反应概率 P(E) 计算 Boltzmann 加权热速率常数 (约化单位)。

    k(T) = (1/k_BT) ∫ P(E) · exp(−E/k_BT) dE

    P(E) 为有限温度波包扫描的反应概率; 只使用 E > min_energy 的数据。
    返回与 temperatures 同长的 k(T) 数组 (约化单位: 无量纲速率)。
    """
    E = np.asarray(E_grid, dtype=float)
    P = np.asarray(P_grid, dtype=float)
    order = np.argsort(E)
    E, P = E[order], P[order]
    mask = E >= min_energy
    E, P = E[mask], P[mask]

    rates = np.zeros(len(temperatures))
    for i, T in enumerate(temperatures):
        kT = KB_AU * T
        boltzmann = np.exp(-E / kT)
        Q = np.trapezoid(boltzmann, E)
        N_T = np.trapezoid(P * boltzmann, E)
        rates[i] = N_T / max(Q, 1e-30)
    return rates


def arrhenius_fit(temperatures: np.ndarray,
                  rates: np.ndarray) -> ArrheniusFit:
    """Arrhenius 拟合: ln k = ln A − Ea/(R·T)。

    使用 E_a (Hartree→kJ/mol) 和 R = 8.314 J/(mol·K)。
    """
    T = np.asarray(temperatures, dtype=float)
    k = np.asarray(rates, dtype=float)
    # 避免零速率
    mask = k > 0
    if mask.sum() < 2:
        return ArrheniusFit(ea_kj_mol=0.0, log_a=0.0, r_squared=0.0,
                            temperatures=T, rates=k)
    inv_T = 1.0 / T[mask]
    ln_k = np.log(k[mask])
    # 线性回归: ln_k = ln_A − (Ea/R) * (1/T)
    coeffs = np.polyfit(inv_T, ln_k, 1)
    slope, intercept = coeffs
    ea_R = -slope                        # Ea/R in Kelvin
    ea_kj_mol = ea_R * 8.314 / 1000.0    # Ea in kJ/mol
    log_a = intercept / np.log(10)       # log10(A) (化学标准)
    # R²
    pred = np.polyval(coeffs, inv_T)
    ss_res = np.sum((ln_k - pred) ** 2)
    ss_tot = np.sum((ln_k - np.mean(ln_k)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return ArrheniusFit(ea_kj_mol=ea_kj_mol, log_a=log_a, r_squared=r2,
                        temperatures=T, rates=k)


def plot_arrhenius(temperatures: np.ndarray, rates: np.ndarray,
                   fit: Optional[ArrheniusFit] = None,
                   save_path: str = "arrhenius.png") -> str:
    """生成 Arrhenius 图 (ln k vs 1/T)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    T = np.asarray(temperatures, dtype=float)
    k = np.asarray(rates, dtype=float)
    inv_T = 1000.0 / T   # 1000/T for readable axis
    ax.plot(inv_T, np.log(k), "o", color="#2c3e50", ms=6, label="k(T)")
    if fit is not None and fit.ea_kj_mol > 0:
        T_fit = np.linspace(T.min(), T.max(), 100)
        inv_T_fit = 1000.0 / T_fit
        k_fit = np.exp(fit.log_a - fit.ea_kj_mol * 1000 / (8.314 * T_fit))
        ax.plot(inv_T_fit, np.log(k_fit), "--", color="#c0392b", lw=1.5,
                label=f"Arrhenius fit (Ea={fit.ea_kj_mol:.1f} kJ/mol)")
    ax.set_xlabel("1000/T (K$^{-1}$)")
    ax.set_ylabel("ln k(T) (reduced)")
    ax.set_title("Arrhenius Analysis")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
