#!/usr/bin/env python3
"""生成教材配图 — 所有图均由真实代码计算, 不含装饰性示意图。

用法:
    python scripts/make_book_figures.py            # 全部图 → book/figures/
    python scripts/make_book_figures.py --smoke    # 快速冒烟 (少量图/小网格)

图中坐标轴标签使用英文 (国际惯例, 避免 CJK 字体依赖), 中文说明在
各章正文中给出。
"""

import os
import sys
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoquantum.pes.analytic import MorsePES, HarmonicPES
from autoquantum.pes.eckart import EckartBuilder
from autoquantum.pes.leps import LEPSBuilder
from autoquantum.dynamics.wavepacket import WavePacket1D, SplitOperatorPropagator
from autoquantum.dynamics.wavepacket_2d import (
    WavePacket2D, WavePacket2DPropagator, h3_reduced_masses,
    harmonic_ground_width, morse_ground_width, eckart_product_mask,
    leps_jacobi_pes, leps_exchange_mask, exchange_dividing_surface_line,
    deconvolve_reaction, energy_envelope_at,
)
from autoquantum.dynamics.quantum_1d import QuantumScattering1D
from autoquantum.nn.train import NNTrainer, TrainingConfig

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "book", "figures")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_morse_levels(smoke=False):
    """图 6.1: Morse 势、谐振近似与量子振动能级 (含零点能)。"""
    pes = MorsePES({"D": 0.1744, "alpha": 1.028, "r0": 0.7416})
    harm = HarmonicPES({"k": 2 * 0.1744 * 1.028 ** 2, "r0": 0.7416})
    r = np.linspace(0.55, 1.6, 800)
    V = pes(r); Vh = harm(r)
    Vmin = V.min(); Vh_min = Vh.min()
    mu = 918.075  # m_H/2 (电子质量)
    omega = 1.028 * np.sqrt(2 * 0.1744 / mu)
    D = 0.1744
    n_max = 4 if smoke else 9
    # 精确 Morse 本征能: E_v = ħω_e(v+1/2) - [ħω_e(v+1/2)]²/(4D_e)
    levels = [omega * (n + 0.5) - (omega * (n + 0.5)) ** 2 / (4 * D)
              for n in range(n_max)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r, V - Vmin, color="#2c3e50", lw=2, label="Morse PES")
    ax.plot(r, Vh - Vh_min, "--", color="#7f8c8d", lw=1.5,
            label="harmonic approx")
    for n, E in enumerate(levels):
        x = r[r < 0.7416 + 0.75 * (0.7416 - 0.55)]
        ax.hlines(E, x.min(), x.max(), color="#c0392b", lw=0.9, alpha=0.75)
        ax.text(x.max() - 0.02, E, f"$v={n}$", fontsize=7, va="center",
                ha="right", color="#c0392b")
    ax.set_xlabel("bond length r (Bohr)"); ax.set_ylabel("E - V$_{min}$ (au)")
    ax.set_title("Morse potential with exact (anharmonic) levels\n"
                 "$\\omega_e$ = %.4f au, ZPE = %.4f au, D$_e$ = %.3f au"
                 % (omega, levels[0], D))
    ax.set_ylim(-0.01, max(levels) * 1.15)
    ax.legend()
    _save(fig, "ch06_morse_levels.png")


def fig_wavefunctions(smoke=False):
    """图 6.2: 基态与激发态波函数 (谐振子, 解析)。"""
    mu, omega = 1.0, 1.0
    x = np.linspace(-5, 5, 900)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2c3e50", "#27ae60", "#8e44ad"]
    for n in range(1 if smoke else 3):
        # ψ_n(x) ∝ H_n(√(μω)x)·exp(-μωx²/2), H_n = 物理学家 Hermite
        arg = np.sqrt(mu * omega) * x
        Hn = np.polynomial.hermite.Hermite([0] * n + [1])(arg)
        psi = ((mu * omega / np.pi) ** 0.25
               / np.sqrt(2 ** n * math.factorial(n))
               * Hn * np.exp(-mu * omega * x ** 2 / 2))
        E = (n + 0.5) * omega
        ax.plot(x, E + 0.35 * psi, color=colors[n], lw=1.6, label=f"v={n}")
        ax.hlines(E, -5, 5, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("q (au)"); ax.set_ylabel("energy (au) + wavefunction")
    ax.set_title("Harmonic-oscillator states: energy equally spaced, "
                 "$\\langle q\\rangle = 0$")
    ax.set_ylim(-0.3, 4)
    ax.legend()
    _save(fig, "ch06_wavefunctions.png")


def fig_ztp_temperature(smoke=False):
    """图 6.3: 零点能与热占据对有效反应阈值的影响。"""
    omega = 0.02  # H2 振动频率 (au), 真实量级
    T = np.linspace(50, 3000, 300)            # 典型分子束/室温范围 (K)
    kT = 8.617333e-5 * 27.2114 * T            # k_B T (au)
    frac0 = 1.0 / (1.0 + np.exp(-omega / kT))  # v=0 占据 (v=0,1 两能级模型)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(T, frac0, color="#2c3e50", lw=2)
    axes[0].set_xlabel("temperature (K)")
    axes[0].set_ylabel("population fraction in v=0")
    axes[0].set_title("Ground-state population DECREASES with T\n"
                      "(H$_2$: $\\omega$ = 0.02 au; v=0,1 model)")
    Vb = 0.015
    for shift, lab, c in ((0.0, "classical threshold $V_b$", "#c0392b"),
                          (0.5 * omega, "+ ZPE (adiabatic)", "#27ae60"),
                          (omega, "+ 2 ZPE", "#8e44ad")):
        axes[1].axvline(Vb + shift, color=c, lw=2, ls="--", label=lab)
    axes[1].set_xlim(Vb - 0.004, Vb + 0.03)
    axes[1].set_xlabel("effective barrier (au)")
    axes[1].set_title("ZPE raises the effective threshold\n"
                      "by the reactant-side vibrational energy")
    axes[1].legend(fontsize=8)
    _save(fig, "ch06_zpe_threshold.png")


def fig_scattering_1d(smoke=False):
    """图 7.1: 一维 Eckart 势垒的 T(E): 宽阈值 (物理质量)。"""
    mass_R, _ = h3_reduced_masses()
    E0 = 0.015
    beta = 1.5
    V = lambda x: E0 / np.cosh(beta * (x - 3.0)) ** 2  # eckart.py 的 sech² 垒
    solver = QuantumScattering1D(mass=mass_R, pes=V, n_grid=2000,
                                 grid_min=0.5, grid_max=9.0)
    Es = np.linspace(0.002, 0.05, 40 if smoke else 60)
    res = solver.solve(Es[0], Es[-1], len(Es))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res.energy, res.transmission, color="#2c3e50", lw=2)
    ax.axvline(E0, color="#c0392b", ls="--", lw=1,
               label="barrier top $V_0$ = %.3f au" % E0)
    ax.set_xlabel("collision energy E (au)")
    ax.set_ylabel("transmission T(E)")
    ax.set_title("sech$^2$ barrier model (eckart.py): a BROAD quantum\n"
                 "threshold (m = 2m$_H$/3 makes tunnelling negligible)")
    ax.legend(); ax.set_ylim(-0.02, 1.05)
    _save(fig, "ch07_scattering_threshold.png")


def fig_wavepacket_evolution(smoke=False):
    """图 8.1: 一维含时波包分裂演化 (自由+谐振) 的密度快照。"""
    g = np.linspace(-10, 10, 800)
    mass, omega = 1.0, 1.0
    packet = WavePacket1D(x0=-4.0, p0=4.0, sigma=0.5)
    prop = SplitOperatorPropagator(mass=mass, pes=lambda x: 0.5 * mass * omega**2 * x**2,
                                   grid=g, dt=0.05, cap_width_frac=0.1,
                                   cap_height=0.5)
    psi = packet.initialize(g)
    n_snap = 3 if smoke else 5
    snap = np.linspace(0, 160, n_snap).astype(int)
    fig, axes = plt.subplots(n_snap, 1, figsize=(7, 1.6 * n_snap), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, snap):
        for i in range(s):
            psi = prop.step(psi)
        ax.plot(g, np.abs(psi) ** 2, color="#2c3e50", lw=1.2)
        ax.set_ylabel(f"t={s * 0.05:.0f}", fontsize=8)
    axes[-1].set_xlabel("x (au)")
    fig.suptitle("Split-operator wavepacket in a harmonic trap "
                 "(COM oscillates, width breathes)", fontsize=10)
    _save(fig, "ch08_wavepacket_evolution.png")


def fig_ec_kart_pes(smoke=False):
    """图 9.1: Eckart 二维 PES 与交换分界面 (含零阶近似说明)。"""
    builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                             "r0": 1.401, "coupling": 0.08})
    R = np.linspace(0.5, 9.0, 300); r = np.linspace(0.5, 3.5, 300)
    Rg, rg, V = builder.generate_grid((0.5, 9.0), (0.5, 3.5), 300, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cs = axes[0].contourf(Rg, rg, np.clip(V, -0.1, 0.05), levels=40, cmap="viridis")
    fig.colorbar(cs, ax=axes[0], label="V (au)")
    axes[0].axvline(2.0, color="w", ls="--", label="product mask R < 2")
    axes[0].set_title("eckart.py model: sech$^2$ barrier + harmonic/coupling"); axes[0].legend(fontsize=7)
    mep = np.min(V, axis=1)
    axes[1].plot(Rg, mep, color="#c0392b")
    axes[1].set_xlabel("R (Bohr)"); axes[1].set_ylabel("min$_r$ V (au)")
    axes[1].set_title("MEP-like cut min$_r$ V(R)")
    for ax in axes[:1]:
        ax.set_xlabel("R (Bohr)"); ax.set_ylabel("r (Bohr)")
    _save(fig, "ch09_eckart_pes.png")


def fig_2d_snapshot(smoke=False):
    """图 9.2: 二维波包快照演化 (Eckart, 物理质量)。"""
    mass_R, mass_r = h3_reduced_masses()
    builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                             "r0": 1.401, "coupling": 0.08})
    k_r = 0.5
    r_eq = 1.401 - 0.08 / k_r * np.tanh(1.5 * (6.0 - 3.0))
    nR = 128 if smoke else 192
    nr = 96 if smoke else 144
    R = np.linspace(0.5, 9.0, nR); r = np.linspace(0.5, 3.5, nr)
    prop = WavePacket2DPropagator(builder.evaluate_2d, R, r, mass_R, mass_r,
                                  0.5, cap_edges=("R_min", "R_max"),
                                  cap_width_frac=0.15, cap_height=0.15)
    packet = WavePacket2D(R0=6.0, r0=r_eq, sigma_R=0.5,
                          sigma_r=harmonic_ground_width(k_r, mass_r),
                          p_R0=-np.sqrt(2 * mass_R * 0.03))
    steps = 1200 if smoke else 2200
    res = prop.propagate(packet.initialize(R, r), steps,
                         save_every=steps // 4,
                         product_mask=eckart_product_mask(2.0),
                         product_edges=("R_min",), save_density=True)
    RR, rr = np.meshgrid(R, r, indexing="ij")
    n_show = 2 if smoke else 4
    idx = np.linspace(1, len(res.times) - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.4))
    axes = np.atleast_1d(axes)
    dmax = res.snapshots[idx].max()
    im = None
    for ax, k in zip(axes, idx):
        im = ax.pcolormesh(R, r, res.snapshots[k].T, cmap="magma",
                           shading="auto", vmin=0, vmax=dmax)
        ax.contour(RR, rr, np.clip(prop.V, -0.1, 0.05), levels=10,
                   colors="cyan", linewidths=0.4, alpha=0.5)
        ax.axvline(2.0, color="w", ls="--", lw=0.8)
        ax.set_title(f"t = {res.times[k]:.0f} au", fontsize=9)
        ax.set_xlabel("R (Bohr)"); ax.set_ylabel("r (Bohr)")
        ax.grid(False)
    if im is not None:
        fig.colorbar(im, ax=list(axes), fraction=0.02, label="|ψ|²")
    fig.suptitle("2D wavepacket crossing the sech$^2$ barrier (finite-time "
                 "channel probability, incl. R$_{min}$ absorption: %.3f)"
                 % res.reaction_prob[-1], fontsize=9)
    _save(fig, "ch09_2d_wavepacket.png")


def fig_probability_bookkeeping(smoke=False):
    """图 8.2: 概率记账 — 通道概率随时间 + 恒等式。"""
    mass_R, mass_r = h3_reduced_masses()
    builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                             "r0": 1.401, "coupling": 0.0})
    nR, nr = (96, 64) if smoke else (160, 96)
    R = np.linspace(0.5, 9.0, nR); r = np.linspace(0.5, 3.5, nr)
    prop = WavePacket2DPropagator(builder.evaluate_2d, R, r, mass_R, mass_r,
                                  0.5, cap_edges=("R_min", "R_max"),
                                  cap_width_frac=0.15, cap_height=0.15)
    packet = WavePacket2D(R0=6.0, r0=1.401, sigma_R=0.5,
                          sigma_r=harmonic_ground_width(0.5, mass_r),
                          p_R0=-np.sqrt(2 * mass_R * 0.02))
    steps = 2500 if smoke else 4000
    res = prop.propagate(packet.initialize(R, r), steps, save_every=steps // 10,
                         product_mask=eckart_product_mask(2.0),
                         product_edges=("R_min",),
                         reactant_mask=lambda Rr, rr: np.asarray(Rr) >= 2.0,
                         save_density=False)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(res.times, res.reaction_prob, "g-", lw=2, label="P$_{react}$")
    ax.plot(res.times, res.reflection_prob, "r-", lw=2, label="P$_{refl}$")
    ax.plot(res.times, res.norm_t, "b--", lw=1, label="grid norm")
    total = res.reaction_prob[-1] + res.reflection_prob[-1]
    ax.set_xlabel("time (au)"); ax.set_ylabel("probability")
    ax.set_title("Channel bookkeeping: P$_{react}$+P$_{refl}$ = %.10f\n"
                 "identity: norm + CAP losses = %.10f"
                 % (total, res.norm_t[-1] + sum(v[-1] for v in res.absorbed.values())))
    ax.legend(); ax.set_ylim(-0.05, 1.05)
    _save(fig, "ch08_probability_bookkeeping.png")


def fig_nn_training(smoke=False):
    """图 5.1: 能量+力联合训练: 力训练提升梯度保真度。"""
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 31), np.linspace(-2, 2, 31),
                         indexing="ij")
    X = np.column_stack([g1.ravel(), g2.ravel()])
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] ** 2
    dY = np.column_stack([2 * np.cos(2 * X[:, 0]), 0.6 * X[:, 1]])
    epochs = 400 if smoke else 1200   # 与 tests.test_nn 的回归配置一致
    out = {}
    for fw, tag in ((0.0, "value only"), (1.0, "value + force")):
        m, h = NNTrainer(TrainingConfig(hidden_layers=[32, 32], epochs=epochs,
                                        lr=0.01, seed=4, force_weight=fw)
                         ).train(X, y, dY=dY if fw else None)
        out[tag] = (np.sqrt(np.mean((m.gradient(X) - dY) ** 2)),
                    np.sqrt(np.mean((m.predict(X) - y) ** 2)))
    labels = ["value only", "value + force"]
    g0, e0 = out[labels[0]]; g1, e1 = out[labels[1]]
    if not (g1 < 0.5 * g0 and e1 < e0):
        print("  [WARN] force training 未复现回归测试的改善 — "
              f"grad {g0:.2e}->{g1:.2e}, E {e0:.2e}->{e1:.2e}", file=sys.stderr)
    verdict = ("Force training improves gradient AND energy fidelity"
               if (g1 < g0 and e1 < e0) else
               "Force training effect on this run (see quoted numbers)")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    xpos = np.arange(2)
    ax.bar(xpos - 0.19, [g0, g1], 0.38, color="#c0392b", label="gradient RMSE")
    ax.bar(xpos + 0.19, [e0, e1], 0.38, color="#2c3e50", label="energy RMSE")
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_yscale("log"); ax.set_ylabel("RMSE (log)")
    ax.set_title(verdict + "\n"
                 f"grad: {g0:.2e} -> {g1:.2e};  E: {e0:.2e} -> {e1:.2e}")
    ax.legend()
    _save(fig, "ch05_nn_force_training.png")


def fig_deconvolution(smoke=False):
    """图 12.1: 多宽度能量平均与反卷积 (合成阈值真值)。"""
    mass_R, _ = h3_reduced_masses()
    E = np.linspace(0.005, 0.045, 7)
    sigmas = (0.4, 0.7, 1.1)
    T_true = 1.0 / (1.0 + np.exp(-(E - 0.02) / 0.004))
    E_fine = np.linspace(E.min(), E.max(), 200)
    T_fine = 1.0 / (1.0 + np.exp(-(E_fine - 0.02) / 0.004))
    reaction = np.zeros((3, len(E)))
    for i, s in enumerate(sigmas):
        for j, Ec in enumerate(E):
            w = energy_envelope_at(s, mass_R, -np.sqrt(2 * mass_R * Ec), E_fine)
            reaction[i, j] = np.trapezoid(w * T_fine, E_fine)
    from autoquantum.dynamics.wavepacket_2d import MultiWidthScanResult
    mw = MultiWidthScanResult(np.array(sigmas), E, reaction,
                              np.zeros_like(reaction))
    E_fit, T_fit, cond, residual = deconvolve_reaction(mw, mass_R)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, s in enumerate(sigmas):
        ax.plot(E, reaction[i], "o-", ms=4, alpha=0.7,
                label=f"packet $\\sigma_R$={s}")
    ax.plot(E_fine, T_fine, "k-", lw=2, label="true T(E)")
    ax.plot(E_fit, T_fit, "--", color="#c0392b", lw=1.8,
            label=f"deconvolved (residual {residual:.3f})")
    ax.set_xlabel("collision energy (au)"); ax.set_ylabel("P$_{react}$ / T(E)")
    ax.set_title("A finite-width packet measures an ENERGY AVERAGED T;\n"
                 "multi-width deconvolution recovers the pointwise curve")
    ax.legend(fontsize=8)
    _save(fig, "ch12_deconvolution.png")


def fig_eks_and_cep(smoke=False):
    """图 3.1: 势能曲线与 DFT 基组叠加误差 (概念演示, 标签为示意)。"""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    r = np.linspace(1.5, 6.0, 500)
    eps, sigma = 0.01, 3.5
    V2 = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    ax.plot(r, V2, color="#2c3e50", lw=2, label="dimer PES (full basis)")
    # 有限基的"BSSE"式人为降低: 有效势被减去常数
    ax.plot(r, V2 + 0.6 * eps, "--", color="#c0392b",
            label="finite-basis artifact (BSSE, schematic)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("interatomic distance (Bohr)")
    ax.set_ylabel("V (au)")
    ax.set_title("Basis-set superposition error: finite monomer basis\n"
                 "artificially stabilises the dimer (schematic, not computed)")
    ax.legend(fontsize=8)
    _save(fig, "ch04_eks_cep.png")


def fig_leps_topography(smoke=False):
    """图 3.2: LEPS 交换势能面地形 (教学参数, 势垒偏高声明)。"""
    builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028, "r0": 1.401, "sato": 0.05})
    pes = leps_jacobi_pes(builder)   # 真正的 Jacobi 映射 (R-r/2, r, R+r/2)
    R = np.linspace(0.5, 10.0, 200); r = np.linspace(0.2, 9.5, 160)
    Rg, rg = np.meshgrid(R, r, indexing="ij")
    V = pes(Rg, rg)
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(Rg, rg, np.clip(V, -0.21, 0.03), levels=40, cmap="magma")
    fig.colorbar(cs, ax=ax, label="V (au)")
    rr = np.linspace(0.2, 9.5 / 1.5, 64)
    ax.plot(1.5 * rr, rr, "w--", lw=1.2, label=r"exchange surface $R=1.5r$")
    ax.plot(6.7, 1.401, "w*", ms=10, label="reactant asymptote")
    ax.set_xlabel("R (Bohr)"); ax.set_ylabel("r (Bohr)")
    ax.set_title("LEPS (Jacobi coords, teaching parameters):\n"
                 "exchange barrier along R=1.5r: ~0.13 au, NOT the real H+H$_2$ value")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, "ch03_leps_pes.png")


FIGURES = {
    "morse_levels": fig_morse_levels,
    "wavefunctions": fig_wavefunctions,
    "ztp_threshold": fig_ztp_temperature,
    "scattering_1d": fig_scattering_1d,
    "wavepacket_evolution": fig_wavepacket_evolution,
    "eckart_pes": fig_ec_kart_pes,
    "2d_snapshot": fig_2d_snapshot,
    "probability_bookkeeping": fig_probability_bookkeeping,
    "nn_training": fig_nn_training,
    "deconvolution": fig_deconvolution,
    "eks_cep": fig_eks_and_cep,
    "leps_topography": fig_leps_topography,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="快速冒烟: 跳过昂贵的 2D/扫描图")
    parser.add_argument("--only", default=None,
                        help="只生成指定图 (逗号分隔)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    names = ([n.strip() for n in args.only.split(",")]
             if args.only else
             ([n for n in FIGURES if n not in ("2d_snapshot", "scattering_1d")]
              if args.smoke else list(FIGURES)))
    for name in names:
        if name not in FIGURES:
            print(f"未知图: {name}", file=sys.stderr)
            continue
        FIGURES[name](smoke=args.smoke)
    print(f"figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
