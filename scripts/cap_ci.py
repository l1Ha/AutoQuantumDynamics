#!/usr/bin/env python3
"""CAP-CI: 复吸收势 + 组态相互作用, 第一性原理计算共振态能量与宽度。

原理
----
共振态嵌在连续谱中, 无法作为 L² 束缚本征态获得。CAP (Complex Absorbing
Potential) 在电子坐标空间外加复势 −iηW(r), 把出射电子波"吸收"为复本征值:

    E(η) = E_R − iΓ/2,      Γ = −2 Im E(η)

对 η 扫描, 真正的共振在稳定化轨迹上表现为 |η dE/dη| 极小 (Le Roy 稳定点)。

实现要点 (本脚本, 基于 PySCF + numpy):
1. CAP 单电子积分用 DFT 数值网格: W_μν = Σ_g w_g χ_μ(r_g) w_cap(r_g) χ_ν(r_g),
   每原子二次型 CAP: w_cap(r) = 0 (r<r_cap) 否则 (r−r_cap)²;
2. 复 FCI 矩阵在 (nα, nβ) 扇区用**实数引擎逐列构造**:
   H = H_real − iη W  →  列 j = contract(H_real, e_j) − iη contract(W, e_j);
   活性空间小 (≤10 轨道), 显式稠密复矩阵可直接 numpy 对角化;
3. 扫 η ∈ [η_min, η_max], 取 |η dE/dη| 最小点为稳定化点。

用法:
    python scripts/cap_ci.py --system He-Li --R 5.556 --ncas 9 --rcap 12
    python scripts/cap_ci.py --system He-minus --validate   # 已知共振验证
"""

from __future__ import annotations

import argparse
import numpy as np

HA_EV = 27.211386245988


# ---------------------------------------------------------------------------
# CAP 单电子积分 (DFT 数值网格)
# ---------------------------------------------------------------------------

def cap_matrix(mol, r_cap: float, grid_level: int = 3, power: float = 2.0):
    """在 AO 基下构造 CAP 单电子矩阵 W_μν = Σ_g w_g χ_μ χ_ν w_cap(r_g)。

    多中心二次型 CAP: w_cap(r) = Σ_atoms (|r − R_A| − r_cap)²·Θ(|r−R_A| − r_cap)。
    返回对称实矩阵 (nao, nao), 单位 Hartree。
    """
    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    coords = grids.coords            # (ng, 3) Bohr
    weights = grids.weights          # (ng,)
    ao = dft.numint.eval_ao(mol, coords)   # (ng, nao)

    w_cap = np.zeros(len(weights))
    for ia in range(mol.natm):
        R = mol.atom_coord(ia)
        d = np.linalg.norm(coords - R, axis=1)
        w_cap += np.where(d > r_cap, (d - r_cap) ** power, 0.0)

    wao = (ao * (weights * w_cap)[:, None]).T @ ao   # (nao, nao)
    return 0.5 * (wao + wao.T)


# ---------------------------------------------------------------------------
# 复 FCI 矩阵 (实数引擎逐列构造)
# ---------------------------------------------------------------------------

def cap_ci_resonance(mol, mo_coeff, ncas, nelec, r_cap, etas=None,
                     grid_level=3, verbose=True):
    """对给定几何做 CAP-CI 扫描, 返回稳定化后的 (E_R, Γ) 与全部 η 轨迹。

    nelec: (nα, nβ) 在活性空间内。
    """
    from pyscf import ao2mo, fci
    from pyscf.fci import direct_spin1

    C = np.ascontiguousarray(mo_coeff[:, :ncas])
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h1e = C.T @ h_core @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), ncas)
    W_ao = cap_matrix(mol, r_cap, grid_level=grid_level)
    w1e = C.T @ W_ao @ C
    w1e = 0.5 * (w1e + w1e.T)

    if etas is None:
        etas = np.array([0.0005, 0.001, 0.002, 0.004, 0.008, 0.016])

    from math import comb
    na, nb = nelec
    dim = comb(ncas, na) * comb(ncas, nb)
    h2e = direct_spin1.absorb_h1e(h1e, eri, ncas, nelec, 0.5)
    w2e = direct_spin1.absorb_h1e(w1e, np.zeros_like(eri), ncas, nelec, 0.5)

    # 实部矩阵 (一次构造, 复用)
    H_real = np.zeros((dim, dim))
    for j in range(dim):
        e = np.zeros(dim); e[j] = 1.0
        ci = e.reshape(comb(ncas, na), comb(ncas, nb))
        H_real[:, j] = direct_spin1.contract_2e(h2e, ci, ncas, nelec).ravel()
    H_real = 0.5 * (H_real + H_real.T)

    W_mat = np.zeros((dim, dim))
    for j in range(dim):
        e = np.zeros(dim); e[j] = 1.0
        ci = e.reshape(comb(ncas, na), comb(ncas, nb))
        W_mat[:, j] = direct_spin1.contract_2e(w2e, ci, ncas, nelec).ravel()
    W_mat = 0.5 * (W_mat + W_mat.T)

    e_nuc = mol.energy_nuc()      # ⚠ 本征值是纯电子能量, 必须加 E_nuc 才是总能量
    traj = []
    for eta in etas:
        H = H_real - 1j * eta * W_mat
        ev = np.linalg.eigvals(H) + e_nuc
        traj.append((eta, ev))
    return traj, etas


def stabilize(traj, etas, e_window=None, verbose=True):
    """从 η 轨迹中挑选共振: 在能量窗口内 |Im| 最大, 且 η dE/dη 最小。"""
    cand = []
    for k, (eta, ev) in enumerate(traj):
        for E in ev:
            if e_window and not (e_window[0] < E.real < e_window[1]):
                continue
            # 与下一个 η 的最近邻态比较
            if k + 1 < len(traj):
                ev2 = traj[k + 1][1]
                j = np.argmin(np.abs(ev2 - E))
                dE = (ev2[j] - E) / (etas[k + 1] - eta) * eta
            else:
                dE = np.nan
            cand.append((abs(E.imag), abs(dE), eta, E))
    if not cand:
        return None
    cand.sort(key=lambda t: (np.nan_to_num(t[1], nan=1e9)))
    best = cand[0]
    if verbose:
        print(f"  稳定化点: η = {best[2]:.4f}, E = {best[3].real:.6f} Ha, "
              f"Γ = {-2*best[3].imag*1e3:.4f} meV "
              f"({-2*best[3].imag*HA_EV:.4f} eV), |ηdE/dη| = {best[1]:.2e}")
    return best


# ---------------------------------------------------------------------------
# 验证: He⁻ ¹S Feshbach 共振 (文献 E ≈ 19.37 eV, Γ ≈ 0.9–1.1 eV)
# ---------------------------------------------------------------------------

def _cap_scan(mol, mo_coeff, ncas, nelec, r_cap, etas, ref_energy=None,
              verbose=True):
    """通用 CAP-CI 扫描: 返回 (etas, traj)。traj[k] = (eta, 复本征值数组)。"""
    return cap_ci_resonance(mol, mo_coeff, ncas, nelec, r_cap, etas=etas,
                            verbose=verbose)


def bound_state_check(basis="aug-cc-pVTZ", r_cap=8.0):
    """验证 1: 束缚态 (He+Li⁺) — 追踪接近 SCF 基态能量的本征值, Γ 应 ≈ 0。"""
    from pyscf import gto, scf
    print("=" * 74)
    print("CAP-CI 验证 1 (束缚态): He+Li⁺ @ R=5 bohr — Γ 应随 r_cap 增大趋于 0")
    print("=" * 74)
    mol = gto.M(atom="He 0 0 0; Li 0 0 5.0", basis=basis, spin=0, charge=1,
                unit="Bohr", verbose=0)
    mf = scf.RHF(mol); e_scf = mf.kernel()
    ncas = min(10, mol.nao)
    traj, etas = _cap_scan(mol, mf.mo_coeff, ncas, (2, 2), r_cap,
                           np.array([0.0002, 0.0005, 0.001, 0.002, 0.004]))
    print(f"  SCF 基态能量 = {e_scf:.6f} Ha (CAP-CI 追踪目标)")
    print(f"  {'eta':>8} {'最接近 SCF 的 E':>18} {'Γ (meV)':>12}")
    for eta, ev in traj:
        j = np.argmin(np.abs(ev - e_scf))
        print(f"  {eta:8.4f} {ev[j].real:18.6f} {-2*ev[j].imag*1e3:12.4f}")


def he_li_width(basis="aug-cc-pVTZ", R=5.556, ncas=9, r_cap=12.0,
                emin=None, emax=None):
    """He*+Li ²Σ⁺ 自电离宽度 (R 单位 Bohr), 输出 η 轨迹诊断。"""
    from pyscf import gto, scf
    print("=" * 74)
    print(f"CAP-CI: He*+Li 自电离宽度 @ R={R:.3f} bohr = {R*0.529177:.3f} Å")
    print(f"  活性空间 {ncas} 轨道, CAP r_cap={r_cap} bohr")
    print("  参考 MRCI (Pro_HeLi_Enhanced): Γ_2Σ(3.3 Å) ≈ 10.57 meV (峰值)")
    print("=" * 74)
    mol = gto.M(atom=f"He 0 0 0; Li 0 0 {R}", basis=basis, spin=3, unit="Bohr",
                verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    ncas = min(ncas, mol.nao)
    etas = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032])
    traj, etas = _cap_scan(mol, mf.mo_coeff, ncas, (3, 2), r_cap, etas)
    he_li = -9.601572
    if emin is None: emin = he_li - 0.08
    if emax is None: emax = he_li + 0.08
    print(f"  能量窗口 [{-emin+2*he_li:.4f}, ...]→ 相对 He*+Li 渐近 {emin-he_li:+.3f} ~ {emax-he_li:+.3f} Ha")
    print(f"  {'eta':>8} {'E_R (Ha)':>13} {'E_R-渐近 (eV)':>14} {'Γ (meV)':>11} {'|Im| rank':>10}")
    for eta, ev in traj:
        inw = (ev.real > emin) & (ev.real < emax)
        if not inw.any():
            print(f"  {eta:8.4f}   (窗口内无本征值)")
            continue
        sub = ev[inw]
        order = np.argsort(-np.abs(sub.imag))
        for rank, j in enumerate(order[:3]):
            E = sub[j]
            print(f"  {eta:8.4f} {E.real:13.6f} {(E.real-he_li)*HA_EV:14.4f} "
                  f"{-2*E.imag*1e3:11.4f} {rank:10d}")
    return traj, etas


def main():
    ap = argparse.ArgumentParser(description="CAP-CI 共振态宽度计算")
    ap.add_argument("--system", default="validation",
                    choices=["validation", "he-li"])
    ap.add_argument("--basis", default="aug-cc-pVTZ")
    ap.add_argument("--R", type=float, default=5.556, help="核间距 (Bohr)")
    ap.add_argument("--ncas", type=int, default=9)
    ap.add_argument("--rcap", type=float, default=12.0, help="CAP 起始半径 (Bohr)")
    a = ap.parse_args()
    if a.system == "validation":
        bound_state_check(a.basis, a.rcap)
    else:
        he_li_width(a.basis, a.R, a.ncas, a.rcap)


if __name__ == "__main__":
    main()
