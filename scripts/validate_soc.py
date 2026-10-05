#!/usr/bin/env python3
"""自旋-轨道耦合 (单电子 Breit–Pauli) 验证 — 服务器运行。

判据:
  A. **精确标定**: 类氢离子 (Z=1,2,3) 的 SOC 常数 ζ 必须 = α²Z⁴/48
     (由 <r⁻³> = Z³/24 精确得出); 同时检验 Z⁴ 标度与基组收敛。
  B. **原子精细结构**: F/Cl/Br ²P 分裂 (3/2)ζ vs 实验 404.14/882.36/3685.3 cm⁻¹,
     报告单电子 BP 的**逐元素比值** (二电子项未实现的如实证据)。
  C. **原点/旋转不变性**: 平移到任意位置后 SOC 矩阵不变; 全局旋转后耦合
     矢量模长不变 (检验逐原子 l_A/r_A³ 构造与三分量一致性)。
  D. **对称性选择定则** (CH₂, C₂ᵥ): X̃³B₁–ã¹A₁ 耦合只有 B₁ 分量非零
     (h_z 分量严格为零), 且 M=+1/−1 分量等量 (Wigner–Eckart)。
  E. **两层交叉验证 + 实验**: OH ²Π 的轨道层 ζ 与 CI 层 SOC 必须一致,
     并 vs 实验 A = 139.2 cm⁻¹; 给出 A(R) 与精细结构分辨势 V_±(R)。
  F. **自旋纯度与 Hermitian 性**: CASCI 态 <S²> 纯净 (fix_spin_), SOC 矩阵 Hermitian。
"""
from __future__ import annotations
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.soc import (  # noqa: E402
    ALPHA, AU2CM, soc_integrals, soc_integrals_mo, soc_zeta,
    splitting_atomic_p, splitting_pi, soc_matrix_states, so_coupled_energies,
    auto_soc_orbitals, casci_ci_vectors, trans_density_1body,
)

from pyscf import gto, scf, fci  # noqa: E402
from pyscf.fci import spin_op  # noqa: E402


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def mo_lchar(mol, mo):
    """每个 MO 的角动量期望特征 (系数平方加权)。"""
    ao_l = np.zeros(mol.nao, dtype=int)
    idx = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        ao_l[idx:idx + 2 * l + 1] = l
        idx += 2 * l + 1
    w = np.asarray(mo) ** 2
    return (w * ao_l[:, None]).sum(axis=0) / w.sum(axis=0)


def lowest_p_trio(mol, mf):
    """1 电子离子: 最低的 3 重简并 p 型 MO 组。"""
    lch = mo_lchar(mol, mf.mo_coeff)
    e = np.array(mf.mo_energy)
    virt = [i for i in range(mol.nao) if mf.mo_occ[i] < 0.5 and lch[i] > 0.8]
    virt.sort(key=lambda i: e[i])
    return virt[:3]


def scaled_p_basis(el, z, n_p, lo=-7, base=0.06):
    """非收缩偶温 p 基组 (每个 primitive 独立收缩, 按 Z² 缩放) + 3 个 s 函数。

    非收缩是必须的: 单收缩的固定径向形状无法逼近类氢 2p。
    """
    lines = [f"{el}    S",
             "    3.42525091  0.15432897",
             "    0.62391373  0.53532814",
             "    0.16885540  0.44463454",
             f"{el}    P"]
    for i, k in enumerate(range(lo, lo + n_p)):
        c = ["0.0"] * n_p
        c[i] = "1.0"
        lines.append("    " + f"{base * (2.0 ** k) * z * z:.10f}" + "  "
                     + "  ".join(c))
    return "\n".join(lines)


def bare_hydrogenic_p_orbitals(mol, n_take=3):
    """裸核哈密顿量 H = T + V_nuc 在 p 子空间的**精确**广义本征解。

    不能用 SCF 虚轨道: 1 电子体系的 RHF 虚轨道感受自身库仑屏蔽
    (F = h + J/2) → 有效核电荷变 Z/2 (实测轨道能量 = −(Z/2)²/8, ζ 偏小 8 倍),
    这是 SCF 假象而非积分错误。裸核本征值 = 精确类氢能级 −Z²/(2n²)。
    """
    from scipy.linalg import eigh as gen_eigh
    nao = mol.nao
    ao_l = np.zeros(nao, dtype=int)
    idx = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for _ in range(mol.bas_nctr(ib)):
            ao_l[idx:idx + 2 * l + 1] = l
            idx += 2 * l + 1
    pidx = np.nonzero(ao_l == 1)[0]
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    w, v = gen_eigh(H[np.ix_(pidx, pidx)], S[np.ix_(pidx, pidx)])
    order = np.argsort(w)
    pick = order[:n_take]
    C = np.zeros((nao, n_take))
    C[pidx[:, None], np.arange(n_take)[None, :]] = v[:, pick]
    return C, w[pick]


def ci_block(ci):
    return np.asarray(ci).ravel()


# ---------------------------------------------------------------------------
def test_A_hydrogenic_exact():
    sec("A. 精确标定: 类氢离子 ζ = α²Z⁴/48 (= (2/3)·精确精细结构分裂 α²Z⁴/32)")
    ok = True
    for z, el in ((1, "H"), (2, "He"), (3, "Li")):
        row = []
        e_err = []
        for n_p in (10, 14, 18):
            mol = gto.M(atom=f"{el} 0 0 0", basis=scaled_p_basis(el, z, n_p),
                        charge=z - 1, spin=1, verbose=0)
            C, w = bare_hydrogenic_p_orbitals(mol)
            # 轨道能量必须 = 精确类氢 2p (−Z²/8)
            e_err.append(abs(w.mean() + z * z / 8.0))
            h_mo = soc_integrals_mo(mol, C)
            zeta = soc_zeta(h_mo, [0, 1, 2])
            z_ex = ALPHA ** 2 * z ** 4 / 48.0 * AU2CM
            row.append((n_p, zeta, zeta / z_ex))
        good = (0.999 < row[-1][2] < 1.001) and (e_err[-1] < 5e-3)
        ok &= good
        ss = " | ".join(f"n_p={n}: 比 {r:.5f}" for n, v, r in row)
        print(f"  Z={z}: {ss} | 2p 轨道能误差 ≤ {max(e_err):.1e} Ha (精确 "
              f"{-z*z/8:.6f}) | ζ(n_p=18)={row[-1][1]:.5f} cm⁻¹  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    # 分裂关系: 3/2 ζ 应等于精确精细结构 α²Z⁴/32
    z = 2
    mol = gto.M(atom="He 0 0 0", basis=scaled_p_basis("He", 2, 18),
                charge=1, spin=1, verbose=0)
    C, _ = bare_hydrogenic_p_orbitals(mol)
    h_mo = soc_integrals_mo(mol, C)
    zeta = soc_zeta(h_mo, [0, 1, 2])
    split = splitting_atomic_p(zeta)
    exact_fs = ALPHA ** 2 * z ** 4 / 32.0 * AU2CM
    good = 0.99 < split / exact_fs < 1.01
    ok &= good
    print(f"  ²P 分裂 (3/2)ζ = {split:.5f} vs 精确精细结构 α²Z⁴/32 = "
          f"{exact_fs:.5f} cm⁻¹  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_B_atomic_fine_structure():
    sec("B. 原子 ²P 精细结构 vs 实验 (cc-pVTZ, 单电子 BP)")
    exp = {"F": 404.14, "Cl": 882.36, "Br": 3685.3}
    ratios = {}
    ok = True
    for el, e_exp in exp.items():
        mol = gto.M(atom=f"{el} 0 0 0", basis="cc-pVTZ", spin=1, verbose=0)
        mf = scf.ROHF(mol).run(conv_tol=1e-10)
        orb = auto_soc_orbitals(mol, mf)
        h_mo = soc_integrals_mo(mol, mf.mo_coeff)
        zeta = soc_zeta(h_mo, orb)
        split = splitting_atomic_p(zeta)
        r = split / e_exp
        ratios[el] = r
        print(f"  {el}: ζ={zeta:8.2f} → Δ(3/2)ζ = {split:8.2f} cm⁻¹ | 实验 "
              f"{e_exp:8.2f} | 比 {r:.3f}  (轨道 {orb})", flush=True)
        ok &= bool(0.5 < r < 2.0)
    spread = max(ratios.values()) / min(ratios.values())
    print(f"  三元素比值 F/Cl/Br = {ratios['F']:.3f}/{ratios['Cl']:.3f}/"
          f"{ratios['Br']:.3f} → 最大/最小 = {spread:.3f}", flush=True)
    print("  说明: 单电子 BP 无二电子屏蔽 → 比值系统性偏离 1; 比值的一致性"
          "说明差异可按比例因子归一。", flush=True)
    ok &= bool(spread < 1.6)
    return ok


def test_C_invariance():
    sec("C. 原点平移不变性 + 全局旋转不变性 (CH₂/DZ, CAS(2,2))")
    from pyscf import mcscf
    rc, rh = 1.1023, 2.0452   # C–H, H–H 距离 (Bohr 附近)
    xh = rh / 2.0
    zh = np.sqrt(max(rc ** 2 - xh ** 2, 1e-9))
    coords0 = np.array([[0.0, 0.0, 0.0],
                        [-xh, 0.0, zh], [xh, 0.0, zh]])
    basis = "cc-pvdz"

    def build(coords, rot=None):
        c = coords if rot is None else coords @ np.asarray(rot).T
        mol = gto.M(atom=[("C", c[0]), ("H", c[1]), ("H", c[2])],
                    basis=basis, unit="Bohr", spin=2, verbose=0)
        mf = scf.ROHF(mol).run(conv_tol=1e-10)   # 三重态参考 (2S=2)
        return mol, mf, c

    def soc_bundle(mol, mf):
        """返回 (vec_norm2, soc_matrix) —— 单-三态耦合。"""
        mo = mf.mo_coeff
        # 活性轨道: 最高两个部分占据/近简并轨道 = a1, b1
        orb = sorted([i for i in range(mol.nao) if 0 < mf.mo_occ[i] < 2])[:2]
        mc = mcscf.CASCI(mf, 2, (1, 1))
        mc.verbose = 0
        idx = list(orb)
        # sort_mo: 把活性轨道放到位
        mo_sorted = mcscf.sort_mo(mc, mo, idx, base=0)
        h_all = soc_integrals_mo(mol, mo)      # 未排序基上取活性块
        h_act = h_all[:, idx, :][:, :, idx]
        _, es, cis = casci_ci_vectors(mf, mo_sorted, 2, (1, 1), ss=0.0, nroots=1)
        _, et, cit = casci_ci_vectors(mf, mo_sorted, 2, (1, 1), ss=2.0, nroots=1)
        _, etp, citp = casci_ci_vectors(mf, mo_sorted, 2, (2, 0), ss=2.0, nroots=1)
        _, etm, citm = casci_ci_vectors(mf, mo_sorted, 2, (0, 2), ss=2.0, nroots=1)
        states = [
            {"ci": cis[0], "nelec": (1, 1), "label": "1A1"},
            {"ci": cit[0], "nelec": (1, 1), "label": "3B1(M=0)"},
            {"ci": citp[0], "nelec": (2, 0), "label": "3B1(M=+1)"},
            {"ci": citm[0], "nelec": (0, 2), "label": "3B1(M=-1)"},
        ]
        H = soc_matrix_states(h_act, states)
        vec = np.array([H[0, 1], H[0, 2], H[0, 3]])
        return float(np.sum(np.abs(vec) ** 2)), H, (es, et, etp, etm)

    mol0, mf0, c0 = build(coords0)
    n0, H0, _ = soc_bundle(mol0, mf0)
    # 平移 (任意矢量)
    shift = np.array([3.1, -2.4, 5.7])
    mol1, mf1, c1 = build(coords0 + shift)
    n1, H1, _ = soc_bundle(mol1, mf1)
    d_trans = float(np.abs(H0 - H1).max())
    # 旋转 (随机正交矩阵)
    rng = np.random.default_rng(20261004)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    mol2, mf2, c2 = build(coords0, rot=Q)
    n2, H2, _ = soc_bundle(mol2, mf2)
    d_rot = abs(n0 - n2) / max(n0, 1e-30)
    good = (d_trans < 1e-8) and (d_rot < 1e-6) and n0 > 1e-12
    print(f"  平移 SOC 矩阵最大偏差 = {d_trans:.2e} (<1e-8)", flush=True)
    print(f"  旋转耦合矢量模² : 原 {n0:.6e} vs 旋转后 {n2:.6e} → 相对差 "
          f"{d_rot:.2e} (<1e-6)", flush=True)
    print(f"  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_D_symmetry_rules():
    sec("D. 对称性选择定则 (CH₂ C₂ᵥ: X̃³B₁–ã¹A₁ 只有 B₁(y) 分量)")
    from pyscf import mcscf
    rc, rh = 1.1023, 2.0452
    xh = rh / 2.0
    zh = np.sqrt(max(rc ** 2 - xh ** 2, 1e-9))
    coords = np.array([[0.0, 0.0, 0.0], [-xh, 0.0, zh], [xh, 0.0, zh]])
    mol = gto.M(atom=[("C", coords[0]), ("H", coords[1]), ("H", coords[2])],
                basis="cc-pvdz", unit="Bohr", spin=2, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-10)
    orb = sorted([i for i in range(mol.nao) if 0 < mf.mo_occ[i] < 2])[:2]
    mc = mcscf.CASCI(mf, 2, (1, 1))
    mc.verbose = 0
    mo = mcscf.sort_mo(mc, mf.mo_coeff, orb, base=0)
    h_all = soc_integrals_mo(mol, mf.mo_coeff)   # 未排序基
    h_act = h_all[:, orb, :][:, :, orb]
    _, es, cis = casci_ci_vectors(mf, mo, 2, (1, 1), ss=0.0, nroots=1)
    _, et, cit = casci_ci_vectors(mf, mo, 2, (1, 1), ss=2.0, nroots=1)
    _, _, citp = casci_ci_vectors(mf, mo, 2, (2, 0), ss=2.0, nroots=1)
    _, _, citm = casci_ci_vectors(mf, mo, 2, (0, 2), ss=2.0, nroots=1)
    # <S²> 纯度检查
    s2_s = spin_op.spin_square(cis[0], 2, (1, 1))[0]
    s2_t = spin_op.spin_square(cit[0], 2, (1, 1))[0]
    states = [
        {"ci": cis[0], "nelec": (1, 1), "label": "1A1"},
        {"ci": cit[0], "nelec": (1, 1), "label": "3B1(M=0)"},
        {"ci": citp[0], "nelec": (2, 0), "label": "3B1(M=+1)"},
        {"ci": citm[0], "nelec": (0, 2), "label": "3B1(M=-1)"},
    ]
    H = soc_matrix_states(h_act, states)
    c_z = abs(H[0, 1])
    c_p = abs(H[0, 2])
    c_m = abs(H[0, 3])
    main = max(c_p, c_m)
    good = (c_z < 1e-6 * main) and abs(c_p - c_m) < 1e-6 * main and main > 1e-6
    print(f"  |c(M=0, z 分量)| = {c_z:.3e} cm⁻¹ → 必须 ≈ 0 (A₂ 对称性禁阻)", flush=True)
    print(f"  |c(M=+1)| = {c_p:.6f}, |c(M=-1)| = {c_m:.6f} cm⁻¹ → B₁(y) 分量, "
          f"两者等量", flush=True)
    print(f"  <S²> 单重态 = {s2_s:.6f} (应 0), 三重态 = {s2_t:.6f} (应 2)", flush=True)
    print(f"  Hermitian: 最大反对称偏差 = "
          f"{np.abs(H - H.conj().T).max():.2e}", flush=True)
    good &= (abs(s2_s) < 1e-6) and (abs(s2_t - 2.0) < 1e-6)
    print(f"  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_E_oh_two_layer():
    sec("E. OH ²Π: 轨道层 ζ vs CI 层 SOC (交叉验证) + 实验 A=139.2 cm⁻¹")
    from pyscf import mcscf
    r_oh = 1.8324  # Bohr (~0.97 Å)
    mol = gto.M(atom=[("O", 0, 0, 0), ("H", 0, 0, r_oh)], basis="cc-pVTZ",
                spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-10)
    pi = auto_soc_orbitals(mol, mf, n_take=2)
    if len(pi) < 2:
        print(f"  π 对识别失败: {pi}", flush=True)
        return False
    h_mo = soc_integrals_mo(mol, mf.mo_coeff)
    zeta_orb = soc_zeta(h_mo, pi)
    # CI 层: 同一轨道基上的 CASCI (3 电子, 2 轨道) 两个 Λ 分量
    mc = mcscf.CASCI(mf, 2, (2, 1))
    mc.verbose = 0
    mo = mcscf.sort_mo(mc, mf.mo_coeff, pi, base=0)
    h_act = soc_integrals_mo(mol, mf.mo_coeff)[:, pi, :][:, :, pi]
    _, e_ci, ci = casci_ci_vectors(mf, mo, 2, (2, 1), ss=0.75, nroots=2)
    states = [{"ci": ci[0], "nelec": (2, 1), "label": "Pi_a"},
              {"ci": ci[1], "nelec": (2, 1), "label": "Pi_b"}]
    H = soc_matrix_states(h_act, states)
    zeta_ci = abs(H[0, 1])
    d_rel = abs(zeta_ci - zeta_orb) / zeta_orb
    print(f"  轨道层 ζ_π = {zeta_orb:.4f} cm⁻¹ | CI 层 |<Π_a|H_SO|Π_b>| = "
          f"{zeta_ci:.4f} cm⁻¹ | 相对差 {d_rel:.2e}", flush=True)
    print(f"  ²Π 分裂 = |ζ| = {splitting_pi(zeta_orb):.4f} vs 实验 A = 139.2 "
          f"cm⁻¹ → 比 {splitting_pi(zeta_orb)/139.2:.3f}", flush=True)
    ok = (d_rel < 1e-6) and (0.5 < splitting_pi(zeta_orb) / 139.2 < 2.0)
    # A(R): 精细结构分辨势 V_±(R) = V(R) ∓ A(R)/2 (²Π_1/2 / ²Π_3/2)
    print("  A(R) 扫描 (SOC-PES 演示):", flush=True)
    rows = []
    for r in (1.70, 1.90, 2.10, 2.30):
        m = gto.M(atom=[("O", 0, 0, 0), ("H", 0, 0, r)], basis="cc-pVTZ",
                  spin=1, verbose=0)
        mfr = scf.ROHF(m).run(conv_tol=1e-10)
        hh = soc_integrals_mo(m, mfr.mo_coeff)
        oc = np.asarray(mfr.mo_occ); ee = np.asarray(mfr.mo_energy)
        pp = auto_soc_orbitals(m, mfr, n_take=2)
        zr = soc_zeta(hh, pp)
        v_pi = float(np.sum(oc * ee))     # 粗势能 (仅演示分裂, 不作定量)
        rows.append((r, zr, v_pi))
        print(f"    R={r:.2f} Bohr: A(R) = {zr:7.3f} cm⁻¹ → "
              f"分裂 V(²Π_3/2)−V(²Π_1/2) = {zr/219474.63:.6f} Ha", flush=True)
    if len(rows) >= 2:
        ok &= all(abs(zr) > 1.0 for _, zr, _ in rows)
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def test_F_spin_purity():
    sec("F. 自旋纯度 (fix_spin_) 与 SOC 矩阵 Hermitian 性 (N₂/STO-3G CAS(2,2))")
    ok = True
    mol = gto.M(atom="N 0 0 0; N 0 0 2.1", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    h_mo = soc_integrals_mo(mol, mf.mo_coeff)
    ncas = 2
    ncore = (mol.nelectron - 2) // 2
    h_all = soc_integrals_mo(mol, mf.mo_coeff)         # 未排序基
    h_act = h_all[:, ncore:ncore + 2, :][:, :, ncore:ncore + 2]
    _, e_s, ci_s = casci_ci_vectors(mf, mf.mo_coeff, ncas, (1, 1), ss=0.0,
                                    nroots=1)
    _, e_t, ci_t = casci_ci_vectors(mf, mf.mo_coeff, ncas, (2, 0), ss=2.0,
                                    nroots=1)
    s2s = spin_op.spin_square(ci_s[0], ncas, (1, 1))[0]
    s2t = spin_op.spin_square(ci_t[0], ncas, (2, 0))[0]
    states = [{"ci": ci_s[0], "nelec": (1, 1)},
              {"ci": ci_t[0], "nelec": (2, 0)}]
    H = soc_matrix_states(h_act, states)
    herm = np.abs(H - H.conj().T).max()
    print(f"  <S²>: singlet(M=+1 扇区) = {s2s:.6f}, triplet(M=0) = {s2t:.6f}",
          flush=True)
    print(f"  SOC 矩阵 Hermitian 偏差 = {herm:.2e}", flush=True)
    ok &= (abs(s2s) < 1e-6) and (abs(s2t - 2.0) < 1e-6) and (herm < 1e-12)
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def test_G_density_and_wet():
    sec("G. 跃迁密度对照 PySCF 参考 + Wigner–Eckart 三分量等量")
    from pyscf import mcscf
    ok = True
    # --- G1: 'aa'/'bb' 密度与 PySCF make_rdm1s 一致 (用 Hermitian 算符收缩比对,
    #         与密度矩阵转置约定无关) ---
    rng = np.random.default_rng(7)
    mol = gto.M(atom="N 0 0 0; N 0 0 2.1", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    ncas, nelec = 4, (3, 1)
    from pyscf.fci import direct_spin1
    from pyscf.fci import cistring
    na = len(cistring.gen_occslst(range(ncas), nelec[0]))
    nb = len(cistring.gen_occslst(range(ncas), nelec[1]))
    ci1 = rng.normal(size=(na, nb))
    ci2 = rng.normal(size=(na, nb))
    ref_a, ref_b = direct_spin1.make_rdm1s(ci1.ravel(), ncas, nelec)
    my_a = trans_density_1body("aa", ci1.ravel(), ci1.ravel(), ncas, nelec)
    my_b = trans_density_1body("bb", ci1.ravel(), ci1.ravel(), ncas, nelec)
    h = rng.normal(size=(ncas, ncas))
    h = 0.5 * (h + h.T)                     # Hermitian 测试算符
    v_ref_a = float(np.sum(h * ref_a))
    v_my_a = float(np.sum(h * my_a))
    v_ref_b = float(np.sum(h * ref_b))
    v_my_b = float(np.sum(h * my_b))
    d_a = abs(v_ref_a - v_my_a) / max(abs(v_ref_a), 1e-12)
    d_b = abs(v_ref_b - v_my_b) / max(abs(v_ref_b), 1e-12)
    # 跃迁密度 (两不同态)
    ra, rb = direct_spin1.trans_rdm1s(ci1.ravel(), ci2.ravel(), ncas, nelec)
    ta = trans_density_1body("aa", ci1.ravel(), ci2.ravel(), ncas, nelec)
    tb = trans_density_1body("bb", ci1.ravel(), ci2.ravel(), ncas, nelec)
    da = abs(float(np.sum(h * ra)) - float(np.sum(h * ta))) / max(
        abs(float(np.sum(h * ra))), 1e-12)
    db = abs(float(np.sum(h * rb)) - float(np.sum(h * tb))) / max(
        abs(float(np.sum(h * rb))), 1e-12)
    good1 = max(d_a, d_b, da, db) < 1e-10
    ok &= good1
    print(f"  'aa' 密度 vs PySCF: 相对差 对角 {d_a:.2e} / 跃迁 {da:.2e}", flush=True)
    print(f"  'bb' 密度 vs PySCF: 相对差 对角 {d_b:.2e} / 跃迁 {db:.2e}  "
          f"{'OK' if good1 else 'FAIL'}", flush=True)

    # --- G1b: CI 布局一致性 (迹恒等式): Σ_p D_aa[p,p] = nα, Σ_p D_bb = nβ
    #      CI 布局若错位 (地址与系数错配) 该恒等式立刻失败 ---
    def body_trace():
        from pyscf import mcscf as _mc
        m = gto.M(atom=[("O", 0, 0, 0), ("H", 0, 0, 1.8324)], basis="cc-pVTZ",
                  spin=1, verbose=0)
        mfr = scf.ROHF(m).run(conv_tol=1e-10)
        occ = np.asarray(mfr.mo_occ); ee = np.asarray(mfr.mo_energy)
        sm = [i for i in range(m.nao) if 0 < occ[i] < 2]
        pi = list(sm) + [i for i in range(m.nao)
                         if occ[i] < 0.5 and any(abs(ee[i] - ee[j]) < 5e-3
                                                for j in sm)]
        pi = sorted(set(pi))[:2]
        mc = _mc.CASCI(mfr, 2, (2, 1)); mc.verbose = 0
        mo = _mc.sort_mo(mc, mfr.mo_coeff, pi, base=0)
        _, e, ci = casci_ci_vectors(mfr, mo, 2, (2, 1), ss=0.75, nroots=2)
        d_a = trans_density_1body("aa", ci[0], ci[0], 2, (2, 1))
        d_b = trans_density_1body("bb", ci[0], ci[0], 2, (2, 1))
        return float(np.trace(d_a)), float(np.trace(d_b)), len(ci), ci[0].shape
    tr_a, tr_b, nci, cshape = self_run = body_trace()
    good_tr = abs(tr_a - 2.0) < 1e-8 and abs(tr_b - 1.0) < 1e-8
    ok &= good_tr
    print(f"  CI 布局恒等式: Σ D_aa = {tr_a:.10f} (应 2), Σ D_bb = {tr_b:.10f} "
          f"(应 1) | CI 数 {nci}, 形状 {cshape}  {'OK' if good_tr else 'FAIL'}",
          flush=True)

    # --- G2: WET 三分量等量 (低对称 CH2: 全部非零且等量) ---
    coords = np.array([[0.0, 0.0, 0.0],
                       [-1.02, 0.07, 1.86], [1.04, -0.05, 1.90]])   # 低对称
    mol2 = gto.M(atom=[("C", coords[0]), ("H", coords[1]), ("H", coords[2])],
                 basis="cc-pvdz", unit="Bohr", spin=2, verbose=0)
    mf2 = scf.ROHF(mol2).run(conv_tol=1e-10)
    orb = sorted([i for i in range(mol2.nao) if 0 < mf2.mo_occ[i] < 2])[:2]
    mc = mcscf.CASCI(mf2, 2, (1, 1))
    mc.verbose = 0
    mo = mcscf.sort_mo(mc, mf2.mo_coeff, orb, base=0)
    h_act = soc_integrals_mo(mol2, mf2.mo_coeff)[:, orb, :][:, :, orb]
    _, _, cis = casci_ci_vectors(mf2, mo, 2, (1, 1), ss=0.0, nroots=1)
    _, _, cit = casci_ci_vectors(mf2, mo, 2, (1, 1), ss=2.0, nroots=1)
    _, _, citp = casci_ci_vectors(mf2, mo, 2, (2, 0), ss=2.0, nroots=1)
    _, _, citm = casci_ci_vectors(mf2, mo, 2, (0, 2), ss=2.0, nroots=1)
    states = [{"ci": cis[0], "nelec": (1, 1)},
              {"ci": cit[0], "nelec": (1, 1)},
              {"ci": citp[0], "nelec": (2, 0)},
              {"ci": citm[0], "nelec": (0, 2)}]
    H = soc_matrix_states(h_act, states)
    v = np.array([abs(H[0, 1]), abs(H[0, 2]), abs(H[0, 3])])
    good2 = (v.min() > 1e-6) and ((v.max() - v.min()) / v.max() < 1e-8)
    ok &= good2
    print(f"  WET 三分量 |c(M=0)|,|c(M=+1)|,|c(M=-1)| = "
          f"{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f} cm⁻¹ (相对展宽 "
          f"{(v.max()-v.min())/v.max():.2e} < 1e-8)  "
          f"{'OK' if good2 else 'FAIL'}", flush=True)
    # 单重态-单重态 SOC 必须为零 (自旋选择定则)
    s_ss = abs(H[0, 0])
    good3 = s_ss < 1e-10
    ok &= good3
    print(f"  单重态对角 |<¹A₁|H_SO|¹A₁>| = {s_ss:.2e} (应 0)  "
          f"{'OK' if good3 else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__} | numpy: {np.__version__}", flush=True)
    print(f"α = {ALPHA:.12f} (1/α = {1/ALPHA:.6f})", flush=True)
    t0 = time.time()
    only = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    pool = [(nm, fn) for nm, fn in
            (("A", test_A_hydrogenic_exact),
             ("B", test_B_atomic_fine_structure),
             ("C", test_C_invariance),
             ("D", test_D_symmetry_rules),
             ("E", test_E_oh_two_layer),
             ("F", test_F_spin_purity),
             ("G", test_G_density_and_wet))
            if not only or nm in only]
    all_ok = True
    for _, fn in pool:
        try:
            all_ok &= bool(fn())
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            all_ok = False
    print(flush=True)
    print("=" * 78, flush=True)
    print(f"{'全部通过' if all_ok else '存在失败项'}  "
          f"(总耗时 {time.time() - t0:.0f}s)", flush=True)
    sys.exit(0 if all_ok else 1)
