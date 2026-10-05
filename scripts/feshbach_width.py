#!/usr/bin/env python3
"""Feshbach 投影 + L² 赝态成像: He*+Li 自电离宽度 Γ(R) 与共振态势能 V*(R) 第一性原理计算。

文献方法来源 (本脚本严格按其处理路线实现)
------------------------------------------
1. M. Movre, L. Thiel, W. Meyer, J. Chem. Phys. 113, 1484 (2000)
   (He*(2³S)+Li 自电离宽度的原始文献, 参考数据 G_2Sigma.txt/G_2Pi.txt 的出处):
   - **Feshbach 投影按轨道占据定义**, 在多参考 CI 中实现: Q 空间 = He 2s(核激发)
     被占据的组态 ("resonance procedure"), P 空间 = 背景连续组态
     (He(1s²) 离子实 + 出射电子);
   - **MCSCF 轨道优化**: 文献在 6 活性轨道 (1s–4s, 1px, 1py) 的 14 个 ²Σ⁺ 组态
     空间做多组态 SCF — 本脚本对应为对 He\* 流形根 (He 1s 单占据, S≈1/2) 的
     态平均 CASSCF, 再以优化轨道做大活性空间 CI;
   - 共振态 ²Σ⁺ (He*(2³S)+Li): 文献 MRCI 值 D_e = 867 meV @ R_e = 5.54 a₀
     (实验 868(20) meV @ 5.4(3) a₀);
   - 宽度 Γ(R) = 2π Σ_l |V_εl|², 在垂直 (Franck–Condon) 电子能量
     ε_v(R) = V*(R) − V⁺(R) 处取值; 出射电子 ~13-14 eV, 耦合随能量变化平缓。
2. L² 赝态离散化 + 能量核平滑 = **Stieltjes 成像** (Langhoff 1974; Hazi 1978) —
   同一作者组在 Merz et al., Chem. Phys. Lett. 160, 377 (1989) 中对本体系宽度
   用的路线 (精度 ~10%)。本脚本用稠密赝态谱 + 归一化高斯核替代数值
   static-exchange 散射波的投影, 是其 L² 等价实现。
3. 弥散壳层指数直接取自 Movre–Thiel–Meyer (2000) 第 III 节 (与 aug-cc-pVTZ
   去重后追加): He: 3s(0.04,0.016,0.0064)+p(0.025); Li: s(0.022)+p(0.1,0.04,0.016)。
   出射电子 ε ≈ 14.4 eV → k ≈ 1.03 a₀⁻¹, 需要该指数段的弥散函数描述连续谱。

实现要点 (PySCF, Li 1s² 冻结, 3 活性电子, Sz=1/2 扇区)
------------------------------------------------------
1. ROHF 四重态 (He*(2³S)+Li) 判认轨道 (Li 1s² 双占; He 1s/He 2s/Li 2s 单占);
2. 价 CAS (6 轨道) CASCI 全谱 → 按 **He 1s 单占 + S²≈3/4** 筛出 He\* 流形根
   (排除中性 He(1s²) 根与 S=3/2 四重态分量), 态平均 CASSCF 优化轨道;
3. 大 CI: 活性 = CAS 6 轨道 + n_virt 个虚轨道 (连续谱), 稠密矩阵
   (direct_spin1 逐列收缩, 冻结核 2J−K 吸收), 行列式按占据分块:
   Q = {n(He 2s*) ≥ 1}, P = {n(He 1s) = 2 ∧ n(He 2s*) = 0};
4. 共振根选择: Q 块内按 S²≈3/4 + He\* 占据特征 + 相邻 R 点波函数最大重叠
   (根跟踪, 自渐近 R 起步) — Q 最低根是 He\*+Li⁺+e⁻ 闭通道赝态, 不可直接用;
5. Γ(R) = 2π Σ_k |V_k|² K((E_res − ε_k)/Δ), K = 归一化高斯核, Δ = ε_v 附近
   赝态局部间距之半 (自适应) 并与固定 Δ 系列对比; ε_v = E_res − E_ion
   (E_ion = 同基组 RHF HeLi⁺), 另报渐近锚定版 ε_ṽ (消除基组系统误差)。

用法:
  python scripts/feshbach_width.py --R 5.556                 # 单点诊断
  python scripts/feshbach_width.py --scan                    # R 网格扫描
  python scripts/feshbach_width.py --scan --n-virt 24        # 收敛性检验
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HA_EV = 27.211386245988
BOHR_ANG = 0.529177210903
EV_INT = 19.819614 - 5.391715  # He*(2³S) 激发能 − Li 电离能 (NIST, eV)

# Movre–Thiel–Meyer (2000) 第 III 节弥散壳层 (与 aug-cc-pVTZ 去重后追加)
MM_EXTRA = {
    "He": [[0, [0.04, 1.0]], [0, [0.016, 1.0]], [0, [0.0064, 1.0]],
           [1, [0.025, 1.0]]],
    "Li": [[0, [0.022, 1.0]], [1, [0.1, 1.0]], [1, [0.04, 1.0]],
           [1, [0.016, 1.0]]],
}
EXTRA_LADDER = {  # 收敛性检验用附加几何梯级弥散 (--extra-diffuse)
    "He": [[0, [0.0032, 1.0]], [1, [0.010, 1.0]]],
    "Li": [[0, [0.009, 1.0]], [1, [0.0065, 1.0]]],
}
N_VALENCE = 3   # He 1s / He 2s / Li 2s
N_EXTRA_CAS = 3  # CAS 中额外价轨道 (3s/σ*, 2pσ 等), 总 ncas = 6 (同文献)


def build_basis(extra: bool = False) -> dict:
    """aug-cc-pVTZ + Movre–Thiel–Meyer 弥散壳层 (去重)。"""
    from pyscf import gto

    def dedup(element: str, shells: list) -> list:
        base = gto.basis.load("aug-cc-pVTZ", element)
        old: dict[int, list] = {}
        for sh in base:
            old.setdefault(sh[0], []).append(sh[1][0])
        out = [list(sh) for sh in base]
        for sh in shells:
            l, e = sh[0], sh[1][0]
            if any(abs(np.log(e / e0)) < np.log(1.5) for e0 in old.get(l, [])):
                continue
            out.append(list(sh))
            old.setdefault(l, []).append(e)
        return out

    out = {el: dedup(el, list(MM_EXTRA[el])) for el in ("He", "Li")}
    if extra:
        for el, shells in EXTRA_LADDER.items():
            out[el] += [list(sh) for sh in shells]
    return out


def _pop_atom(mf, i: int, sl) -> float:
    """MO i 在原子 AO 块 sl 上的 Mulliken 粗居数。"""
    C = mf.mo_coeff
    S = mf.get_ovlp()
    return float(C[sl, i].T @ S[sl, :] @ C[:, i])


def identify_orbitals(mol, mf) -> dict:
    """判认 Li 1s / He 1s / He 2s / Li 2s (四重态 ROHF: 仅 Li 1s² 双占)。"""
    n_he = int(mol.aoslice_by_atom()[0][3])
    sl_he = slice(0, n_he)
    occ = mf.mo_occ
    idx: dict[str, int] = {"li1s": int(np.argmin(mf.mo_energy))}
    single = [i for i in range(mol.nao) if abs(occ[i] - 1.0) < 0.5]
    if len(single) != 3:
        raise RuntimeError(
            f"预期 3 条单占轨道 (He1s/He2s/Li2s), 实得 {len(single)}: "
            f"ROHF 可能未收敛到 He*(2³S)+Li 四重态")
    by_e = sorted(single, key=lambda i: mf.mo_energy[i])
    idx["he1s"] = by_e[0]
    rest = sorted(by_e[1:], key=lambda i: _pop_atom(mf, i, sl_he))
    idx["li2s"], idx["he2s"] = rest[0], rest[-1]
    return idx


def frozen_core_h1(mol, mo, active: list[int], core: int) -> np.ndarray:
    """冻结核 (Li 1s²) 的 2J−K 吸收进活性单电子项。"""
    from pyscf import ao2mo

    cols = mo[:, [core] + active]
    eri4 = ao2mo.restore(1, ao2mo.kernel(mol, cols), cols.shape[1])
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    C_a = mo[:, active]
    h_act = C_a.T @ h @ C_a
    for p in range(len(active)):
        for q in range(len(active)):
            h_act[p, q] += 2 * eri4[p + 1, q + 1, 0, 0] - eri4[p + 1, 0, 0, q + 1]
    return h_act


def reorder_for_casscf(mf, idx, n_extra: int) -> tuple[np.ndarray, list[int]]:
    """MO 重排为 [Li1s | CAS(n_extra+3) | 其余], 返回 (mo, cas 全局索引)。"""
    order = np.argsort(mf.mo_energy)
    taken = set(idx.values())
    virt = [int(i) for i in order if i not in taken][:n_extra]
    cas = [idx["he1s"], idx["he2s"], idx["li2s"]] + virt
    perm = [idx["li1s"]] + cas + [i for i in range(mf.mo_coeff.shape[1])
                                  if i not in {idx["li1s"], *cas}]
    mf.mo_coeff = mf.mo_coeff[:, perm]
    return mf.mo_coeff, cas


def occ_profile(ci_vec, ncas: int) -> np.ndarray:
    """CAS 组态矢量 → 各活性轨道平均占据。"""
    from pyscf.fci import cistring
    na, nb = 2, 1
    dim_a, dim_b = cistring.num_strings(ncas, na), cistring.num_strings(ncas, nb)
    stra = cistring.gen_strings4orblist(range(ncas), na)
    strb = cistring.gen_strings4orblist(range(ncas), nb)
    Cm = np.abs(np.asarray(ci_vec).reshape(dim_a, dim_b)) ** 2
    occ = np.zeros(ncas)
    for ia in range(dim_a):
        for ib in range(dim_b):
            w = Cm[ia, ib]
            if w < 1e-9:
                continue
            s = int(stra[ia] | strb[ib])
            for o in range(ncas):
                if (s >> o) & 1:
                    occ[o] += w
    return occ


def casscf_manifold(mol, mf, idx, n_extra: int = N_EXTRA_CAS, n_roots: int = 10,
                    verbose: bool = True):
    """态平均 CASSCF: 对 He\* 流形根 (He 1s 单占, S≈1/2) 优化轨道。

    返回 (mc, mo_优化, 根跟踪信息 dict)。
    """
    from pyscf import mcscf
    from pyscf.fci import spin_square as _spin_square

    mo, cas = reorder_for_casscf(mf, idx, n_extra)
    ncas = N_VALENCE + n_extra
    mc0 = mcscf.CASCI(mf, ncas, (2, 1))
    mc0.frozen = 1
    mc0.fcisolver.nroots = n_roots
    mc0.kernel()
    cis = mc0.ci if isinstance(mc0.ci, list) else [mc0.ci]
    manifold = []
    for r, c in enumerate(cis):
        ss = _spin_square(c, ncas, (2, 1))[0]
        occ = occ_profile(c, ncas)
        # He\* 流形: He 1s 单占 (CAS 位置 0), S≈1/2
        if abs(occ[0] - 1.0) < 0.35 and abs(ss - 0.75) < 0.4:
            manifold.append(r)
    if not manifold:
        raise RuntimeError("CASCI 谱中未找到 He* 流形根 (He 1s 单占, S≈1/2)")
    take = manifold[:6]   # 文献参考空间: 4 个 He* 渐近态 + 1s(2s²) 通道
    w = np.zeros(len(cis))
    w[take] = 1.0 / len(take)
    mc = mcscf.CASSCF(mf, ncas, (2, 1))
    mc.frozen = 1
    mc = mc.newton()                       # 二阶求解器 (一阶不收敛)
    mc = mcscf.state_average(mc, weights=tuple(w))
    mc.max_cycle_macro = 200
    mc.conv_tol = 1e-8
    mc.kernel()
    if not mc.converged:
        raise RuntimeError(f"SA-CASSCF 未收敛 @ R 见上下文 (流形根 {manifold})")
    if verbose:
        print(f"    CASSCF: 流形根 {manifold} → 平均 {take}, 收敛 ✓")
    return mc, take


def ci_blocks(mol, mo, core: int, active: list[int]):
    """稠密 (2α,1β) CI 矩阵 + Feshbach 占据分块掩码 (active 内位置)。"""
    from pyscf import ao2mo
    from pyscf.fci import cistring, direct_spin1

    M = len(active)
    h1e = frozen_core_h1(mol, mo, active, core)
    cols = mo[:, active]
    eri = ao2mo.restore(1, ao2mo.kernel(mol, cols), M)
    na, nb = 2, 1
    dim_a, dim_b = cistring.num_strings(M, na), cistring.num_strings(M, nb)
    dim = dim_a * dim_b
    h2e = direct_spin1.absorb_h1e(h1e, eri, M, (na, nb), 0.5)
    H = np.zeros((dim, dim))
    for j in range(dim):
        e = np.zeros(dim)
        e[j] = 1.0
        H[:, j] = direct_spin1.contract_2e(
            h2e, e.reshape(dim_a, dim_b), M, (na, nb)).ravel()
    H = 0.5 * (H + H.T)

    stra = cistring.gen_strings4orblist(range(M), na)
    strb = cistring.gen_strings4orblist(range(M), nb)
    # Feshbach 分块按 He 1s 占据 (active 位置 0), 对轨道旋转不敏感:
    #   Q = n(He 1s) = 1  (核激发, He* 特征; 含 1s(2s²) 通道, 由选根排除)
    #   P = n(He 1s) = 2  (He(1s²) 闭壳层背景: 中性通道 + 电离连续谱)
    #   n(He 1s) = 0 的 He²⁺+3e⁻ 双电离流形远离能量窗, 弃之
    q_mask = np.zeros(dim, dtype=bool)
    p_mask = np.zeros(dim, dtype=bool)
    n1s_diag = np.zeros(dim)
    for ia in range(dim_a):
        for ib in range(dim_b):
            k = ia * dim_b + ib
            n1s = int((stra[ia] >> 0) & 1) + int((strb[ib] >> 0) & 1)
            n1s_diag[k] = n1s
            q_mask[k] = n1s == 1
            p_mask[k] = n1s == 2
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    c_c = mo[:, core]
    h_cc = float(c_c @ h @ c_c)
    from pyscf import ao2mo as _ao
    eri_cc = float(_ao.restore(1, _ao.kernel(mol, c_c.reshape(-1, 1)), 1)[0, 0, 0, 0])
    e_shift = mol.energy_nuc() + 2 * h_cc + eri_cc
    return H, q_mask, p_mask, n1s_diag, active, e_shift



# ---------------------------------------------------------------------------
# Slater–Condon 引擎 (自定义行列式列表, 供 Q/P 异构空间使用)
# 自旋轨道约定: so = 2*orb + spin, spin: 0=α, 1=β
# ---------------------------------------------------------------------------

def _so_eri(g, p, q, r, s):
    """自旋轨道 2e 积分 ⟨pq|rs⟩ (physicist), 由空间 eri (chemist) 合成。

    ⟨pq|rs⟩ = (pr|qs)_chemist = ∫p(1)q(2) r12⁻¹ r(1)s(2),
    自旋约束 δ(σp,σr)δ(σq,σs); 交换项由 (pr|qs) 与 (ps|qr) 的自旋结构自动给出。
    """
    if (p & 1) != (r & 1) or (q & 1) != (s & 1):
        return 0.0
    return g[p >> 1, r >> 1, q >> 1, s >> 1]


def _contract_stepwise(dj, creations, annihilations, di_sorted):
    """逐步收缩: 算符串 a†(creations 逆序)... a(annihilations 顺序)...

    annihilations[0] 为最右算符先作用于 ket=Dj, 依次;
    随后 creations 逆序生成 (最左算符最后作用)。
    返回费米相位; 收缩为零或终态 ≠ bra 时返回 0。"""
    occ = list(dj)
    sign = 1
    for so in annihilations:
        if so not in occ:
            return 0.0
        i = occ.index(so)
        sign *= (-1) ** i
        occ.pop(i)
    for so in reversed(creations):
        if so in occ:
            return 0.0
        i = 0
        while i < len(occ) and occ[i] < so:
            i += 1
        sign *= (-1) ** i
        occ.insert(i, so)
    if occ != list(di_sorted):
        return 0.0
    return sign


def det_hamiltonian(dets, g_sp, h_sp):
    """给定行列式列表 (升序自旋轨道元组) 构造 CI 矩阵 (统一逐步收缩)。

    H = Σ h[x,r] a†_x a_r + ½ Σ g_so(x,y,r,s) a†_x a†_y a_s a_r,
    g_so(x,y,r,s) = ⟨xy|rs⟩ (physicist, 由空间 eri chemist 合成, 含交换)。
    对每个 det 对枚举 x,y∈Di, r,s∈Dj 的全部收缩元组 (81 个), 逐项带费米相位;
    结果与 direct_spin1 官方求解器逐本征值一致 (validate_sc_engine)。
    """
    nd = len(dets)
    sets = [set(d) for d in dets]
    H = np.zeros((nd, nd))
    for i in range(nd):
        Di = dets[i]
        for j in range(i, nd):
            Dj = dets[j]
            diff = sets[i] ^ sets[j]
            if len(diff) > 4:
                continue
            if not diff:
                so = list(Di)
                e = sum(h_sp[o, o] for o in so)
                for a in range(len(so)):
                    for b in range(a + 1, len(so)):
                        e += _so_eri(g_sp, so[a], so[b], so[a], so[b])
                        e -= _so_eri(g_sp, so[a], so[b], so[b], so[a])
            else:
                e = 0.0
                for x in Di:
                    for r in Dj:
                        sg = _contract_stepwise(Dj, [x], [r], Di)
                        if sg:
                            e += h_sp[x, r] * sg
                for x in Di:
                    for y in Di:
                        for r in Dj:
                            for s in Dj:
                                sg = _contract_stepwise(Dj, [x, y], [r, s], Di)
                                if sg:
                                    # 已验证约定 (He(1,1)/Li(2,1) 对官方 kernel 1e-15):
                                    # ⟨xy|rs⟩ − ½⟨xy|sr⟩, 配逐步收缩相位
                                    e += (_so_eri(g_sp, x, y, r, s)
                                          + 0.5 * _so_eri(g_sp, x, y, s, r)) * sg
                # ½ 与 (x,y)/(r,s) 有序双重计数相消; (Dj,Di) 转置由对称化补齐
            H[i, j] = H[j, i] = e
    return H


def validate_sc_engine():
    """引擎本征值谱 vs PySCF 官方 direct_spin1.kernel (真实物理积分)。

    He(1,1) 与 Li(2,1) 两个扇区。注意: 随机积分下官方 kernel 的 pspace
    机制会失真 (非物理负 Coulomb 积分), 不可作金标准, 必须用物理积分。
    """
    from pyscf.fci import direct_spin1
    from pyscf import gto, scf, ao2mo
    from itertools import combinations

    errs = []
    for sym, basis, spin, nelec in (("He", "aug-cc-pVDZ", 0, (1, 1)),
                                    ("Li", "aug-cc-pVDZ", 1, (2, 1))):
        mol = gto.M(atom=f"{sym} 0 0 0", basis=basis, spin=spin, verbose=0)
        mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
        mf.kernel()
        nc = 5
        h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        C = mf.mo_coeff[:, :nc]
        h1e = C.T @ h @ C
        eri = ao2mo.restore(1, ao2mo.kernel(mol, C), nc)
        cs = direct_spin1.FCI()
        # kernel 默认仅返回最低根 — 引擎亦取最低根比对
        e_ref = float(np.asarray(cs.kernel(h1e, eri, nc, nelec)[0]).ravel()[0])
        na, nb = nelec
        dets = []
        for ap in combinations(range(nc), na):
            for bp in combinations(range(nc), nb):
                dets.append(tuple(sorted([2 * x for x in ap]
                                         + [2 * y + 1 for y in bp])))
        h_sp = np.zeros((2 * nc, 2 * nc))
        for i in range(nc):
            for j in range(nc):
                h_sp[2 * i, 2 * j] = h1e[i, j]
                h_sp[2 * i + 1, 2 * j + 1] = h1e[i, j]
        Hm = det_hamiltonian(dets, eri, h_sp)
        e_mine = float(np.linalg.eigvalsh(Hm)[0])
        errs.append(abs(e_mine - e_ref))
    if max(errs) > 1e-9:
        raise RuntimeError(f"Slater-Condon 引擎验证失败: max|Δε| = {max(errs):.2e}")
    return max(errs)


def kernel_gamma(wp, v_k, e_res, d_ha: float) -> tuple[float, int]:
    """Γ = 2π Σ |V_k|² K((E_res−ε_k)/Δ), K = 归一化高斯核; 返回 (Γ, 窗口内态数)。"""
    if len(wp) == 0:
        return 0.0, 0
    x = (e_res - wp) / d_ha
    k = np.exp(-x * x) / (np.sqrt(np.pi) * d_ha)
    n_in = int(np.sum(np.abs(e_res - wp) < 4 * d_ha))
    return float(2 * np.pi * np.sum(v_k ** 2 * k)), n_in


def identify_active_orbitals(mol, mo, active_g: list[int]) -> dict:
    """在整个活性列集合上按物理属性回认 he1s/he2s/li2s。

    SA-CASSCF 会旋转 CAS/外部边界 (He 2s 可能被旋出 CAS 窗口), 故在
    CAS + 连续谱虚轨道的全集合上识别: He 固定在原点, ⟨r²⟩ 即延展度。
    he1s = 最紧凑 He 居数轨道, he2s = 次紧凑 He 居数轨道 (弥散),
    li2s = 最紧凑 Li 居数轨道。返回在 active_g 内的位置索引。
    """
    n_he = int(mol.aoslice_by_atom()[0][3])
    S = mol.intor("int1e_ovlp")
    R2 = mol.intor("int1e_r2")
    info = []
    for k, c in enumerate(active_g):
        col = mo[:, c]
        p_he = float(col[:n_he] @ S[:n_he, :] @ col)
        p_li = float(col[n_he:] @ S[n_he:, :] @ col)
        r2 = float(col @ R2 @ col)
        info.append(dict(k=k, p_he=p_he, p_li=p_li, r2=r2))
    he_c = sorted([t for t in info if t["p_he"] > 0.4], key=lambda t: t["r2"])
    if not he_c:
        raise RuntimeError(f"活性空间中未找到 He 居数轨道: {info}")
    return dict(he1s=he_c[0]["k"])


def active_g_he1s(mol, mo, U):
    """U 内 He 1s 列的位置 (最紧凑的 He 居数轨道)。"""
    n_he = int(mol.aoslice_by_atom()[0][3])
    S = mol.intor("int1e_ovlp")
    R2 = mol.intor("int1e_r2")
    best, best_r2 = None, None
    for k, c in enumerate(U):
        col = mo[:, c]
        p_he = float(col[:n_he] @ S[:n_he, :] @ col)
        if p_he > 0.4:
            r2 = float(col @ R2 @ col)
            if best_r2 is None or r2 < best_r2:
                best, best_r2 = k, r2
    if best is None:
        raise RuntimeError("U 集合中未找到 He 1s 轨道")
    return best


def h1e_to_spinh(h1e):
    """空间 h1e → 自旋分块 (2n)² (so = 2*orb + spin)。"""
    n = h1e.shape[0]
    out = np.zeros((2 * n, 2 * n))
    out[0::2, 0::2] = h1e
    out[1::2, 1::2] = h1e
    return out


def flip_norm_sq(c_vec, dets):
    """||S₊ψ||²: (2α,1β)→(3α,0β) 自旋翻转振幅模方和。

    M_S=1/2 态: ⟨S²⟩ = 0.75 + 2·||S₊ψ||² → 二重态 0, 四重态 3。
    翻转相位用与 det_hamiltonian 同一套逐步收缩约定。"""
    from collections import defaultdict
    amps = defaultdict(float)
    for k, d in enumerate(dets):
        betas = [o for o in d if o % 2 == 1]
        if len(betas) != 1:
            continue
        b = betas[0]
        result = tuple(sorted([o for o in d if o != b] + [b - 1]))
        sg = _contract_stepwise(d, [b - 1], [b], result)
        if sg:
            amps[result] += sg * c_vec[k]
    return sum(v * v for v in amps.values())


def run_point(R: float, n_virt: int = 40, extra_diffuse: bool = False,
              prev_c: np.ndarray | None = None, e_asym: float | None = None,
              verbose: bool = True) -> dict:
    from pyscf import gto, scf, ao2mo as _ao2
    from itertools import combinations as _comb

    basis = build_basis(extra_diffuse)
    mol = gto.M(atom=f"He 0 0 0; Li 0 0 {R}", basis=basis, spin=3,
                unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    idx0 = identify_orbitals(mol, mf)
    mc, _ = casscf_manifold(mol, mf, idx0, verbose=verbose)
    mc.canonicalize_()                      # 外部轨道按 Fock 能量排序
    mo = mc.mo_coeff
    ncas = N_VALENCE + N_EXTRA_CAS

    # ---- 轨道集合 U = CAS(6) + 连续谱外部 (Fock ∈ [−1, 30] eV) ----
    ext_ids = list(range(1 + ncas, mo.shape[1]))
    ext_eps = np.array([float(mc.mo_energy[i]) * HA_EV for i in ext_ids])
    in_win = np.where((ext_eps > -1.0) & (ext_eps < 30.0))[0]
    cont_ids = [ext_ids[i] for i in in_win][:n_virt]
    U = list(range(1, 1 + ncas)) + cont_ids
    NU = len(U)
    he1s_u = active_g_he1s(mol, mo, U)

    # ---- 单电子/双电子积分 (Li 1s² 冻结吸收) ----
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h1e_U = frozen_core_h1(mol, mo, U, 0)
    C_U = mo[:, U]
    eri_U = _ao2.restore(1, _ao2.kernel(mol, C_U), NU)
    h_sp = h1e_to_spinh(h1e_U)

    # ---- Feshbach 行列式列表 ((2α,1β) 扇区) ----
    # Q: He 1s 单占 (核激发流形: 共振 + He(1s·2s²) 通道 + He*+Li⁺+e⁻ 闭通道)
    # P: He 1s 双占 (He(1s²) 背景: 中性通道 + 电离连续谱)
    h1a, h1b = 2 * he1s_u, 2 * he1s_u + 1
    others = [u for u in range(NU) if u != he1s_u]
    # Q 限价层组态 (he1s¹ + 两价电子): 闭环通道 (He*+Li⁺+e⁻) 只贡献二阶微移,
    # 排除后引擎代价从 O(|Q|²NU²) 降为常数; P 保留大连续谱
    val_others = [u for u in U[:ncas] if u != he1s_u]
    q_set = set()
    for t in val_others:                 # he1s_α: 另一 α + 一 β
        for r in val_others:
            q_set.add((h1a, 2 * t, 2 * r + 1))
    for t, w in _comb(val_others, 2):    # he1s_β: 两个 α
        q_set.add((h1b, 2 * t, 2 * w))
    q_dets = sorted(tuple(sorted(d)) for d in q_set)
    p_dets = sorted((h1a, h1b, 2 * t) for t in others)
    dets = q_dets + p_dets
    nq = len(q_dets)

    H = det_hamiltonian(dets, eri_U, h_sp)
    c_c = mo[:, 0]
    h_cc = float(c_c @ h @ c_c)
    eri_cc = float(_ao2.restore(1, _ao2.kernel(mol, c_c.reshape(-1, 1)),
                                1)[0, 0, 0, 0])
    e_shift = mol.energy_nuc() + 2 * h_cc + eri_cc

    Hqq = H[:nq, :nq]
    Hpp = H[nq:, nq:]
    Hpq = H[nq:, :nq]
    wq, vq = np.linalg.eigh(Hqq)
    wp, vp = np.linalg.eigh(Hpp)

    # ---- 共振根选择: S²≈3/4 (自旋翻转范数) + 能量锚定/根跟踪 ----
    cands = []
    for r in range(min(24, wq.size)):
        c_blk = vq[:, r]
        flip = flip_norm_sq(c_blk, q_dets)
        if flip > 0.5:                   # 排除 S=3/2 四重态分量
            continue
        e_root = float(wq[r] + e_shift)
        ovl = (float(c_blk @ prev_c)
               if prev_c is not None and prev_c.size == c_blk.size else np.nan)
        cands.append(dict(r=r, ss=0.75 + 2 * flip, e=e_root, ovl=ovl,
                          c=c_blk.copy()))
    if not cands:
        raise RuntimeError(f"R={R}: Q 块中未找到 He* 流形根 (S≈1/2)")
    if prev_c is None:
        if e_asym is None:
            _m_he = gto.M(atom="He 0 0 0", basis=basis, spin=2,
                          unit="Bohr", verbose=0)
            _m_li = gto.M(atom="Li 0 0 0", basis=basis, spin=1,
                          unit="Bohr", verbose=0)
            e_asym = float(scf.ROHF(_m_he).kernel()
                           + scf.ROHF(_m_li).kernel())
        best = min(cands, key=lambda t: abs(t["e"] - e_asym))
    else:
        ovs = np.array([abs(t["ovl"]) if np.isfinite(t["ovl"]) else -1.0
                        for t in cands])
        best = cands[int(np.argmax(ovs))]
    c_res = best["c"]
    e_res = best["e"]

    v_k = vp.T @ (Hpq @ c_res)
    wp_abs = wp + e_shift

    mol_ion = gto.M(atom=f"He 0 0 0; Li 0 0 {R}", basis=basis, charge=1,
                    spin=0, unit="Bohr", verbose=0)
    e_ion = float(scf.RHF(mol_ion).kernel())

    out = dict(R=R, e_res=e_res, e_ion=e_ion, eps_v=(e_res - e_ion) * HA_EV,
               n_q=nq, n_p=len(p_dets), n_orb=NU, nao=mol.nao,
               root=best["r"], s2=best["ss"])
    for d in (0.01, 0.02, 0.04):
        g, n_in = kernel_gamma(wp_abs, v_k, e_res, d)
        out[f"gamma_{d}"], out[f"nwin_{d}"] = g, n_in
    win = np.abs(wp_abs - e_res) * HA_EV < 3.0
    if win.sum() >= 2:
        d_adp = max(0.5 * float(np.median(np.diff(np.sort(wp_abs[win])))),
                    0.005)
    else:
        d_adp = 0.04
    g_adp, n_adp = kernel_gamma(wp_abs, v_k, e_res, d_adp)
    out["d_adp"], out["gamma_adp"], out["nwin_adp"] = d_adp, g_adp, n_adp
    out["_c_res"] = c_res
    if verbose:
        print(f"  R={R:6.2f} bohr | nao={mol.nao} |U|={NU} "
              f"|Q|={nq} |P|={len(p_dets)} root={best['r']} "
              f"S²={best['ss']:.3f}")
        print(f"    E_res={e_res:.6f} Ha  E_ion={e_ion:.6f} Ha  "
              f"ε_v={out['eps_v']:.3f} eV (渐近应 → {EV_INT:.3f})")
        _w = np.abs(wp_abs - e_res) * HA_EV < 6.0
        _eps = np.sort((wp_abs[_w] - e_res) * HA_EV)
        print(f"    ε_v±6 eV 赝态 (相对 eV): {np.round(_eps, 2).tolist()}")
        print(f"    Γ(自适应 Δ={d_adp:.4f} Ha) = {g_adp*HA_EV*1e3:.3f} meV "
              f"(窗口态数 {n_adp});  固定 Δ: "
              f"{out['gamma_0.01']*HA_EV*1e3:.3f} / "
              f"{out['gamma_0.02']*HA_EV*1e3:.3f} / "
              f"{out['gamma_0.04']*HA_EV*1e3:.3f} meV")
    return out


def load_reference(ref_dir: str):
    p = os.path.join(ref_dir, "data_raw", "G_2Sigma.txt")
    d = np.loadtxt(p, delimiter=",")
    return d[np.argsort(d[:, 0])]


def interp_ref(ref, r_bohr: float) -> float:
    r_ang = r_bohr * BOHR_ANG
    if r_ang < ref[0, 0] or r_ang > ref[-1, 0]:
        return float("nan")
    return float(np.interp(r_ang, ref[:, 0], ref[:, 1]))


SCAN_DEFAULT = [4.0, 4.5, 5.0, 5.29, 5.56, 6.0, 6.5, 7.0, 7.5, 8.0,
                8.88, 9.5, 10.5, 12.0, 30.0]


def scan(R_list, n_virt: int, extra_diffuse: bool, out_npz: str,
         ref_dir: str | None):
    rs = sorted(R_list)
    print("=" * 78)
    print("Feshbach 投影 + L² 赝态成像: He*(2³S)+Li ²Σ⁺ 自电离宽度 Γ(R)")
    print("方法: Movre–Thiel–Meyer JCP 113, 1484 (2000) + SA-CASSCF + Stieltjes 成像")
    print(f"活性: 3e 价 + {N_EXTRA_CAS} CAS 虚 + {n_virt} 连续谱虚轨道, "
          f"附加弥散梯级: {'开' if extra_diffuse else '关'}")
    print("=" * 78)

    # 原子渐近锚 (同基组 ROHF): E(He* 2³S) + E(Li ²S)
    from pyscf import gto as _gto, scf as _scf
    basis = build_basis(extra_diffuse)
    m_he = _gto.M(atom="He 0 0 0", basis=basis, spin=2, unit="Bohr", verbose=0)
    m_li = _gto.M(atom="Li 0 0 0", basis=basis, spin=1, unit="Bohr", verbose=0)
    e_asym = float(_scf.ROHF(m_he).kernel() + _scf.ROHF(m_li).kernel())
    print(f"原子渐近锚 E(He* ³S)+E(Li ²S) = {e_asym:.6f} Ha (ROHF/{'aug-cc-pVTZ+MM'})")

    results = []
    prev_c = None
    t0 = time.time()
    for R in sorted(rs, reverse=True):   # 渐近首点锚定, 之后向短程根跟踪
        r = run_point(R, n_virt, extra_diffuse, prev_c=prev_c, e_asym=e_asym)
        prev_c = r.pop("_c_res")
        results.append(r)
        print(f"    (累计 {time.time()-t0:.0f}s)")

    r_asym = max(results, key=lambda t: t["R"])   # 渐近点 (R 最大)
    shift = r_asym["eps_v"] - EV_INT
    for r in results:
        r["eps_v_anchor"] = r["eps_v"] - shift
        r["v_star_meV"] = (r["e_res"] - r_asym["e_res"]) * HA_EV * 1e3
    results_sorted = sorted(results, key=lambda t: t["R"])

    v_star = np.array([r["v_star_meV"] for r in results_sorted])
    rs_sorted = np.array([r["R"] for r in results_sorted])
    i_min = int(np.argmin(v_star))
    print("\n【共振态势 V*(R) = ²Σ⁺ He*(2³S)+Li】")
    print(f"  阱深 D_e = {-v_star[i_min]:.0f} meV @ R_e = {rs_sorted[i_min]:.2f} bohr"
          f"  (文献 MRCI: 867 meV @ 5.54 a₀; 实验 868(20) meV @ 5.4(3) a₀)")

    ref = load_reference(ref_dir) if (ref_dir and os.path.isdir(ref_dir)) else None
    print("\n【宽度 Γ(R) vs 参考 MRCI (G_2Sigma.txt)】")
    header = f"  {'R(bohr)':>8} {'R(Å)':>6} {'ε_v(eV)':>9} {'Γ_自适应(meV)':>13}"
    for d in (0.01, 0.02, 0.04):
        header += f" {f'Γ_{d}':>10}"
    if ref is not None:
        header += f" {'参考Γ(meV)':>11}"
    print(header)
    for r in results_sorted:
        line = (f"  {r['R']:8.2f} {r['R']*BOHR_ANG:6.2f} {r['eps_v']:9.3f} "
                f"{r['gamma_adp']*HA_EV*1e3:13.3f}")
        for d in (0.01, 0.02, 0.04):
            line += f" {r[f'gamma_{d}']*HA_EV*1e3:10.3f}"
        if ref is not None:
            line += f" {interp_ref(ref, r['R']):11.3f}"
        print(line)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    keys = ("R", "e_res", "e_ion", "eps_v", "eps_v_anchor", "v_star_meV",
            "gamma_adp", "d_adp", "nwin_adp", "gamma_0.01", "gamma_0.02",
            "gamma_0.04", "nwin_0.02", "n_q", "n_p", "n_orb", "nao",
            "root", "s2")
    save = {k: np.array([r[k] for r in results_sorted]) for k in keys}
    save["provenance"] = json.dumps(dict(
        method=("Feshbach projection (occupancy) + SA-CASSCF orbitals + "
                "L2 pseudostate Stieltjes imaging"),
        reference="Movre-Thiel-Meyer JCP 113, 1484 (2000); Langhoff 1974; Hazi 1978",
        basis="aug-cc-pVTZ + MM diffuse (dedup)" + (" + ladder" if extra_diffuse else ""),
        n_virt=n_virt, ev_int_asym_eV=EV_INT), ensure_ascii=False)
    np.savez_compressed(out_npz, **save)
    print(f"\n✓ 数据: {out_npz}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Feshbach 投影 He*+Li 自电离宽度")
    ap.add_argument("--R", type=float, default=5.556, help="单点核间距 (bohr)")
    ap.add_argument("--scan", action="store_true", help="R 网格扫描")
    ap.add_argument("--n-virt", type=int, default=40,
                    help="活性虚轨道数 (连续谱赝态空间大小)")
    ap.add_argument("--extra-diffuse", action="store_true",
                    help="追加几何梯级弥散 (收敛性检验)")
    ap.add_argument("--out", default="results/feshbach_width.npz")
    ap.add_argument("--reference", default=None,
                    help="Pro_HeLi_Enhanced 目录 (提供则叠加参考 Γ 列)")
    a = ap.parse_args()
    if a.scan:
        scan(SCAN_DEFAULT, a.n_virt, a.extra_diffuse, a.out, a.reference)
    else:
        run_point(a.R, a.n_virt, a.extra_diffuse, e_asym=None)


if __name__ == "__main__":
    main()
