#!/usr/bin/env python3
"""CAP-CI 共振搜索: 在谱中找 Γ 在 η 扫描下形成平台的态 (真实共振判据)。"""
import numpy as np
from pyscf import gto, scf
from cap_ci import cap_ci_resonance

HA_EV = 27.211386245988


def augmented_basis(base="aug-cc-pVTZ", extra_diffuse=True):
    """在 aug-cc-pVTZ 上追加额外弥散 s/p/d 壳层 (提升连续谱描述能力)。"""
    from pyscf import gto as _gto
    if not extra_diffuse:
        return base
    out = {}
    for el in ("He", "Li"):
        b = _gto.basis.load(base, el)
        b = list(b) + [[0, [0.030, 1.0]], [0, [0.012, 1.0]],
                       [1, [0.030, 1.0]], [2, [0.060, 1.0]]]
        out[el] = b
    return out


def search(R=6.236, basis="aug-cc-pVTZ", ncas=9, r_cap=6.0,
           etas=(0.001, 0.002, 0.004, 0.008, 0.016), extra_diffuse=False):
    mol = gto.M(atom=f"He 0 0 0; Li 0 0 {R}", basis=augmented_basis(basis, extra_diffuse),
                spin=3, unit="Bohr", verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    traj, etas = cap_ci_resonance(mol, mf.mo_coeff, min(ncas, mol.nao), (3, 2),
                                  r_cap, etas=np.array(etas))
    he_li = -9.601572
    print("=" * 78)
    print(f"共振搜索 @ R={R:.3f} bohr, ncas={ncas}, r_cap={r_cap} bohr")
    print("判据: 真实共振的 Γ 在 η 扫描下形成平台 (|dΓ/dη| 小); 连续谱赝态 Γ∝η")
    print("=" * 78)
    # 收集所有窗口内、Γ 在可观测范围的态, 按 η 分组
    rows = []
    for eta, ev in traj:
        for E in ev:
            g = -2 * E.imag * 1e3
            if 0.5 < g < 3000 and abs(E.real - he_li) < 0.15:
                rows.append((eta, E.real, g))
    rows.sort(key=lambda t: (t[0], t[1]))
    # 按 (E, Γ) 邻近性跨 η 追踪
    print(f"\n谱统计: 共 {len(rows)} 个窗口内态 (Γ ∈ 0.5–3000 meV)")
    print(f"{'eta':>8} {'Γ 分布 (meV, 前 12 个由小到大)':<70}")
    for eta in sorted(set(r[0] for r in rows)):
        gs = sorted(g for e, rr, g in rows if e == eta)
        s = " ".join(f"{g:7.2f}" for g in gs[:12])
        print(f"{eta:8.4f} {s}")
    # 平台检测: 对每个 η_k 的态, 在 η_{k+1} 找 Γ 相对变化最小的伙伴
    print("\n平台检测 (跨 η 追踪 Γ 稳定性, 只列 |ΔΓ/Γ| < 30% 的):")
    found = []
    et = sorted(set(r[0] for r in rows))
    for k in range(len(et) - 1):
        g1 = [(E, g) for e, E, g in rows if e == et[k]]        # (能量, Γ)
        g2 = [(E, g) for e, E, g in rows if e == et[k + 1]]
        for E1, gg1 in g1:
            best = None
            for E2, gg2 in g2:
                if abs(E2 - E1) > 0.02:      # 能量需连续 (Ha)
                    continue
                rel = abs(gg2 - gg1) / max(gg1, 1e-9)
                if best is None or rel < best[0]:
                    best = (rel, E2, gg2)
            if best and best[0] < 0.30:
                found.append((et[k], E1, gg1, best[0]))
                print(f"  η={et[k]:.4f}: E={E1:.6f} (相对渐近 {(E1-he_li)*HA_EV:+.3f} eV), "
                      f"Γ={gg1:8.3f} meV → η'={et[k+1]:.4f}: Γ={best[2]:8.3f} meV "
                      f"(ΔΓ/Γ={best[0]*100:5.1f}%)")
    if not found:
        print("  (无平台: 谱中所有态均表现为 Γ∝η 的连续谱赝态)")
    return found


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=6.236)
    ap.add_argument("--ncas", type=int, default=9)
    ap.add_argument("--rcap", type=float, default=6.0)
    ap.add_argument("--basis", default="aug-cc-pVTZ")
    ap.add_argument("--extra-diffuse", action="store_true",
                    help="追加额外弥散 s/p/d 壳层 (连续谱描述)")
    a = ap.parse_args()
    search(a.R, a.basis, a.ncas, a.rcap, extra_diffuse=a.extra_diffuse)
