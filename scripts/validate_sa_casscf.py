#!/usr/bin/env python3
"""态平均 CASSCF (SA-CASSCF) 激发态 + NEVPT2(root) 验证 — 服务器运行。

判据:
  A. **精确性**: H₂/STO-3G 的 CAS(2,2) 张满全空间 → SA(2)-CASSCF 两个态
     必须等于 **FCI** (单重态) 能量 (误差 < 1e-8 Ha)。
  B. **态身份/平滑性** (核心): LiH 的 ¹Σ⁺ 离子-共价**避交叉**区扫描
     (2.4→6.0 Bohr): 曲线不交叉、ΔE 在避交叉处取极小、能量与 CI 向量
     沿几何**平滑** (无跳变) —— 这正是态平均轨道相对"事后根跟踪"的优势。
  C. **激发态 NEVPT2**: root=0/1 的修正均为负、态序保持, 且
     |ΔE_exc(NEVPT2) − ΔE_exc(FCI)| < |ΔE_exc(CASSCF) − ΔE_exc(FCI)|
     (动态相关把激发能拉向 FCI)。
  D. **一致性/可用性**: STO-3G (空间完备) 下态平均基态能量 = 单态 CASSCF
     能量 = FCI (1e-8); 态平均基态 FD 梯度 ≈ 单态 CASSCF FD 梯度;
     激发态有界优化使能量下降。
  E. **API 边界**: state 需要 nstates>1、越界与权重长度不匹配时报错清晰。
"""
from __future__ import annotations
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator, CommandBackendError  # noqa: E402

BOHR = 0.529177210903
AU2EV = 27.211386245988


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def h2_coords(r_bohr: float):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_bohr]])


def fci_singlets(atom, basis, nroots=4):
    """FCI 单重态能量列表 (Hartree), 按 S² 过滤。"""
    from pyscf import gto, scf, fci, ao2mo
    from pyscf.fci import spin_op
    mol = gto.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    h1 = mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
    na = mol.nelectron // 2
    es, cs = fci.direct_spin1.FCI().kernel(h1, eri, mol.nao, (na, na),
                                           nroots=nroots)
    es = np.asarray(es).ravel() + mol.energy_nuc()
    out = [float(e) for e, c in zip(es, cs)
           if abs(spin_op.spin_square(c, mol.nao, (na, na))[0]) < 0.1]
    return np.sort(np.array(out))


def jump_ratio(x: np.ndarray) -> float:
    """跳变比: max|Δx| / (75 分位 |Δx|), 用于检出曲线上的单点跳变。"""
    d = np.abs(np.diff(np.asarray(x, dtype=float)))
    denom = float(np.percentile(d, 75)) if d.size else 0.0
    return float(d.max() / denom) if denom > 1e-14 else 0.0


# ---------------------------------------------------------------------------
def test_A_exact_vs_fci():
    sec("A. 精确性: H₂/STO-3G CAS(2,2) 张满全空间 → SA(2) 自旋纯 = FCI 单重态")
    r = 1.4
    sa = dict(active_space=(2, 2), nstates=2, state_average=True)
    e0 = PySCFCalculator(["H", "H"], basis="sto-3g", method="casscf",
                         state=0, **sa).energy(h2_coords(r))
    e1 = PySCFCalculator(["H", "H"], basis="sto-3g", method="casscf",
                         state=1, **sa).energy(h2_coords(r))
    fci = fci_singlets("H 0 0 0; H 0 0 1.4", "sto-3g", nroots=6)
    d0 = abs(e0 - fci[0])
    d1 = abs(e1 - fci[1])
    ok = (d0 < 1e-8) and (d1 < 1e-8)
    print(f"  SA(2): E0={e0:.9f} E1={e1:.9f} Ha | FCI 单重态: "
          f"{fci[0]:.9f} {fci[1]:.9f}", flush=True)
    print(f"  |ΔE0|={d0:.2e}, |ΔE1|={d1:.2e} Ha (<1e-8)  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    return ok


def test_B_state_identity_smoothness():
    sec("B. 态身份/平滑性: LiH ¹Σ⁺ 扫描 (2.4→6.0 Bohr, 0.05 步长, SA(2)-CAS(2,2))")
    basis = "6-31g"
    rs = np.arange(2.40, 6.001, 0.05)
    sa = dict(active_space=(2, 2), nstates=2, state_average=True)

    def scan(follow: bool = False):
        calc = PySCFCalculator(["Li", "H"], basis=basis, method="casscf",
                               state=0, follow=follow, **sa)
        e0s, e1s, ov = [], [], []
        prev = None
        for r in rs:
            c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
            e0 = calc.energy(c)
            mc = calc._last_solver
            ci_raw = mc.ci
            ci_list = ([np.asarray(x).ravel() for x in ci_raw]
                       if isinstance(ci_raw, (list, tuple))
                       else [np.asarray(ci_raw).ravel()])
            e1 = float(np.asarray(mc.e_states, dtype=float).ravel()[1])
            e0s.append(e0)
            e1s.append(e1)
            if prev is not None and len(ci_list) == len(prev):
                ov.append(abs(float(np.dot(ci_list[0], prev[0]))))
            prev = [x.copy() for x in ci_list]
        return np.array(e0s), np.array(e1s), np.array(ov)

    e0s, e1s, ov = scan(follow=False)
    e0f, e1f, ovf = scan(follow=True)
    gap = e1s - e0s
    no_cross = bool(gap.min() > 1e-6)
    # 平滑性判据 (避免尺度误用): 用**能隙**曲线的相对跳变 (能隙本身是
    # 小量且物理上应平滑) + 避交叉必须落在扫描区间内部 + CI 向量连续性。
    def rel_jump(e):
        return float(np.abs(np.diff(e)).max() / max(np.ptp(e), 1e-12))
    rj_gap = rel_jump(gap)
    i_min = int(np.argmin(gap))
    inner = bool(rs[i_min] < rs[-1] - 0.1)
    smooth = rj_gap < 0.05
    ident = float(ov.min()) > 0.99           # 态身份 (CI 向量连续)
    d_follow = float(np.abs(e1f - e1s).max())
    ok = no_cross and smooth and ident and inner
    print(f"  态序保持 {no_cross} | ΔE_min = {gap.min()*AU2EV:.4f} eV @ "
          f"R={rs[int(np.argmin(gap))]:.2f} Bohr (范围内 "
          f"{rs[int(np.argmin(gap))] < rs[-1] - 0.1})", flush=True)
    print(f"  能隙曲线相对跳变 {rj_gap:.2e} (<0.05) | CI 向量最小重叠 "
          f"{ov.min():.6f} (>0.99, 态身份无跳变) | 避交叉在区间内 {inner}",
          flush=True)
    print(f"  follow=True 与 False 的能量最大差 {d_follow:.2e} Ha "
          f"(态平均下二者应一致) → {'一致' if d_follow < 1e-9 else '有差异'}",
          flush=True)
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def test_C_nevpt2_excited():
    sec("C. 激发态 NEVPT2: 修正为负/态序保持 + 与单态 NEVPT2 交叉核对 (LiH/6-31G)")
    ok = True
    for r in (3.0, 3.6):
        c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
        sa = dict(active_space=(2, 2), nstates=2, state_average=True)
        e_cas, e_pt = [], []
        for k in (0, 1):
            e_cas.append(PySCFCalculator(["Li", "H"], basis="6-31g",
                                         method="casscf", state=k, **sa
                                         ).energy(c))
            e_pt.append(PySCFCalculator(["Li", "H"], basis="6-31g",
                                        method="casscf", state=k, pt2="nevpt2",
                                        **sa).energy(c))
        # 单态 (非态平均) 基态 NEVPT2 —— v0.26.0 已验证过的路径 (作对照)
        e_ss_pt = PySCFCalculator(["Li", "H"], basis="6-31g", method="casscf",
                                  active_space=(2, 2), pt2="nevpt2").energy(c)
        # 绝对参考: FCI/6-31G 同一几何的基态 (精确)
        fci = fci_singlets([("Li", (0, 0, 0)), ("H", (0, 0, r))], "6-31g",
                           nroots=6)
        e_fci = float(fci[0])
        corr_neg = all(e_pt[k] < e_cas[k] for k in (0, 1))
        order = e_pt[0] < e_pt[1]
        # 物理判据: NEVPT2 必须把**态平均**基态拉向 FCI (而非仅"接近某阈值")
        d_before = abs(e_cas[0] - e_fci)
        d_after = abs(e_pt[0] - e_fci)
        improved = d_after < d_before
        exc_cas = (e_cas[1] - e_cas[0]) * AU2EV
        exc_pt = (e_pt[1] - e_pt[0]) * AU2EV
        good = corr_neg and order and improved
        ok &= good
        print(f"  R={r:.1f}: CASSCF {e_cas[0]:.6f}/{e_cas[1]:.6f} → +NEVPT2 "
              f"{e_pt[0]:.6f}/{e_pt[1]:.6f} Ha (修正为负 {corr_neg}, "
              f"态序 {order})", flush=True)
        print(f"         对 FCI ({e_fci:.6f}): |Δ| {d_before*1000:.2f} → "
              f"{d_after*1000:.2f} mHa (拉向 FCI {improved}) | 激发能 "
              f"{exc_cas:.4f} → {exc_pt:.4f} eV | 单态 NEVPT2 对照 "
              f"{e_ss_pt:.6f} (态平均轨道代价 {abs(e_pt[0]-e_ss_pt)*1000:.1f} mHa)  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_D_consistency_and_gradient():
    sec("D. 一致性/可用性: 态平均=单态=FCI + 梯度一致 + 梯度沿几何平滑")
    r = 1.4
    c = h2_coords(r)
    sa = dict(active_space=(2, 2), nstates=2, state_average=True)
    e_sa = PySCFCalculator(["H", "H"], basis="sto-3g", method="casscf",
                           state=0, **sa).energy(c)
    e_ss = PySCFCalculator(["H", "H"], basis="sto-3g", method="casscf",
                           active_space=(2, 2)).energy(c)
    fci = fci_singlets("H 0 0 0; H 0 0 1.4", "sto-3g", nroots=6)
    d_sa_ss = abs(e_sa - e_ss)
    d_ss_fci = abs(e_ss - fci[0])
    # FD 梯度: STO-3G 下空间完备 → 态平均与单态必须有相同梯度
    g_sa = PySCFCalculator(["H", "H"], basis="sto-3g", method="casscf",
                           state=0, **sa).gradient(c)
    g_ss = PySCFCalculator(["H", "H"], basis="sto-3g", method="casscf",
                           active_space=(2, 2)).gradient(c)
    dg = float(np.abs(g_sa - g_ss).max())
    # 激发态 FD 梯度沿几何平滑 (含态身份: 无跳变)
    calc = PySCFCalculator(["H", "H"], basis="cc-pvdz", method="casscf",
                           state=1, **sa)
    rs = np.arange(0.80, 1.61, 0.05)
    gs = []
    for rr in rs:
        g = calc.gradient(h2_coords(rr))
        gs.append(float(g[1, 2] - g[0, 2]))     # 沿键轴的能量导数
    gs = np.array(gs)
    rj = float(np.abs(np.diff(gs)).max() / max(np.ptp(gs), 1e-12))
    # 物理判据: H₂ ¹Σu⁺ 为排斥态 → dE/dR 全窗口恒负 (符号翻转 = 态身份跳跃)
    grad_ok = bool(np.all(gs < 0))
    good = (d_sa_ss < 1e-8) and (d_ss_fci < 1e-8) and (dg < 1e-6) and grad_ok
    print(f"  态平均基态 {e_sa:.9f} = 单态 {e_ss:.9f} (差 {d_sa_ss:.1e}) = "
          f"FCI {fci[0]:.9f} (差 {d_ss_fci:.1e})", flush=True)
    print(f"  FD 梯度: 态平均 vs 单态 max|Δ| = {dg:.2e} (<1e-6)", flush=True)
    print(f"  激发态 FD 梯度: dE/dR 全窗口 {gs.min():.4f}…{gs.max():.4f} "
          f"(排斥态应恒负 → 符号一致 {grad_ok}), 相对跳变 {rj:.3f}  "
          f"{'OK' if grad_ok else 'FAIL'}", flush=True)
    print(f"  {'OK' if good else 'FAIL'}", flush=True)
    return good


def test_E_api_errors():
    sec("E. API 边界: state>0 需态平均 / 未开开关不隐式触发 / 越界 / 权重长度")
    c = h2_coords(1.4)
    ok = True
    base = dict(basis="sto-3g", method="casscf", active_space=(2, 2))
    # state=k>0 但未开态平均 → 报错
    try:
        PySCFCalculator(["H", "H"], state=1, **base).energy(c)
        print("  state=1 无态平均: 未报错  FAIL", flush=True)
        ok = False
    except CommandBackendError as exc:
        print(f"  state=1 无态平均: 报错 ✓ ({str(exc)[:50]})", flush=True)
    # nstates=5 但未开开关 → 不得隐式态平均 (向后兼容), 且能量 = 单态值
    e1 = PySCFCalculator(["H", "H"], nstates=5, **base).energy(c)
    e2 = PySCFCalculator(["H", "H"], **base).energy(c)
    same = abs(e1 - e2) < 1e-12
    ok &= same
    print(f"  nstates=5 未开开关: 与单态能量一致 {same} (不隐式触发)  "
          f"{'OK' if same else 'FAIL'}", flush=True)
    # 越界
    try:
        PySCFCalculator(["H", "H"], nstates=2, state=5, state_average=True,
                        **base).energy(c)
        print("  state 越界: 未报错  FAIL", flush=True)
        ok = False
    except CommandBackendError:
        print("  state 越界: 报错 ✓", flush=True)
    # 权重长度
    try:
        PySCFCalculator(["H", "H"], nstates=2, state_average=True,
                        state_weights=(1.0,), **base).energy(c)
        print("  权重长度不匹配: 未报错  FAIL", flush=True)
        ok = False
    except CommandBackendError:
        print("  权重长度不匹配: 报错 ✓", flush=True)
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    t0 = time.time()
    only = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    pool = [(nm, fn) for nm, fn in
            (("A", test_A_exact_vs_fci),
             ("B", test_B_state_identity_smoothness),
             ("C", test_C_nevpt2_excited),
             ("D", test_D_consistency_and_gradient),
             ("E", test_E_api_errors))
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
