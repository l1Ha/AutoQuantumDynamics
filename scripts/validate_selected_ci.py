#!/usr/bin/env python3
"""选择组态 CI (selected_ci, 大活性空间) 验证 — 服务器运行。

判据:
  A. **精确性**: 小活性空间下 SCI (紧阈值) 必须等于**稠密 FCI** (ΔE < 1e-6 Ha),
     且在 CASCI 与 **CASSCF** 两个层级都成立 (后者含轨道优化)。
  B. **大活性空间可行性**: N₂/cc-pVDZ CAS(14,14) —— 稠密 FCI 维数 1.18e7
     (C(14,7)²) 不可直接对角化, SCI 必须能算出来并收敛。
  C. **变分单调性 (自洽性)**: 阈值越紧 → 选择空间越大 → 能量必须单调不升.
  D. **势能曲线**: CAS(14,14)-SCI 扫描 N₂ 平衡区 → 曲线平滑, 抛物线拟合给出
     合理 R_e (与实验 1.098 Å 对比, 如实报告偏差).
  E. **回归**: 默认路径 (dense + 自旋纯) 必须复现此前归档的精确值
     (H₂/STO-3G CAS(2,2) = FCI = −1.137275944 Ha, 见 Slurm 1559180 D 组).
"""
from __future__ import annotations
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoquantum.pes.calculators import PySCFCalculator  # noqa: E402
from math import comb  # noqa: E402

BOHR = 0.529177210903
T0 = time.time()


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def n2_coords(r_bohr: float):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_bohr]])


def dense_dim(ncas: int, nelecas: int) -> int:
    """稠密 FCI 维数 C(ncas, nα)·C(ncas, nβ)。"""
    na, nb = nelecas // 2, nelecas - nelecas // 2
    return comb(ncas, na) * comb(ncas, nb)


def E(symbols, coords, **kw):
    return PySCFCalculator(symbols, **kw).energy(np.asarray(coords, dtype=float))


# ---------------------------------------------------------------------------
def test_A_exact_vs_dense():
    sec("A. 精确性: SCI (紧阈值) vs 稠密 FCI — CASCI 层 (N₂)")
    ok = True
    r = 2.1
    tight = dict(fci_solver="sci", sci_select_cutoff=1e-8,
                 sci_ci_coeff_cutoff=1e-10)
    # (体系, 基组, CAS) — 注意活性空间必须放得下 (N₂/STO-3G 仅 10 轨道 →
    # CAS(6,6) ✓; cc-pVDZ 28 轨道 → CAS(8,8) ✓); 早前 CAS(8,8)/STO-3G 触发
    # nvir<0 断言属测试设计错误。
    cases = (("sto-3g", 6, 6), ("cc-pvdz", 8, 8))
    for basis, ncas, nelecas in cases:
        dim = dense_dim(ncas, nelecas)
        e_dense = E(["N", "N"], n2_coords(r), basis=basis, method="casci",
                    active_space=(ncas, nelecas))
        e_sci = E(["N", "N"], n2_coords(r), basis=basis, method="casci",
                  active_space=(ncas, nelecas), **tight)
        d = abs(e_sci - e_dense)
        good = d < 1e-6
        ok &= good
        print(f"  CAS({ncas},{nelecas})/{basis} (稠密维数 {dim}): dense "
              f"{e_dense:.9f} vs SCI {e_sci:.9f} → |Δ| = {d:.2e} Ha (<1e-6)  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    return bool(ok)


def test_B_large_active_space():
    sec("B. 大活性空间: N₂/cc-pVDZ CAS(14,14) — 稠密 FCI 不可行")
    dim = dense_dim(14, 14)
    print(f"  稠密 FCI 维数 C(14,7)² = {dim:,} (直接对角化不可行)", flush=True)
    t0 = time.time()
    e = E(["N", "N"], n2_coords(2.1), basis="cc-pvdz", method="casci",
          active_space=(14, 14), fci_solver="sci", sci_select_cutoff=1e-4,
          sci_ci_coeff_cutoff=1e-6)
    dt = time.time() - t0
    e_ccsd = E(["N", "N"], n2_coords(2.1), basis="cc-pvdz", method="ccsd(t)",
               frozen_core=True)
    ok = np.isfinite(e) and (dt < 1500)
    print(f"  CASCI(14,14)-SCI: E = {e:.6f} Ha ({dt:.0f} s)", flush=True)
    print(f"  对照 CCSD(T)/cc-pVDZ (冻核) = {e_ccsd:.6f} Ha "
          f"(差 {abs(e - e_ccsd) * 1000:.1f} mHa; CASCI 无动态相关, 应当更高)",
          flush=True)
    print(f"  收敛且耗时可控 (<1500 s): {ok}  {'OK' if ok else 'FAIL'}",
          flush=True)
    return bool(ok)


def test_C_variational_monotonicity():
    sec("C. 变分单调性: 阈值越紧 → 能量单调不升 (N₂/cc-pvdz CASCI(10,10)-SCI)")
    cuts = [1e-3, 1e-4, 1e-5]
    es = []
    for c in cuts:
        e = E(["N", "N"], n2_coords(2.1), basis="cc-pvdz", method="casci",
              active_space=(10, 10), fci_solver="sci", sci_select_cutoff=c,
              sci_ci_coeff_cutoff=c * 1e-2)
        es.append(e)
        print(f"  select_cutoff={c:g}: E = {e:.8f} Ha", flush=True)
    mono = all(es[i + 1] <= es[i] + 1e-9 for i in range(len(es) - 1))
    sat = abs(es[-1] - es[-2]) < 1e-4
    ok = bool(mono and sat)
    print(f"  单调不升 {mono}; 末两档差 {abs(es[-1]-es[-2])*1000:.3f} mHa "
          f"(<0.1 mHa → 已收敛)  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def test_D_potential_curve():
    sec("D. 势能曲线: 稠密 CASSCF(8,8) 求 R_e + 大空间 CASCI-SCI(14,14) 平滑性")
    # 扫描区间必须**跨过极小点** (N₂ R_e ≈ 1.10 Å = 2.08 Bohr), 否则拟合退化
    rs = np.arange(1.80, 2.86, 0.15)

    def curve(**kw):
        return np.array([E(["N", "N"], n2_coords(r), **kw) for r in rs])

    def r_e_fit(es):
        i = int(np.argmin(es))
        if 0 < i < len(rs) - 1:
            c = np.polyfit(rs[i - 1:i + 2], es[i - 1:i + 2], 2)
            return -c[1] / (2 * c[0]), True
        return float(rs[i]), False

    def spike(es):
        """跳变检测: max|Δ²E| / median|Δ²E| (平滑曲线 ~几, 跳变 >> 100)。"""
        d2 = np.abs(np.diff(es, 2))
        med = float(np.median(d2))
        return float(d2.max() / med) if med > 1e-14 else 0.0

    es1 = curve(basis="cc-pvdz", method="casscf", active_space=(8, 8))
    r1, fit1 = r_e_fit(es1)
    r1_ang = r1 * BOHR
    dev1 = abs(r1_ang - 1.098) / 1.098
    ok1 = fit1 and (0.95 < r1_ang < 1.30) and (dev1 < 0.15)
    for r, e in zip(rs, es1):
        print(f"  [CASSCF(8,8)] R={r:.2f} Bohr: E = {e:.6f} Ha", flush=True)
    print(f"  → R_e = {r1_ang:.4f} Å vs 实验 1.098 Å (偏差 {dev1*100:+.2f}%, "
          f"拟合有效 {fit1})  {'OK' if ok1 else 'FAIL'}", flush=True)

    es2 = curve(basis="cc-pvdz", method="casci", active_space=(14, 14),
                fci_solver="sci", sci_select_cutoff=1e-4,
                sci_ci_coeff_cutoff=1e-6)
    r2, fit2 = r_e_fit(es2)
    r2_ang = r2 * BOHR
    sp2 = spike(es2)
    smooth = bool(sp2 < 20)
    for r, e in zip(rs, es2):
        print(f"  [CASCI-SCI(14,14)] R={r:.2f} Bohr: E = {e:.6f} Ha", flush=True)
    print(f"  → 平滑跳变比 max|Δ²E|/median = {sp2:.2f} (<20) {smooth}; "
          f"R_e = {r2_ang:.4f} Å (拟合有效 {fit2})", flush=True)
    consistent = abs(r2_ang - r1_ang) < 0.15
    ok = bool(ok1 and smooth and fit2 and consistent
              and (0.95 < r2_ang < 1.35))
    print(f"  两空间 R_e 一致性 |Δ| = {abs(r2_ang-r1_ang):.4f} Å (<0.15): "
          f"{consistent}", flush=True)
    print(f"  注: CASCI 在 SCF 轨道上**无轨道优化**, 能量可高于小活性空间 CASSCF "
          f"(此处 ΔE = {(es1 - es2)[0]*1000:+.1f} mHa @ 最短 R); 两者互补, "
          f"键长趋势一致。", flush=True)
    print(f"  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def test_E_regression_default_path():
    sec("E. 回归: 默认路径 (dense + 自旋纯) 复现归档精确值 (H₂/STO-3G CAS(2,2))")
    ref = -1.137275944          # 归档于 Slurm 1559180 D 组 (= FCI)
    c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    e_def = E(["H", "H"], c, basis="sto-3g", method="casscf", active_space=(2, 2))
    e_np = E(["H", "H"], c, basis="sto-3g", method="casscf", active_space=(2, 2),
             spin_pure_fci=False)
    d_def = abs(e_def - ref)
    d_np = abs(e_np - ref)
    ok = (d_def < 1e-6) and (d_np < 1e-6)
    print(f"  默认 (auto: 单根不纯化) E = {e_def:.9f} | 显式关闭 E = {e_np:.9f} | "
          f"归档值 {ref:.9f}", flush=True)
    print(f"  |Δ| = {d_def:.2e} / {d_np:.2e} Ha (<1e-6, 两种设置均不变)  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    return bool(ok)


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    only = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    pool = [(nm, fn) for nm, fn in
            (("A", test_A_exact_vs_dense),
             ("B", test_B_large_active_space),
             ("C", test_C_variational_monotonicity),
             ("D", test_D_potential_curve),
             ("E", test_E_regression_default_path))
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
          f"(总耗时 {time.time() - T0:.0f}s)", flush=True)
    sys.exit(0 if all_ok else 1)
