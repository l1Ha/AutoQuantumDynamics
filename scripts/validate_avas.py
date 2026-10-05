#!/usr/bin/env python3
"""AVAS 自动活性空间验证 — 服务器运行。

判据:
  A. **基本正确性**: AVAS 的 ncas 等于目标 AO 标签数 (O 2p → 3; O 2p; H 1s → 5);
     能量低于 RHF (引入静态相关); 活动空间被 CASCI/CASSCF 正确采用。
  B. **AVAS + CASSCF**: 轨道优化后能量进一步下降且收敛。
  C. **组合能力** (v0.32–0.34 的成果可叠加): AVAS + 态平均 CASSCF(激发态)、
     AVAS + 选择组态 CI、AVAS + NEVPT2 均可运行。
  D. **阈值敏感性**: threshold 0.1/0.2/0.4 的 ncas 与能量变化**如实报告**。
  E. **开壳层**: 三重态 O₂ 的 AVAS 可运行 (openshell_option)。
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
T0 = time.time()


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


H2O = np.array([[0.0, 0.0, 0.0],
                [0.0, 1.4306, 1.1074],
                [0.0, -1.4306, 1.1074]])       # Bohr
N2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.1]])
O2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.28]])


def sec_energy(symbols, coords, **kw):
    return PySCFCalculator(list(symbols), **kw).energy(np.asarray(coords, float))


def test_A_basic():
    sec("A. 基本正确性: ncas 与目标一致 + 闭壳层恒等式 + 开壳层真相关")
    ok = True
    # A1: 闭壳层恒等式 —— H₂O 的 'O 2p' → CAS(3,6): CI 维数 = C(3,3)² = 1 (单行列式)
    #     → AVAS-CASCI 必须**精确等于** RHF (数学恒等式, 不是缺陷)
    c = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="casci",
                        avas="O 2p")
    e = c.energy(H2O)
    ncas = int(c.provenance["avas"].split("ncas=")[1].split(",")[0])
    e_rhf = sec_energy(["O", "H", "H"], H2O, basis="cc-pvdz", method="rhf")
    good1 = (ncas == 3) and abs(e - e_rhf) < 1e-9
    ok &= good1
    print(f"  H₂O 'O 2p' → CAS(3,6) (单行列式): E {e:.8f} == RHF {e_rhf:.8f} "
          f"(|Δ| {abs(e - e_rhf):.2e})  {'OK' if good1 else 'FAIL'}", flush=True)
    # A2: 开壳层 N₂ 'N 2p' → CAS(6,6): 有真实 (静态) 相关 → 能量必须低于 RHF
    c2 = PySCFCalculator(["N", "N"], basis="cc-pvdz", method="casci",
                         avas="N 2p")
    e2 = c2.energy(N2)
    ncas2 = int(c2.provenance["avas"].split("ncas=")[1].split(",")[0])
    e2_rhf = sec_energy(["N", "N"], N2, basis="cc-pvdz", method="rhf")
    good2 = (ncas2 >= 6) and (e2 < e2_rhf - 1e-3)
    ok &= good2
    print(f"  N₂ 'N 2p' → ncas={ncas2} (≥6, 阈值可能多纳入): E {e2:.8f} vs RHF {e2_rhf:.8f} "
          f"(相关 {e2_rhf - e2:.6f} Ha)  {'OK' if good2 else 'FAIL'}", flush=True)
    return bool(ok)


def test_B_with_casscf():
    sec("B. AVAS + CASSCF: 轨道优化使能量进一步下降")
    c_ci = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="casci",
                           avas=["O 2p", "H 1s"])     # PySCF 文档的列表写法
    e_ci = c_ci.energy(H2O)
    c_mcscf = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="casscf",
                              avas="O 2p, H 1s")      # 逗号写法
    e_mc = c_mcscf.energy(H2O)
    ok = e_mc < e_ci
    print(f"  AVAS-CASCI {e_ci:.8f} → AVAS-CASSCF {e_mc:.8f} "
          f"(下降 {e_ci - e_mc:.6f} Ha)  {'OK' if ok else 'FAIL'}", flush=True)
    return bool(ok)


def test_C_combinations():
    sec("C. 组合能力: AVAS + 态平均 / SCI / NEVPT2")
    ok = True
    # C1: AVAS + 态平均 CASSCF (激发态)
    try:
        c = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="casscf",
                            avas="O 2p; H 1s", state_average=True,
                            nstates=2, state=1)
        e1 = c.energy(H2O)
        es = c.last_state_energies
        good1 = (es is not None) and (len(es) == 2) and (es[1] > es[0])
        ok &= good1
        print(f"  AVAS + SA(2)-CASSCF: E0/E1 = {es[0]:.6f}/{e1:.6f} Ha  "
              f"{'OK' if good1 else 'FAIL'}", flush=True)
    except Exception as exc:
        print(f"  AVAS + SA-CASSCF 失败: {type(exc).__name__}: {str(exc)[:70]}",
              flush=True)
        ok = False
    # C2: AVAS + 选择组态 CI (较大标签集)
    try:
        c = PySCFCalculator(["N", "N"], basis="cc-pvdz", method="casci",
                            avas="N 2p", fci_solver="sci",
                            sci_select_cutoff=1e-5, sci_ci_coeff_cutoff=1e-7)
        e = c.energy(N2)
        prov = c.provenance["avas"]
        ncas = int(prov.split("ncas=")[1].split(",")[0])
        nele = int(prov.split("nelecas=")[1].rstrip(")"))
        print(f"  (AVAS 给出 ncas={ncas}, nelecas={nele})", flush=True)
        e_dense = sec_energy(["N", "N"], N2, basis="cc-pvdz", method="casci",
                             avas="N 2p", fci_solver="dense")
        d = abs(e - e_dense)
        good2 = d < 1e-4          # SCI(松阈值) 应接近稠密 (同空间)
        ok &= good2
        print(f"  AVAS + SCI (ncas={ncas}): SCI {e:.8f} vs 稠密 {e_dense:.8f} "
              f"(|Δ| {d:.2e})  {'OK' if good2 else 'FAIL'}", flush=True)
    except Exception as exc:
        print(f"  AVAS + SCI 失败: {type(exc).__name__}: {str(exc)[:70]}",
              flush=True)
        ok = False
    # C3: AVAS + NEVPT2
    try:
        c = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="casscf",
                            avas="O 2p; H 1s", pt2="nevpt2")
        e = c.energy(H2O)
        e_cas = sec_energy(["O", "H", "H"], H2O, basis="cc-pvdz",
                           method="casscf", avas="O 2p; H 1s")
        good3 = e < e_cas
        ok &= good3
        print(f"  AVAS + NEVPT2: {e:.8f} vs AVAS-CASSCF {e_cas:.8f} "
              f"(下降 {e_cas - e:.6f})  {'OK' if good3 else 'FAIL'}", flush=True)
    except Exception as exc:
        print(f"  AVAS + NEVPT2 失败: {type(exc).__name__}: {str(exc)[:70]}",
              flush=True)
        ok = False
    return bool(ok)


def test_D_threshold_sensitivity():
    sec("D. 阈值敏感性 (如实报告): threshold 0.1/0.2/0.4")
    for thr in (0.1, 0.2, 0.4):
        c = PySCFCalculator(["O", "H", "H"], basis="cc-pvdz", method="casci",
                            avas="O 2p; H 1s", avas_threshold=thr)
        e = c.energy(H2O)
        prov = c.provenance["avas"]
        ncas = int(prov.split("ncas=")[1].split(",")[0])
        nele = int(prov.split("nelecas=")[1].rstrip(")"))
        print(f"  threshold={thr:g}: ncas={ncas}, nelecas={nele}, "
              f"E = {e:.8f} Ha", flush=True)
    print("  (AVAS 目标轨道数固定 → ncas 一般不随阈值变; 阈值只影响候选"
          "筛选 → 同 ncas 下能量应一致或差 ~1e-6)", flush=True)
    return True


def test_E_openshell():
    sec("E. 开壳层: 三重态 O₂ 的 AVAS")
    e = sec_energy(["O", "O"], O2, basis="cc-pvdz", method="casci", spin=2,
                   avas="O 2p")
    e_rohf = sec_energy(["O", "O"], O2, basis="cc-pvdz", method="rohf", spin=2)
    c = PySCFCalculator(["O", "O"], basis="cc-pvdz", method="casci", spin=2,
                        avas="O 2p")
    c.energy(O2)
    prov = c.provenance["avas"]
    ncas = int(prov.split("ncas=")[1].split(",")[0])
    ok = (ncas >= 6) and (e < e_rohf - 1e-3)
    print(f"  三重态 O₂: ncas={ncas} (O 2p ×2 = 6; 开壳层可能多纳入), "
          f"AVAS-CASCI {e:.8f} vs ROHF {e_rohf:.8f} (相关 "
          f"{e_rohf - e:.6f} Ha)  {'OK' if ok else 'FAIL'}", flush=True)
    return bool(ok)


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    only = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    pool = [(nm, fn) for nm, fn in
            (("A", test_A_basic), ("B", test_B_with_casscf),
             ("C", test_C_combinations), ("D", test_D_threshold_sensitivity),
             ("E", test_E_openshell))
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
