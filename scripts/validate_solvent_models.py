#!/usr/bin/env python3
"""隐式溶剂模型验证: ddCOSMO / **PCM** / **ddPCM** / **SMD** — 服务器运行。

判据:
  A. ε→1 极限 → 溶剂化能 → 0 (物理极限; ddPCM 因 PySCF 内部除零用 1+δ 逼近)
  B. 介电单调性 (ε = 2 → 78.4, |ΔE_solv| 单调增)
  C. 跨模型/变体一致性 (ddcosmo / pcm 四变体 / ddpcm 同号且量级一致)
  D. 极性趋势 (H₂O ≫ CH₄, He ≈ 0)
  E. Li⁺ Born 标度 (PCM 隐含 Solvent 半径落在物理区间)
  F. 四类入口 (SCF / post-SCF(MP2) / TD-DFT / CASSCF) 对 pcm/ddpcm 全部可用
  G. 梯度: PCM 解析 vs FD; ddPCM 走 FD 且含溶剂响应; SMD 解析 vs FD 实测
  H. SMD: 接口可用 + 非静电项 (CDS) 已计入 + 与实验的数量级比较 (如实记录)
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

AU2A = 0.529177210903
H2K = 627.5094740631          # Hartree → kcal/mol
EV = 27.211386245988

MODELS = ("ddcosmo", "pcm", "ddpcm")


def bohr(xyz_ang):
    return np.asarray(xyz_ang, dtype=float) / AU2A


H2O = ["O", "H", "H"]
X_H2O = bohr([[0.0, 0.0, 0.0], [0.0, 0.7570, 0.5860], [0.0, -0.7570, 0.5860]])
CH4 = ["C", "H", "H", "H", "H"]
X_CH4 = bohr([[0.0, 0.0, 0.0],
              [0.6276, 0.6276, 0.6276], [0.6276, -0.6276, -0.6276],
              [-0.6276, 0.6276, -0.6276], [-0.6276, -0.6276, 0.6276]])
HE = ["He"]
X_HE = np.zeros((1, 3))
LI = ["Li"]
X_LI = np.zeros((1, 3))


def sec(t):
    print(flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def E(symbols, coords, **kw):
    """走 AutoQuantum 计算器接口的单点能 (Hartree)。"""
    calc = PySCFCalculator(symbols, **kw)
    return calc.energy(np.asarray(coords, dtype=float))


def dE_solv(symbols, coords, model, eps=None, solvent=None, **kw):
    """ΔE_solv = E(溶剂) − E(气相), kcal/mol (pyscf 后端, 同一方法/基组)。"""
    kw = dict(kw)
    if solvent is not None:
        kw.update(solvent=solvent, solvent_model=model)
        if eps is not None:
            kw["solvent_eps"] = eps
    else:
        kw.update(solvent_eps=eps, solvent_model=model)
    e_g = E(symbols, coords, **{k: v for k, v in kw.items()
                                if k not in ("solvent", "solvent_eps",
                                             "solvent_model")})
    e_s = E(symbols, coords, **kw)
    return (e_s - e_g) * H2K, e_g, e_s


def fd_grad(calc, coords, h=1e-4):
    g = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for j in range(3):
            cp, cm = coords.copy(), coords.copy()
            cp[i, j] += h
            cm[i, j] -= h
            g[i, j] = (calc.energy(cp) - calc.energy(cm)) / (2 * h)
    return g


# ---------------------------------------------------------------------------
def test_A_eps_limit():
    sec("A. ε→1 极限: 溶剂化能必须 → 0 (H₂O/6-31G*/RHF)")
    ok = True
    for model in MODELS:
        eps = 1.0 if model != "ddpcm" else 1.000001
        try:
            de, _, _ = dE_solv(H2O, X_H2O, model, eps=eps, basis="6-31g*")
            thr = 1e-3 if model != "ddpcm" else 1e-2
            good = abs(de) < thr
            ok &= good
            note = " (PySCF ddPCM 在 ε=1 除零 → 用 1+1e-6 逼近)" \
                if model == "ddpcm" else ""
            print(f"  {model:8s} ε={eps:<9g} ΔE_solv = {de:+.6f} kcal/mol  "
                  f"{'OK' if good else 'FAIL'}{note}", flush=True)
        except Exception as exc:
            print(f"  {model:8s} EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            ok = False
    return ok


def test_B_dielectric_monotonicity():
    sec("B. 介电单调性: |ΔE_solv| 随 ε 单调增 (H₂O/6-31G*/RHF)")
    ok = True
    for model in ("ddcosmo", "pcm"):
        vals = []
        for eps in (2.0, 5.0, 20.0, 78.4):
            de, _, _ = dE_solv(H2O, X_H2O, model, eps=eps, basis="6-31g*")
            vals.append(de)
        mono = all(abs(vals[i]) < abs(vals[i + 1]) + 1e-9
                   for i in range(len(vals) - 1))
        neg = all(v < 0 for v in vals)
        ok &= bool(mono and neg)
        print(f"  {model:8s} " + " → ".join(f"{v:.3f}" for v in vals)
              + f" kcal/mol  单调 {mono}, 全负 {neg}  "
              f"{'OK' if mono and neg else 'FAIL'}", flush=True)
    return ok


def test_C_cross_model():
    sec("C. 跨模型/变体一致性 (H₂O/6-31G*/RHF, ε=78.4)")
    res = {}
    for model in MODELS:
        res[model] = dE_solv(H2O, X_H2O, model, eps=78.4, basis="6-31g*")[0]
    variants = {}
    for var in ("C-PCM", "IEF-PCM", "COSMO", "SS(V)PE"):
        variants[var] = dE_solv(H2O, X_H2O, "pcm", eps=78.4, basis="6-31g*",
                                pcm_variant=var)[0]
    vals = list(res.values())
    same_sign = all(v < 0 for v in vals)
    ratio = max(abs(v) for v in vals) / min(abs(v) for v in vals)
    ok = bool(same_sign and ratio < 2.0)
    print("  模型: " + " | ".join(f"{k} {v:+.3f}" for k, v in res.items())
          + " kcal/mol", flush=True)
    print("  PCM 变体: " + " | ".join(f"{k} {v:+.3f}"
                                     for k, v in variants.items()), flush=True)
    spread = (max(variants.values()) - min(variants.values())) / \
        abs(np.mean(list(variants.values())))
    ok &= bool(spread < 0.05)
    print(f"  符号一致 {same_sign}, 模型间最大/最小比 {ratio:.2f} (<2.0), "
          f"变体相对展宽 {spread * 100:.2f}% (<5%)  {'OK' if ok else 'FAIL'}",
          flush=True)
    return ok


def test_D_polarity_trend():
    sec("D. 极性趋势: H₂O ≫ CH₄, He ≈ 0 (6-31G*/RHF, ε=78.4)")
    ok = True
    for model in MODELS:
        de_w = dE_solv(H2O, X_H2O, model, eps=78.4, basis="6-31g*")[0]
        de_m = dE_solv(CH4, X_CH4, model, eps=78.4, basis="6-31g*")[0]
        de_he = dE_solv(HE, X_HE, model, eps=78.4, basis="6-31g*")[0]
        good = (de_w < -3.0) and (abs(de_m) < 1.0) and (abs(de_he) < 0.05) \
            and (abs(de_w) > 5 * abs(de_m))
        ok &= good
        print(f"  {model:8s} H₂O {de_w:+.3f} | CH₄ {de_m:+.3f} | He {de_he:+.3f}"
              f" kcal/mol  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_E_born_scaling():
    sec("E. Li⁺ Born 标度 (6-31G*/RHF, ε=78.4): 隐含半径 = 163.9/|ΔE|")
    ok = True
    for model in ("ddcosmo", "pcm"):
        de, _, _ = dE_solv(LI, X_LI, model, eps=78.4, basis="6-31g*",
                           charge=1)
        r_impl = 163.9 / abs(de)
        good = 1.0 < r_impl < 3.0
        ok &= good
        print(f"  {model:8s} ΔE_solv = {de:+.3f} kcal/mol → 隐含半径 "
              f"{r_impl:.3f} Å  {'OK (物理区间 1-3 Å)' if good else 'FAIL'}",
              flush=True)
    print("  注: PCM 默认 vdw_scale=1.2 → 隐含半径 ≈ 1.2×r_vdW(Li⁺) 自洽;",
          flush=True)
    print("      ddCOSMO 空腔定义不同 (无 1.2 缩放), 隐含半径更小属预期。",
          flush=True)
    return ok


def test_F_four_entry_points():
    sec("F. 四类入口: SCF / post-SCF(MP2) / TD-DFT / CASSCF (H₂O/6-31G*)")
    ok = True
    for model in ("pcm", "ddpcm"):
        row = []
        # SCF
        de_scf = dE_solv(H2O, X_H2O, model, eps=78.4, basis="6-31g*")[0]
        row.append(("SCF", de_scf))
        # post-SCF: MP2 (能量; 相关能中的溶剂响应)
        de_mp2 = dE_solv(H2O, X_H2O, model, eps=78.4, basis="6-31g*",
                         method="mp2", frozen_core=True)[0]
        row.append(("MP2", de_mp2))
        # TD-DFT: 激发能溶剂位移
        e_g = PySCFCalculator(H2O, basis="6-31g*", method="tddft", xc="b3lyp",
                              nstates=4)
        ex_g = e_g.excitation_spectrum(X_H2O)[0]
        e_s = PySCFCalculator(H2O, basis="6-31g*", method="tddft", xc="b3lyp",
                              nstates=4, solvent_eps=78.4, solvent_model=model)
        ex_s = e_s.excitation_spectrum(X_H2O)[0]
        shift = float(np.asarray(ex_s)[0] - np.asarray(ex_g)[0])  # eV
        # CASSCF
        de_cas = dE_solv(H2O, X_H2O, model, eps=78.4, basis="6-31g*",
                         method="casscf", active_space=(4, 4))[0]
        good = (de_scf < 0 and de_mp2 < 0 and de_cas < 0
                and abs(shift) < 1.5
                and abs(de_mp2) < 3 * abs(de_scf)
                and abs(de_cas) < 3 * abs(de_scf))
        ok &= good
        print(f"  {model:8s} SCF {de_scf:+.3f} | MP2 {de_mp2:+.3f} | "
              f"CASSCF(4,4) {de_cas:+.3f} kcal/mol | TD-DFT 首激发位移 "
              f"{shift:+.3f} eV  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def test_G_gradients():
    sec("G. 溶剂化梯度: 解析 vs 有限差分 (H₂O/6-31G*, ε=78.4)")
    ok = True
    # PCM: 解析溶剂梯度 (PySCF pyscf/solvent/grad/pcm)
    c = PySCFCalculator(H2O, basis="6-31g*", solvent_eps=78.4,
                        solvent_model="pcm")
    g = c.gradient(X_H2O)
    gfd = fd_grad(c, X_H2O)
    d = float(np.abs(g - gfd).max())
    good = d < 1e-5
    ok &= good
    print(f"  pcm      解析-vs-FD max|Δ| = {d:.2e}  "
          f"{'OK' if good else 'FAIL'}", flush=True)
    # ddPCM: 无解析模块 → 计算器内部 FD; 且必须与气相梯度不同 (含溶剂响应)
    c2 = PySCFCalculator(H2O, basis="6-31g*", solvent_eps=78.4,
                         solvent_model="ddpcm")
    assert c2._solvent_fd_grad, "ddPCM 必须走 FD 路径"
    g2 = c2.gradient(X_H2O)
    g2fd = fd_grad(c2, X_H2O)
    d2 = float(np.abs(g2 - g2fd).max())
    cg = PySCFCalculator(H2O, basis="6-31g*")
    ggas = cg.gradient(X_H2O)
    dresp = float(np.abs(g2 - ggas).max())
    good2 = d2 < 1e-6 and dresp > 1e-4
    ok &= good2
    print(f"  ddpcm    FD 自洽 {d2:.2e} (<1e-6), 与气相梯度差 {dresp:.2e} "
          f"(>1e-4, 说明含溶剂响应)  {'OK' if good2 else 'FAIL'}", flush=True)
    # SMD: 解析 vs FD 实测 (决定是否需要 FD 回退)
    cs = PySCFCalculator(H2O, basis="6-31g*", solvent="water",
                         solvent_model="smd")
    gs = cs.gradient(X_H2O)
    gsfd = fd_grad(cs, X_H2O)
    ds = float(np.abs(gs - gsfd).max())
    print(f"  smd      解析-vs-FD max|Δ| = {ds:.2e}  "
          f"{'OK (解析梯度正确)' if ds < 1e-5 else '⚠ 解析梯度不可靠'}", flush=True)
    ok &= bool(ds < 1e-5)
    return ok


def test_H_smd():
    sec("H. SMD 全溶剂化模型: 接口 + 非静电项 + 数量级 (H₂O)")
    ok = True
    # 1) 与纯静电 PCM 的差 → CDS (cavitation/dispersion/structure) 已计入
    de_pcm = dE_solv(H2O, X_H2O, "pcm", eps=78.4, basis="6-31g*")[0]
    de_smd_r = dE_solv(H2O, X_H2O, "smd", solvent="water", basis="6-31g*")[0]
    cds = de_smd_r - de_pcm
    good1 = abs(cds) > 0.2       # 非静电项必须真实存在
    ok &= good1
    print(f"  RHF/6-31G*: SMD {de_smd_r:+.3f} vs PCM 静电 {de_pcm:+.3f} "
          f"kcal/mol → 非静电/半径差 {cds:+.3f}  "
          f"{'OK' if good1 else 'FAIL'}", flush=True)

    # 2) SMD 的原生参数化级别 (M05-2X/6-31G*): 与实验 ΔG*_hyd(H₂O) = −6.32
    #    kcal/mol 比较 (如实记录; PySCF SMD 标注实验性 + 标准态约定可能不同)
    try:
        de_smd_m = dE_solv(H2O, X_H2O, "smd", solvent="water", basis="6-31g*",
                           method="dft", xc="m05-2x")[0]
        dev = de_smd_m - (-6.32)
        print(f"  M05-2X/6-31G*: SMD {de_smd_m:+.3f} kcal/mol; "
              f"实验 −6.32 → 偏差 {dev:+.3f}", flush=True)
        print("  (PySCF SMD 为实验性实现; 本项为数量级核查, 不作定量精度主张)",
              flush=True)
        ok &= bool(abs(dev) < 6.0)
    except Exception as exc:
        print(f"  M05-2X 分支不可用 (环境限制, 跳过): "
              f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)

    # 3) 疏水 solute: 空腔项主导 → ΔG 应为正, 与实验 +1.9~2.0 比较
    de_ch4 = dE_solv(CH4, X_CH4, "smd", solvent="water", basis="6-31g*",
                     method="dft", xc="m05-2x")[0]
    dev_ch4 = de_ch4 - 1.95
    good3 = (de_ch4 > 0) and abs(dev_ch4) < 2.0
    ok &= good3
    print(f"  CH₄/water (M05-2X): SMD {de_ch4:+.3f} kcal/mol; 实验 ΔG*_hyd "
          f"≈ +1.9~2.0 (疏水空腔项为正) → 偏差 {dev_ch4:+.3f}  "
          f"{'OK' if good3 else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    print(f"节点: {os.uname().nodename}", flush=True)
    import pyscf
    print(f"pyscf: {pyscf.__version__}", flush=True)
    t0 = time.time()
    only = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    pool = [(nm, fn) for nm, fn in
            (("A", test_A_eps_limit), ("B", test_B_dielectric_monotonicity),
             ("C", test_C_cross_model), ("D", test_D_polarity_trend),
             ("E", test_E_born_scaling), ("F", test_F_four_entry_points),
             ("G", test_G_gradients), ("H", test_H_smd))
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
