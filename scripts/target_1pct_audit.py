#!/usr/bin/env python3
"""目标: 迭代计算直到各项误差 < 1%。

审计三类可量化指标 (以 NIST / 生产参考数据为基准):
  A. 原子参考量: He*(2³S) 激发能, Li 电离能  (基准: NIST 19.8196 / 5.3917 eV)
  B. He+Li⁺ 离子曲线 V⁺(R)                  (基准: 参考 HeLip.txt 样条)
  C. 自电离宽度 Γ(R)                        (基准: 参考 MRCI 数据; 需 CAP-CI)

方法/基组阶梯 (迭代维度):
  atoms: RHF/ROHF → FCI;            basis: aug-cc-pVTZ → QZ → 5Z
  ion  : RHF → CCSD → CCSD(T);      basis: aug-cc-pVTZ → QZ → 5Z  (+ CP 校正)

用法:
  python scripts/target_1pct_audit.py --stage atoms --basis aug-cc-pV5Z
  python scripts/target_1pct_audit.py --stage ion --method ccsd(t) --basis aug-cc-pVQZ
结果保存 npz, 本地用 --report 与参考数据比较并打印误差表。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

HA_EV = 27.211386245988
NIST = {"He_exc_eV": 19.819614, "Li_IP_eV": 5.391715}


# --------------------------------------------------------------------------
# A. 原子参考量: He(1s²), He*(1s2s ³S), Li(2s), Li⁺ — FCI/RHF/CCSD(T)
# --------------------------------------------------------------------------

def compute_atoms(basis: str, method: str = "fci", out: str = "results/audit_atoms.npz"):
    from pyscf import gto, scf, fci, cc

    def run(atom, spin, charge=0):
        mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge,
                    unit="Bohr", verbose=0)
        mf = scf.ROHF(mol) if spin else scf.RHF(mol)
        e_scf = mf.kernel()
        if method == "scf":
            return float(e_scf)
        if method == "ccsd(t)":
            mycc = cc.CCSD(mf)
            # 开壳层用 UHF 参考的 CCSD(T) (ROHF 参考走 ROCCSD)
            if spin:
                mol_u = gto.M(atom=atom, basis=basis, spin=spin, charge=charge,
                              unit="Bohr", verbose=0)
                mfu = scf.UHF(mol_u); mfu.kernel()
                mycc = cc.CCSD(mfu)
            e_corr = mycc.kernel()[0]
            et = mycc.ccsd_t()
            return float(e_scf + e_corr + et)
        # FCI (小体系精确解, 仅电子数 ≤ 4)。靶定自旋用 Sz 扇区最低根即可
        # (He: (2,0) 扇区最低即 ³S; Li: (2,1) 最低即 ²S; Li⁺: (1,1) 最低即 ¹S),
        # 避免 fix_spin_ 在 ≥64 轨道时的 NotImplementedError。
        h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        h1e = mf.mo_coeff.T @ h_core @ mf.mo_coeff
        from pyscf import ao2mo
        eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
        cs = fci.direct_spin1.FCI()
        na = (mol.nelectron + spin) // 2
        nb = mol.nelectron - na
        e = cs.kernel(h1e, eri, mol.nao, (na, nb))[0]
        return float(np.asarray(e).ravel()[0]) + mol.energy_nuc()

    print(f"[atoms] basis={basis} method={method}")
    e_he_s = run("He 0 0 0", 0)
    e_he_t = run("He 0 0 0", 2)
    e_li = run("Li 0 0 0", 1)
    e_lip = run("Li 0 0 0", 0, charge=1)
    he_exc = (e_he_t - e_he_s) * HA_EV
    li_ip = (e_lip - e_li) * HA_EV
    print(f"  He* 激发能 = {he_exc:.4f} eV  (NIST {NIST['He_exc_eV']}, "
          f"误差 {(he_exc-NIST['He_exc_eV'])/NIST['He_exc_eV']*100:+.3f}%)")
    print(f"  Li 电离能   = {li_ip:.4f} eV  (NIST {NIST['Li_IP_eV']}, "
          f"误差 {(li_ip-NIST['Li_IP_eV'])/NIST['Li_IP_eV']*100:+.3f}%)")
    ok = (abs(he_exc - NIST["He_exc_eV"]) / NIST["He_exc_eV"] < 0.01
          and abs(li_ip - NIST["Li_IP_eV"]) / NIST["Li_IP_eV"] < 0.01)
    print(f"  → {'✓ 达标 (<1%)' if ok else '✗ 未达标'}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, basis=basis, method=method,
                        He_1S=e_he_s, He_3S=e_he_t, Li=e_li, Li_plus=e_lip,
                        he_exc_eV=he_exc, li_ip_eV=li_ip,
                        he_err_pct=(he_exc - NIST["He_exc_eV"]) / NIST["He_exc_eV"] * 100,
                        li_err_pct=(li_ip - NIST["Li_IP_eV"]) / NIST["Li_IP_eV"] * 100)
    print(f"  saved → {out}")
    return ok


# --------------------------------------------------------------------------
# B. He + Li⁺ 离子曲线 V⁺(R) — RHF/CCSD(T) + counterpoise
# --------------------------------------------------------------------------

def compute_ion(basis: str, method: str = "ccsd(t)", r_min: float = 3.2,
                r_max: float = 12.0, n: int = 19, cp: bool = True,
                out: str = "results/audit_ion.npz"):
    from pyscf import gto, scf, cc

    def e_run(atom, charge=0, spin=0):
        mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, unit="Bohr",
                    verbose=0)
        mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
        e = mf.kernel()
        if method == "scf":
            return float(e)
        mycc = cc.CCSD(mf)
        e_corr = mycc.kernel()[0]
        return float(e + e_corr + mycc.ccsd_t())

    print(f"[ion] basis={basis} method={method} CP={cp}")
    e_he = e_run("He 0 0 0")
    e_lip = e_run("Li 0 0 0", charge=1)
    asym = e_he + e_lip

    rg = np.linspace(r_min, r_max, n)
    v_unc = np.zeros(n); v_cp = np.zeros(n)
    for i, R in enumerate(rg):
        e_sup = e_run(f"He 0 0 0; Li 0 0 {R}", charge=1)
        v_unc[i] = (e_sup - asym) * HA_EV * 1e3
        if cp:
            e_he_g = e_run(f"He 0 0 0; ghost:Li 0 0 {R}", charge=0)
            e_lip_g = e_run(f"ghost:He 0 0 0; Li 0 0 {R}", charge=1)
            v_cp[i] = (e_sup - e_he_g - e_lip_g) * HA_EV * 1e3
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  R={R:5.2f}: V_unc={v_unc[i]:+9.3f} meV"
                  + (f" | V_CP={v_cp[i]:+9.3f}" if cp else ""))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, r_grid=rg, v_unc=v_unc, v_cp=v_cp, basis=basis,
                        method=method)
    print(f"  saved → {out}")
    return out


# --------------------------------------------------------------------------
# C. 报告: 与参考数据比较 (本地, 需参考目录)
# --------------------------------------------------------------------------

def report(ref_dir: str, ion_npz: str | None = None, atoms_npz: str | None = None):
    print("=" * 76)
    print("误差 < 1% 审计报告")
    print("=" * 76)
    if atoms_npz and os.path.exists(atoms_npz):
        d = np.load(atoms_npz, allow_pickle=True)
        print(f"[A] 原子参考 ({d['basis']}, {d['method']}):")
        print(f"    He* 激发能误差 = {float(d['he_err_pct']):+.3f}%  "
              f"({'✓' if abs(float(d['he_err_pct'])) < 1 else '✗'})")
        print(f"    Li  电离能误差 = {float(d['li_err_pct']):+.3f}%  "
              f"({'✓' if abs(float(d['li_err_pct'])) < 1 else '✗'})")
    if ion_npz and os.path.exists(ion_npz) and ref_dir:
        sys.path.insert(0, ref_dir)
        from src.potentials import IonicPotential
        from src.constants import HARTREE_TO_EV
        d = np.load(ion_npz, allow_pickle=True)
        rg, v_unc, v_cp = d["r_grid"], d["v_unc"], d["v_cp"]
        ref = IonicPotential()
        vref = np.array([ref(float(x)) for x in rg]) * 1e3 * HARTREE_TO_EV
        depth_ref = -vref.min()
        print(f"\n[B] He+Li⁺ 离子曲线 ({d['basis']}, {d['method']}):")
        print(f"    参考阱深 = {depth_ref:.3f} meV")
        for tag, v in (("未校正", v_unc), ("CP 校正", v_cp)):
            dm = v - vref
            mae = np.abs(dm).mean(); rmse = np.sqrt((dm ** 2).mean())
            print(f"    {tag}: 阱深 {(-v.min()):7.3f} meV (误差 {(-v.min()-depth_ref)/depth_ref*100:+6.2f}%) | "
                  f"MAE {mae:6.3f} meV ({mae/depth_ref*100:5.2f}% of 阱深) | "
                  f"RMSE {rmse:6.3f} meV | max|Δ| {np.abs(dm).max():6.3f} meV")
            print(f"       → MAE{'✓' if mae/depth_ref < 0.01 else '✗'} "
                  f"(<1% 判据: MAE < {depth_ref*0.01:.3f} meV)")
        # 逐点表
        print(f"    {'R(bohr)':>8} {'参考':>10} {'未校正':>10} {'CP':>10} {'Δ_CP':>9}")
        for i in range(0, len(rg), 2):
            print(f"    {rg[i]:8.2f} {vref[i]:10.3f} {v_unc[i]:10.3f} "
                  f"{v_cp[i]:10.3f} {v_cp[i]-vref[i]:+9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["atoms", "ion", "report"], required=True)
    ap.add_argument("--basis", default="aug-cc-pVTZ")
    ap.add_argument("--method", default=None)
    ap.add_argument("--no-cp", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    if a.stage == "atoms":
        compute_atoms(a.basis, a.method or "fci", a.out or "results/audit_atoms.npz")
    elif a.stage == "ion":
        compute_ion(a.basis, a.method or "ccsd(t)", cp=not a.no_cp,
                    out=a.out or "results/audit_ion.npz")
    else:
        report(a.reference or "", a.out or "results/audit_ion.npz",
               "results/audit_atoms.npz")


if __name__ == "__main__":
    main()
