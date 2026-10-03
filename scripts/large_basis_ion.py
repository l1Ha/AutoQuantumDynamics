#!/usr/bin/env python3
"""自建增广大基组 (cc-pV5Z + 等温弥散扩展) 计算 HeLi⁺ 曲线 — 最终基组收敛迭代。

PySCF 缺 Li 的 aug-cc-pV5Z, 本脚本按等温 (even-tempered) 规则在 cc-pV5Z 的
每个角动量上追加 n 个弥散壳层 (指数按 ratio 递减), 构造 aug-cc-pV5Z 等价基组。
He 直接用官方 aug-cc-pV5Z (可用)。

用法:
  python scripts/large_basis_ion.py --test          # 阱底单点测试
  python scripts/large_basis_ion.py --grid          # 全曲线
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from pyscf import gto, scf, cc

HA_EV = 27.211386245988
RE_OUT = "results/audit_ion_aV5Zlike.npz"


def aug_like(el: str, base: str = "cc-pV5Z", n_diffuse: int = 2,
             ratio: float = 3.0, use_official: bool = False):
    """cc-pV5Z + 等温弥散扩展 → aug-cc-pV5Z 等价基组 (按角动量追加壳层)。"""
    if use_official:
        try:
            return gto.basis.load("aug-cc-pV5Z", el)
        except Exception:
            pass
    b = [list(sh) for sh in gto.basis.load(base, el)]
    # 壳层格式: [l, [exp, c1, c2...], [exp, c1, ...], ...]
    min_exp = {}
    for sh in b:
        l = sh[0]
        e = min(row[0] for row in sh[1:])
        min_exp[l] = min(min_exp.get(l, 1e18), e)
    out = list(b)
    for l, emin in sorted(min_exp.items()):
        for k in range(1, n_diffuse + 1):
            out.append([l, [emin / ratio ** k, 1.0]])
    return out


def make_basis():
    return {"He": aug_like("He", use_official=True),
            "Li": aug_like("Li", use_official=False, n_diffuse=3, ratio=2.8)}


def e_ccsdt(basis, atom, charge=0):
    mol = gto.M(atom=atom, basis=basis, charge=charge, unit="Bohr", verbose=0)
    mf = scf.RHF(mol); e0 = mf.kernel()
    mycc = cc.CCSD(mf); ec = mycc.kernel()[0]
    return e0 + ec + mycc.ccsd_t(), mol.nao


def run(grid: bool):
    B = make_basis()
    _, nao = e_ccsdt(B, "He 0 0 0")
    print(f"自建基组 nao = {nao} (He: aug-cc-pV5Z, Li: cc-pV5Z+3 弥散/ℓ)")
    e_he = e_ccsdt(B, "He 0 0 0")[0]
    e_li = e_ccsdt(B, "Li 0 0 0", charge=1)[0]
    asym = e_he + e_li
    rg = (np.array([3.2, 3.69, 4.18, 4.67, 5.16, 5.64, 6.13, 6.62, 7.11, 7.60,
                    8.09, 8.58, 9.07, 9.56, 10.04, 11.02, 12.0])
          if grid else np.array([3.2, 3.69, 4.67, 6.13, 8.09, 12.0]))
    vu = np.zeros(len(rg)); vc = np.zeros(len(rg))
    t0 = time.time()
    for i, R in enumerate(rg):
        e_sup = e_ccsdt(B, f"He 0 0 0; Li 0 0 {R}", charge=1)[0]
        vu[i] = (e_sup - asym) * HA_EV * 1e3
        e_he_g = e_ccsdt(B, f"He 0 0 0; ghost:Li 0 0 {R}")[0]
        e_li_g = e_ccsdt(B, f"ghost:He 0 0 0; Li 0 0 {R}", charge=1)[0]
        vc[i] = (e_sup - e_he_g - e_li_g) * HA_EV * 1e3
        print(f"  R={R:5.2f}: 未校正 {vu[i]:+9.3f} | CP {vc[i]:+9.3f} | "
              f"BSSE {vu[i]-vc[i]:7.3f} meV  [{time.time()-t0:.0f}s]", flush=True)
    # 若测试模式, 补 3.69 附近的细点
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(RE_OUT, r_grid=rg, v_unc=vu, v_cp=vc,
                        basis="He:aug-cc-pV5Z + Li:cc-pV5Z+3diff",
                        method="ccsd(t)")
    print(f"✓ saved → {RE_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    run(a.grid)
