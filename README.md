# AutoQuantum

**分子反应动力学量子计算研究原型** — 势能面构建 → 神经网络拟合 → 量子动力学 → 可视化的自动化流程。

> ⚠️ 这是研究/教学原型，不是经过认证的科学计算软件。定量使用前请阅读
> [功能边界与已知限制](#功能边界与已知限制) 与 [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md)。

> 📦 **相关独立项目**: 势能面计算与拟合子系统已拆分为独立仓库
> **[AutoQuantumPES](https://github.com/l1Ha/AutoQuantumPES)** (`aqpes` 包,
> 含 CLI 与独立测试)。本仓库的 ``autoquantum`` 包**保持原有全部功能不变**,
> 两者共享同一套 PES 内核代码。

## 亚稳态体系与高级电子结构计算 (v0.20.0)

针对如 $\text{He}^* + \text{Li}$ 等处于电离连续谱中的亚稳态碰撞体系，`PySCFCalculator` 提供了高自旋约束与轨道锁定机制：
- **高自旋约束与自旋锁定 (`spin_lock`)**: 支持严格 ROHF/ROKS 及 UHF/UKS，审计 $\langle S^2 \rangle$ 并拦截自旋污染，物理上消除四重态（$^4\Sigma^+$，$S=3/2$）的自电离变分塌陷；
- **最大重叠法 (`use_mom=True`)**: 沿几何采样路径维持特定电子激发/占据组态，避免根翻转；
- **复势能接口 (`resonance_width` / `complex_energy`)**: 支持表征自电离衰变宽度
  $\Gamma(R)$ 与复光学势 $W(R) = V(R) - \frac{i}{2}\Gamma(R)$。
  ⚠ **诚实边界**: 这是**解析模型接口**（指数/盒式），**不是**第一性原理共振计算——
  CAP-CI/Feshbach 投影尚未实现；真实 $\Gamma(R)$ 需外部提供（如文献 MRCI 数据）。

```bash
# 从 CLI 采样高自旋开壳层构型
autoquantum sample --backend pyscf --input ref.xyz -o he_li_highspin.npz \
    --spin 3 --method rohf --basis aug-cc-pVTZ --spin-lock --mom
```

**真实集群实测** (`scripts/calc_metastable_heli.py`，c211 → Slurm `liquid_high`)：
He\*(2³S)+Li 四重态/双重态势能面 (aug-cc-pVTZ, 30 点 ROHF+spin_lock+MOM)，
FCI 校验 He ³S–¹S = **19.88 eV**（实验 19.82 eV），渐近一致性 0.9 mHa，全程 30.6 s。

> ⚠ **2026-10 校核更正**：早期报告的"54 meV 范德华阱"经 counterpoise 校正证伪——
> 该"阱"为 BSSE 赝像，校正后 |V| < 1 meV；"18.8 eV 彭宁电子能量"标注有误，
> 正确值为 **14.4279 eV + [V₂Σ − V⁺]**。与生产参考势（MRCI 4 通道 + HeLi⁺ +
> MRCI 宽度）的完整校核见 `scripts/plot_heli_vs_reference.py`，
> 图见 [book 第 4 章](book/chapters/04-电子结构基础.md)。

**与生产参考数据 (Pro_HeLi_Enhanced) 的交叉校核**（`scripts/plot_heli_vs_reference.py`）：
HeLi⁺ 离子通道互证通过（本工作 RHF −68 meV @ 3.6 bohr vs 参考 −86.5 meV @ 3.65 bohr）；
本工作早期解析宽度模型 Γ=0.04e^(−1.1R) 与参考 MRCI 数据相差数个数量级，应改用后者。

```bash
bash scripts/remote.sh sync && bash scripts/remote.sh submit liquid_high scripts/sbatch_he_li.sbatch
python scripts/calc_metastable_heli.py --replot book/data/he_li_metastable_pes.npz  # 本地重绘
```

## 误差 < 1% 审计 (v0.20.3)

以 NIST 与生产参考数据为基准的迭代审计 (`scripts/target_1pct_audit.py`,
`error_budget_report.py`, `width_model_validation.py`):

| 指标 | 基准 | 结果 | 误差 | <1% |
|---|---|---|---|---|
| He\*(2³S) 激发能 | NIST 19.8196 eV | FCI/aVQZ 19.8729 | **+0.27%** | ✓ |
| Li 电离能 | NIST 5.3917 eV | FCI/aVQZ 5.3701 | **−0.40%** | ✓ |
| He+Li⁺ 阱区 R=5–9 bohr | 参考样条 | CCSD(T)+CP（三基组一致） | **0.81–0.85%** | ✓ |
| Γ(R) 模型（管线输入） | 参考 MRCI 数据 | log-样条+指数尾+ICD | **0.0002%** | ✓ |
| He+Li⁺ 阱深 | 参考(去 BSSE) 76–78 meV | 本工作 78.3 ± 0.6 meV | **0.8%**(1) | ⚠ |
| He+Li⁺ 全区间 MAE | 参考样条（原始） | CCSD(T)+CP/aVQZ | 2.09% | ✗ |
| Γ(R) 第一性原理 | 参考 MRCI | CAP-CI（机制已验证） | 共振不可分辨 | ✗ |

(1) 本工作 4 个 QZ 级基组的 CP 阱深收敛至 77.67 / 78.36 / 78.80 meV
（自身不确定度 0.7% < 1%）；参考的 BSSE 自由阱深因**其方法出处未记录**而只能定到
76–78 meV，故一致性为 0.8–3%。

**参考曲线自身误差的决定性归因**（`scripts/ref_basis_id.py`, `ref_method_id.py`）：
参考 `HeLip.txt` 被 **cc-pVQZ 级别、未做 counterpoise 校正**的 CCSD/CCSD(T)
精确再现（19 点 MAE 0.71 meV、R=3.2 Bohr 处 0.03 meV；混合基组/MP2/SCF/Cartesian
均更差）。该曲线在阱底含 **约 10 meV 未校正 BSSE**（占其 85.9 meV "阱深" 的 12%），
且在 R>5 Å 被拉平为渐近平台（偏离物理 −C₄/R⁴ 尾约 0.5 meV）。因此
"全区间 MAE 2.09%" 主要来自参考数据自身的系统误差。

**CAP-CI** (`scripts/cap_ci.py`) 已实现并在束缚态上验证（Γ 严格 ∝ η → 0）；
He\*+Li 共振（Γ≈10 meV，出射电子 14.4 eV）在现有基组/活性空间下不可分辨——
定量 Γ 需专用连续谱基组或 Feshbach 投影。

## AVAS 自动活性空间 (v0.35.0)

```python
calc = PySCFCalculator(["O","H","H"], basis="cc-pvdz", method="casscf",
                       avas="O 2p|H 1s")     # 或 ["O 2p", "H 1s"]
E, g = calc.energy_and_gradient(coords)       # AVAS 空间 + 轨道优化
# 可叠加: state_average/nstates/state (激发态) | fci_solver="sci" (大空间) | pt2="nevpt2"
```

| 检验（Slurm 1559297, 14 s, exit 0, A–E 全绿） | 结果 |
|---|---|
| 闭壳层**恒等式** | H₂O 'O 2p' → CAS(3,6)（CI 维数 1）时 E **= RHF**（8.5e-14，数学恒等式）✓ |
| 开壳层真相关 | N₂ ncas=7 → **0.132 Ha**；O₂ 三重态 → **0.096 Ha** ✓ |
| AVAS + CASSCF | −76.0652 → **−76.0797**（轨道优化 0.0146 Ha）✓ |
| 组合能力 | AVAS + 态平均 / **选择组态 CI**（5.9e-12 vs 稠密）/ NEVPT2（0.140 Ha）全部可叠加 ✓ |

⚠ 实测标定：PySCF 的 `avas` 只接受**标签列表**或单标签/正则串（`'O 2p|H 1s'`）；
`;`/`,` 分隔串会**静默返回 ncas=0** → 实现统一拆分为列表并在 provenance 记录。

## 复合方法: CBS 外推 + CCSD(T) 加和 (v0.34.0)

```python
from autoquantum.pes.composite import composite_energy, make_energy_fn
fn  = make_energy_fn(["O","H","H"], frozen_core=True)
out = composite_energy(fn, coords)     # E_HF/CBS + E_corr/CBS + δ_CCSD(T)
```

| 检验（Slurm 1559235, 79 s, exit 0, A–D 全绿） | 结果 |
|---|---|
| **公式精确重构** | 幂律 / 三点指数外推对合成序列误差 **0.00e+00 / 2.2e-16** ✓ |
| 外推质量 (vs cc-pV5Z, H₂O) | HF \|Δ\| **9.4e-4** (QZ 2.26e-3)、相关能 **6.9e-3** (QZ 8.7e-3) → 两分量都更接近大基组 ✓ |
| **绝对物理校验** | H₂ **R_e = 0.7414 Å（偏差 0.003%）**（归档收敛值 0.7414；单基组 CCSD(T)/TZ 0.198%）✓ |
| 加和诚实性 | \|δ\|/\|相关\| = 0.045；对 CBS 极限偏差 −1.05 mHa（5Z 本身 +0.25 mHa，变分自洽）✓ |

⚠ 诚实边界：**对已收敛体系两点外推会过冲**（H₂ 上比 5Z 更远离极限；改用
(QZ,5Z) 输入降到 −0.90 mHa）；这是加和近似，与 G4/W1 参数化方案不同 ——
且 G4/G3 专用基组 PySCF 未收录，无法忠实复现，故以"公式公开 + 逐项可核验"替代。

## 大活性空间: CASCI + 选择组态 CI (v0.33.0)

```python
calc = PySCFCalculator(["N","N"], basis="cc-pvdz", method="casci",
                       active_space=(14,14), fci_solver="sci",
                       sci_select_cutoff=1e-4, sci_ci_coeff_cutoff=1e-6)
E = calc.energy(coords)      # 稠密 FCI 维数 1.18e7, 端到端 ~9 s
```

| 检验（Slurm 1559215, 133 s, exit 0, A–E 全绿） | 结果 |
|---|---|
| **精确性** | SCI (紧阈值) ≡ 稠密 FCI: CAS(6,6)/STO-3G 与 CAS(8,8)/cc-pVDZ 均 \|Δ\| < 1e-6 Ha ✓ |
| **大活性空间** | **CASCI(14,14)/cc-pVDZ**（稠密 11,778,624 维）**9 s** 完成；比 CCSD(T) 高 198.7 mHa（无动态相关，合理）✓ |
| 变分单调性 | 阈值 1e-3 → 1e-4 下降，1e-4 与 1e-5 差 **0.003 mHa**（已收敛）；全程单调不升 ✓ |
| 势能曲线 | CASSCF(8,8) R_e = **1.1220 Å** vs 实验 1.098（+2.19%）；CASCI-SCI(14,14) R_e = **1.1215 Å**（一致 **0.0005 Å**），曲线平滑 ✓ |

⚠ 实测限制：PySCF 的 **CASSCF 驱动与 SCI 的 RDM 接口不兼容**（SCI 的 CI 对象是
`(civec, ci_strs)` 扩展形式）→ `method="casscf"`+`fci_solver="sci"` 会明确报错并
指向 `method="casci"`；CASCI 无轨道优化（短 R 处比小活性空间 CASSCF 高 30.4 mHa，
两者互补）。

## 态平均 CASSCF 激发态 (+ NEVPT2) (v0.32.0)

```python
calc = PySCFCalculator(["Li","H"], basis="6-31g", method="casscf",
                       active_space=(2,2), nstates=2, state_average=True,
                       state=1)                    # 第 2 个态 (自旋纯)
E, g = calc.energy_and_gradient(coords)            # 梯度为有限差分
calc2 = PySCFCalculator(..., pt2="nevpt2", state=1)  # 激发态 NEVPT2
```

CLI: `autoquantum scan --method casscf --active-space 2 2 --state-average
--nstates 2 --state 1 ...` → 直接扫描**激发态势能面**。

| 检验（Slurm 1559180, 514 s, exit 0, A–E 全绿） | 结果 |
|---|---|
| **精确性** | H₂/STO-3G 空间完备时 SA(2) = **FCI 单重态**（\|ΔE\| 8.9e-16 / 3.3e-16 Ha）✓ |
| **态身份/平滑性** | LiH 避交叉扫描：无交叉、ΔE 极小 1.5835 eV @ 5.60 Bohr（区间内）、CI 向量最小重叠 **0.999895** ✓ |
| 激发态 NEVPT2 | 修正为负、态序保持；对 FCI 拉近 **25.0 → 9.3 mHa**（态平均轨道代价 8–10 mHa 如实报告）✓ |
| 一致性 | 态平均基态 = 单态 = FCI（**0.0e+00**）；FD 梯度差 1.1e-12 ✓ |

⚠ 实现要点（PySCF 实测约束）：态平均必须用**自旋纯**求解器（默认会给出 ³Σu⁺
而非第二个单重态）；NEVPT2 不接受态平均求解器 → 改用同轨道独立多根 CASCI。

## 自旋-轨道耦合: 单电子 Breit–Pauli (v0.31.0)

```python
calc = PySCFCalculator(["O","H"], basis="cc-pvtz", spin=1, method="rohf")
out  = calc.soc_terms(coords)              # ζ 与精细结构分裂 (cm⁻¹)
out2 = calc.soc_state_interaction(         # 单重态-三重态耦合矩阵
    coords, active_orbitals=[3, 4], singlet_roots=2, triplet_roots=1)
```

CLI: `autoquantum soc --input g.xyz --method rohf --basis cc-pvtz [--orbitals ...]`

| 检验（Slurm 1559052, 19 s, exit 0, A–G 全绿） | 结果 |
|---|---|
| **类氢精确标定** | ζ = α²Z⁴/48 → 比 **0.99997**；²P 分裂 5.84348 vs 精确 5.84366 cm⁻¹ ✓ |
| 原子 ²P vs 实验 | F 1.457、Cl 1.117、Br **0.996**（单电子 BP 随 Z 逼近实验）✓ |
| 平移/旋转不变性 | 平移 4.8e-6 cm⁻¹（相对 3.8e-7）；旋转耦合模长相对差 8.5e-14 ✓ |
| C₂ᵥ 选择定则 | 只有 B₂ 分量；\|c(M=±1)\| 严格等量；M=0 分量 5.8e-15 (禁阻) ✓ |
| OH ²Π 两层交叉验证 | 轨道层 ζ = CI 层耦合 (**相对差 6.3e-16**)；vs 实验 A 比 1.63 ✓ |
| 密度/WET | 跃迁密度 vs PySCF 参考 **2.1e-16**；\|c(+1)\|=\|c(−1)\| 严格相等 ✓ |

⚠ 诚实边界：仅**单电子** Breit–Pauli（无二电子屏蔽）→ 轻元素偏大（F 1.46×）、
重元素接近实验（Br 1.00×）；`z_eff` 可作经验修正。验证期间修复两处真实根因：
libcint `int1e_prinvxp` 缺 −i 因子（使态相互作用矩阵元被 Hermitian 对称化抵消为零）、
`with_common_origin` 对该积分无效（改用 `with_rinv_at_nucleus`）。

## 激发态: EOM-CCSD (v0.29.0)

```python
calc = PySCFCalculator(["H","H"], basis="aug-cc-pvdz", method="eom-ccsd",
                       nstates=4, state=0)      # EOM-EE 单重态
E, g = calc.energy_and_gradient(coords)          # 梯度为有限差分
```

| 检验（Slurm 1558875, 668 s, exit 0） | 结果 |
|---|---|
| **EOM-CCSD vs FCI 单重态激发能** | H₂/aug-cc-pVDZ **偏差 0.000 eV（完全一致）** ✓ |
| vs TD-DFT 交叉一致性 | H₂O/6-31G* 首激发 8.675 vs 8.061 eV（差 0.61）✓ |
| 态身份保持窗口曲线 | 平滑（跳变比 0.129 < 0.3）；态序正确 ✓ |
| FD 梯度可用性 | 窗口内优化 ΔE = −7.7 mHa，\|g\|max ↓ 65× ✓ |

⚠ **负结果（如实记录）**：`follow=True` 的重叠判据根跟踪在 H₂ 交叉窗口**未能改善
连续性**（跳变比 1.951 vs 固定序号 0.057）→ 根跟踪标注为**实验性**，**态身份漂移
限制仍然存在**；改进方向为更细步长 + 微扰/对称性约束选根。

## 隐式溶剂: ddCOSMO / PCM / ddPCM / SMD (v0.28.0 → v0.30.0)

```python
calc = PySCFCalculator(["O","H","H"], basis="6-31g*", method="mp2",
                       solvent="water")      # ddCOSMO (默认), 或 solvent_eps=78.3553
calc = PySCFCalculator(["O","H","H"], basis="6-31g*", solvent_eps=78.4,
                       solvent_model="pcm", pcm_variant="IEF-PCM")   # C-PCM/IEF-PCM/COSMO/SS(V)PE
calc = PySCFCalculator(["O","H","H"], basis="6-31g*", solvent="water",
                       solvent_model="smd")  # 全溶剂化 (含非静电项; 仅 SCF 层)
E, g = calc.energy_and_gradient(coords)
```

覆盖 **SCF / post-SCF(MP2,CCSD) / TD-DFT / CASSCF** 四条路径（PySCF 的四个入口各不相同，
不支持的组合会明确报错而非静默退回气相），内置 11 种常见溶剂介电常数；
CLI 侧 `sample`/`opt`/`scan`/`freq` 均支持 `--solvent/--solvent-eps/--solvent-model/--pcm-variant`。

| 检验（Slurm 1558906, 278 s, exit 0, 8/8） | 结果 |
|---|---|
| ε→1 极限 | ddCOSMO/PCM **0.000000**；ddPCM 0（ε=1 时 PySCF 除零 → 用 1+1e-6 逼近）✓ |
| 介电单调性 | ddCOSMO −2.52→−5.17；PCM −2.83→−7.20 kcal/mol（ε=2→78.4）✓ |
| 跨模型一致性 | ddCOSMO −5.17 / PCM −7.20 / ddPCM −4.82；PCM 四变体展宽 1.34% ✓ |
| 极性趋势 | H₂O −7.20 / CH₄ −0.29 / He −0.00（PCM）✓ |
| **Li⁺ Born 标度** | PCM 隐含半径 2.184 Å ≈ 1.2×r_vdW（vdw_scale=1.2 自洽）✓ |
| 四类入口 | pcm/ddpcm 的 SCF/MP2/CASSCF + TD-DFT 位移（+0.60/+0.42 eV）全部可用 ✓ |
| 溶剂化梯度 | PCM 解析 vs FD **4.4e-07**；ddPCM 无解析模块 → FD（含溶剂响应）✓ |
| SMD 非静电项 | CH₄/water **+2.19 vs 实验 +1.95**（疏水空腔项）；H₂O −8.84 vs −6.32 ⚠ |

⚠ 诚实边界：PySCF 的 SMD 实现自带 "experimental" 标注（`smd_experiment`），H₂O 绝对
值与实验差 ~2.5 kcal/mol（含标准态约定差异），**不作定量精度主张**；ddPCM 在 PySCF
中标注 "under testing"，其 ε=1 会内部除零（已记录）。

## 激发态: TD-DFT / TDHF (v0.27.0)

```python
calc = PySCFCalculator(["O","H","H"], basis="cc-pvdz", method="tddft",
                       xc="b3lyp", nstates=6)
e_exc_eV, f = calc.excitation_spectrum(coords)      # 激发能 + 振子强度

calc_s = PySCFCalculator(..., method="tddft", nstates=6, state=0)
E_excited = calc_s.energy(coords)   # 激发态总能量 → 扫描/优化/NEB/IRC 全可用
```

| 检验 | 结果 |
|---|---|
| H₂O/B3LYP 最低激发 | **7.605 eV vs 实验 ~7.4（2.8%）** ✓ |
| H₂ B¹Σu⁺ 垂直激发 | **12.63 vs 文献 12.5 eV（1.0%）** ✓ |
| 振子强度 | 非负、Σf = 0.558 < 10（TRK 上界）✓ |
| 激发态曲线 | 平滑（max\|d²\|/max\|d¹\| = 0.251）；态序正确 ✓ |

⚠ **已记录的限制**：① 未实现激发态根跟踪（态交叉时 `state` 身份会变，扫描应
限制在态身份保持窗口内）；② TD-HF 对 Rydberg 态势阱形状不准（H₂ B 态实测
0.9–2.2 Å 单调下降）——准确激发态势能面需 **EOM-CCSD/CASSCF**（后续项）。

## NEVPT2 动态相关 (v0.26.0)

```python
calc = PySCFCalculator(["H","H"], basis="cc-pvdz", method="casscf",
                       active_space=(8, 2), pt2="nevpt2")   # (ncas, nelecas)
E, g = calc.energy_and_gradient(coords)     # 梯度默认有限差分
```

在 CASSCF 之上叠加二阶微扰动态相关，使**键能与势垒定量化**。
⚠ PySCF **无 CASPT2 模块**；NEVPT2 属同一层级且**无侵入态问题**，是更稳健替代。

| 体系 | 活性空间 | CASSCF−FCI | NEVPT2−FCI | 改善 |
|---|---|---|---|---|
| H₂, R=1.4 | (2,2) | 16.49 mHa | **5.81 mHa** | 2.8× |
| **H₂, R=1.4** | **(8,2)** | 0.36 mHa | **0.10 mHa** | 3.5× |
| LiH, R=3.0 | (2,2) | 14.64 mHa | **5.79 mHa** | 2.5× |
| LiH, R=4.5 | (2,2) | 8.52 mHa | **3.16 mHa** | 2.7× |

NEVPT2 优化键长 0.7645 Å < CASSCF 0.7704 Å（动态相关使键缩短，物理正确）；
CASSCF FD 梯度与解析梯度两条优化路径给出**同一 R（差 0.005%）**，交叉验证通过。

## IRC 与 X2C 标量相对论 (v0.24.0)

```python
from autoquantum.pes.neb import irc_path
path, info = irc_path(calc, symbols, ts_coords, step=0.08, direction=-1)
# path[0] = 过渡态; 沿虚频方向双向积分即得完整反应路径

calc = PySCFCalculator(symbols, basis="cc-pvdz", method="rhf",
                       relativistic="x2c")     # X2C 标量相对论
```

**服务器验证**（Slurm 1558653 / 1558690）：

| 检验 | 结果 |
|---|---|
| LEPS IRC 双方向 | ΔE = −0.0900 Ha，**单调 100%**，两方向**结果完全一致**（对称反应镜像自洽）✓ |
| H₃/CCSD IRC | 两端 100% 单调；片段 H–H 由 TS 的 0.9422 Å 收敛到 0.7956 Å（优化 H₂ 0.7609 Å）✓ |
| IRC 顶点 | 首步 Δ = −2426 µHa ✓ |
| **X2C vs 精确 Dirac** | H(Z=1) −6.633 vs −6.657 µHa（**0.4%**）；He⁺(Z=2) −107.535 vs −106.514 µHa（**1.0%**）✓ |
| X2C 梯度 vs FD | H2O/cc-pVDZ 7.9e-07 Ha/Bohr ✓ |

修复 IRC 的**两处真实 bug**（解析面快速迭代抓出）：① 质量加权换算方向写反
（`*√m` 应为 `/√m`，会让路径"飞出"）；② 初始步向量未带方向符号导致反向路径
**停滞在过渡态**。改用 Ishida–Morokuma 平均梯度后消除之字形振荡。

## 过渡态搜索: CI-NEB (v0.23.0)

```python
from autoquantum.pes.neb import neb_path, imaginary_mode_count
images, info = neb_path(calc, symbols, coords_reactant, coords_product,
                        n_images=9, climb=True)      # → MEP + TS
ts = images[info["ts_index"]]
print(info["barrier_eV"], info["converged"])
assert imaginary_mode_count(calc, symbols, ts) == 1   # 鞍点判据
```

**服务器验证**（Slurm 1558518 @ xc016, 2344 s, exit 0, 4/4）：

| 检验 | 结果 |
|---|---|
| LEPS 解析面（H+H₂） | 势垒 **2.9961 eV = 解析值 2.9961（0.00%）**，TS 几何与解析鞍点一致 ✓ |
| H₃ 交换（CCSD/cc-pVDZ） | 势垒 **10.241 kcal/mol**（CCSD(T)/CBS 9.60，小基组高估 6.7% 属预期）✓ |
| **鞍点虚频（关键判据）** | TS **恰 1 虚频 = −1464.6 cm⁻¹**，全谱 −1465/240/908/2080 cm⁻¹；对照 H₂ 为 0 虚频、4382.6 cm⁻¹ ✓ |
| TS 几何 | 线性（残差 1.8e-16）、对称（0.000%）、R_HH = 0.9422 Å vs 文献 0.93（1.31%）✓ |

本轮修复两个**关键 bug**（均由严格验证抓出）：
① NEB 切向的负索引回绕（`ext[i-1]` 在 i=0 取到终点 → 路径塌陷，势垒错到 44 eV）；
② **谐振频率漏掉虚频**（取"最大 nvib 个本征值"，而虚频是**负**本征值会被丢弃 →
过渡态被误报 0 虚频，这会让所有 TS 验证静默失效）。已改为在平动/转动正交补
空间内对角化，并补了鞍点回归测试。

## PES 工作流: 几何优化 / 内坐标扫描 / 谐振频率 (v0.22.0)

从「笛卡尔位移网格」升级到商用软件式的工作流 (全部基于解析梯度):

```bash
# 几何优化 (BFGS, 极小点)
autoquantum opt --input h2o.xyz --method ccsd(t) --basis cc-pvtz -o opt.npz

# 内坐标扫描 → PES 训练集 (含能量与力)
autoquantum scan --input h2o.xyz --method mp2 --basis cc-pvtz \
    --mode bond --atoms 0 1 --range 0.9 2.2 --n 21 -o scan.npz
autoquantum scan ... --mode angle --atoms 1 0 2 --range 80 140
autoquantum scan ... --mode relax-bond --atoms 0 1   # 每点约束优化

# 谐振频率 (数值 Hessian + 质量加权 + 平动转动投影)
autoquantum freq --input h2o.xyz --method mp2 --basis cc-pvdz --opt-first
```

Python API:

```python
from autoquantum.pes.optimize import optimize_geometry, harmonic_frequencies
from autoquantum.pes import scan as Scan
opt, info = optimize_geometry(calc, coords0, gtol=1e-5)   # BFGS + Armijo
freqs, finfo = harmonic_frequencies(calc, symbols, opt)   # cm⁻¹, 含虚频数
data = Scan.relaxed_scan_bond(calc, symbols, opt, 0, 1, r_grid)  # → AbInitioData
```

**服务器验证** (Slurm 1558433 @ xc002, 800 s, exit 0, 6/6):

| 检验 | 结果 |
|---|---|
| 几何优化 (BFGS) | H2O 收敛 \|g\|=7.8e-05 且为真极小; H2/CCSD(T)/cc-pVQZ = 0.7417 Å（文献 0.7417）✓ |
| 谐振频率 | H2O/MP2/cc-pVDZ 1679/3854/3974 vs 实验谐振 1649/3832/3943 cm⁻¹（偏差 1.86%），0 虚频 ✓ |
| 内坐标扫描自洽 | 键长扫描极小 vs 优化极小差 0.284%；键角差 0.015° ✓ |
| 松弛扫描 | 每点键长保持 2.2e-16 Bohr；垂直梯度 9.1e-04 ✓ |
| ECP / 赝势 | AuH (Au: cc-pVDZ-PP)：梯度 vs 有限差分 = **4.5e-07** ✓ |
| 端到端 | 81 点 MP2 扫描 → 力训练 → 留出集能量 RMSE **0.178% of span**（判据 <1%），力 RMSE 1.07% ✓ |

## 相关方法后端: MP2 / CCSD / CCSD(T) (v0.21.0)

`PySCFCalculator` 现已支持**相关波函数方法** (含核梯度), 面向 PES 生成的主力方法阶梯:

```python
from autoquantum.pes.calculators import PySCFCalculator
calc = PySCFCalculator(["O", "H", "H"], basis="cc-pvtz", method="ccsd(t)")
E, g = calc.energy_and_gradient(coords)   # 能量 + 核梯度 (Hartree/Bohr)
calc = PySCFCalculator(["O", "H", "H"], basis="cc-pvtz", method="mp2",
                       frozen_core=True)  # 冻结核
```

**服务器验证结果** (`scripts/validate_correlated_methods.py`, Slurm `liquid_high`):

| 检验 | 结果 |
|---|---|
| RHF/cc-pVDZ vs 文献 | H2O +0.019 / N2 +0.117 / H2 −0.005 mHa ✓ |
| **CCSD ≡ FCI** (实现硬检验) | H2 **0.0000 mHa**（2 电子严格相等）/ LiH 0.0108 mHa ✓ |
| 方法阶梯 E_CCSD(T)<E_CCSD<E_MP2<E_SCF | 4/4 体系 ✓ |
| 解析梯度 vs 有限差分 | rhf 4.1e-7 / mp2 3.0e-7 / ccsd 3.5e-7 / **ccsd(t) 3.5e-7** Ha/Bohr ✓ |
| H2 键长基组收敛 | cc-pVDZ 0.7633 → cc-pVTZ 0.7452 → cc-pVQZ 0.7444 Å（实验 0.7414，差 0.41%）✓ |

⚠ **重要实现说明**: PySCF 的 CCSD 解析梯度**不含 (T) 项**。本后端默认对 CCSD(T)
施加 **(T) 项的有限差分修正**（`grad_t_mode="fd"`，成本 6N 次 CCSD(T)），确保
"能量是 CCSD(T)、力也是 CCSD(T)"——否则力训练与几何优化会被系统性偏差污染。
`energy()` 已与梯度计算解耦（否则 CCSD(T) 纯能量调用会慢 13 倍以上）。

**与 Molpro/Gaussian 的能力对比与替代边界**: 见
[CAPABILITY_VS_COMMERCIAL.md](CAPABILITY_VS_COMMERCIAL.md)（含完整的 ✓/✗ 矩阵与
可度量判据）。

## 热速率常数 (v0.16.0)## 热速率常数 (v0.16.0)## 热速率常数 (v0.16.0)

```python
from autoquantum.analysis.rates import thermal_rate_constant, arrhenius_fit
# P(E) 来自集群扫描, 计算各温度下的 Boltzmann 加权速率常数
k_T = thermal_rate_constant(E_grid, P_grid, temperatures=[300, 600, 1200])
fit = arrhenius_fit(temperatures, rates)  # → Ea, log₁₀A, R²
```

## QCT 准经典轨线与量子对比 (v0.19.0)

```python
from autoquantum.dynamics import QCTEnsemble, QCTTrajectory, wigner_sample
ensemble = QCTEnsemble(pes, mass_R, mass_r, dt=0.5, max_steps=3000)
result = ensemble.run(E_grid, R0=6.7, r_mean=1.401, r_sigma=sigma_r,
                      reaction_criterion=lambda R, r: R < 1.5*r, n_traj=200)
# result.reaction_probs — 经典极限 P_react(E), 与量子波包交叉验证
```

运行量子 vs 准经典交叉验证脚本：
```bash
python scripts/compare_qct_quantum.py --n-traj 200 --wp-steps 3000
```

## 集群生产管线 (v0.15.0)

```bash
# 一条命令: 同步代码 → Slurm 提交 → 等待完成 → 取回 → 合并 → 出图
python scripts/production.py \
    --system H3_2D --pes leps \
    --e-min 0.10 --e-max 0.30 --n-points 12 \
    --chunks 4 --grid 192 144 --steps 2500 \
    --partition liquid_high

# GPU 加速 (A100 节点, air 分区)
python scripts/production.py --torch --dtype float32 \
    --partition air --e-min 0.10 --e-max 0.30 --n-points 24
```

结果输出至 `results/` (npz + png)，已加入 `.gitignore`。

## 远程计算 (c211 集群)

集群 (qdmovie): 登录 `login-server`, 计算节点经 **Slurm** 投递。
环境: `/storage/home` 全集群 NFS 共享 — micromamba 环境 `~/aqd-env`
(Python 3.12 + numpy/scipy/matplotlib + autoquantum) 与代码
`~/AutoQuantum` 对所有节点一致 (venv 符号链接跨 OS 镜像会失效,
勿用)。

| 分区 | 节点 | 规格 |
|---|---|---|
| `liquid_high` | xc001-016 (液冷) | 16 × 192 核 EPYC, MaxTime 无限 |
| `air` | xa001-002, xb001-002, xd001 | xb002 含 A100-40GB |

```bash
bash scripts/remote.sh sync                        # 同步代码 (共享存储)
bash scripts/remote.sh test                        # 当前节点跑测试
bash scripts/remote.sh submit liquid_high          # Slurm 投递验证作业
bash scripts/remote.sh submit air myjob.sbatch     # 投递自定义作业脚本
bash scripts/remote.sh run scripts/benchmark.py --quick
bash scripts/remote.sh fetch /storage/home/lih/output_xxx
```

已在 liquid_high (xc016) 与 air (xa002) 双分区验证: 85 项测试全绿。

注: 登录节点 (2 核/3.7G) 仅作跳板; GPU (A100) 加速为路线图项,
当前栈为纯 NumPy CPU。跨平台数值差异 (BLAS) 在 0.2% 量级, 已知。

## v0.11.0 亮点: 对称函数势能面与委员会不确定性

- **对称函数描述符** (`nn/symmetry.py`): Behler-Parrinello 径向+角度
  函数, 逐原子中心, 严格平移/旋转不变;
- **共享原子能量委员会** (`nn/ensemble.py`): E = Σᵢ E_atom(Φᵢ) —
  原子置换按构造严格不变 (H2 交换等), 同种原子共享参数;
- **OOD 不确定性**: 委员会标准差作为主动学习采样信号;
- CLI: `autoquantum fit --data d.npz --symmetry --committee 4`
  (数据集含 (n,3N) 坐标与 symbols);
- 测试: 对称性物理正确性 (置换/平移/旋转不变) + 委员会学习/OOD, 共
  85 项。

```python
from autoquantum.nn.ensemble import train_atomic_committee
committee, info = train_atomic_committee(symbols, coords, energies, n_models=4)
E, sigma = committee.predict_with_uncertainty(coords)  # sigma 越大越 OOD
```

## v0.13.0 亮点: 主动学习闭环

`nn/active_learning.py`: 委员会分歧选点 → Calculator 标注 → 增量训练,
池 RMSE 随轮次下降 (测试固定 >30%); 对接真实后端 (xtb/pyscf) 即可
生产采样。

## v0.12.0 亮点: GPU 加速 (A100 实测)

- `dynamics/wavepacket_2d_torch.py`: torch 后端波包传播 (可选依赖),
  与 NumPy 版逐项对拍验证 (fp64 ~1e-6);
- 集群 A100-40GB 实测: 384×288 网格 **4.6× 加速** (6.4→1.4 ms/步),
  768×576 网格 4.4× (21.0→4.8 ms/步); 已在 liquid_high/air 双分区
  验证 88 项测试。

```python
from autoquantum.dynamics.wavepacket_2d_torch import TorchWavePacket2DPropagator
prop = TorchWavePacket2DPropagator(pes, R, r, mass_R, mass_r, dt=0.5,
                                   dtype="float32")   # A100 fp32 最快
```

## 工程质量基线 (v0.10.0)

- **输入验证**: 网格/能量窗/势能有限性 fail-fast (`core/validation.py`)
- **可信度闸门**: 每次波包运行自动输出健康诊断 — 概率记账、通道和、
  未吸收比例、NaN、能量漂移; 告警直达日志
- **可复现**: `run_manifest.json` (配置+环境+git commit+结果+数据指纹);
  NN 模型自带训练卡 (数据 SHA-256/RMSE/超参)
- **实验性护栏**: 2D 定态求解器需显式 `allow_experimental=True`
- **持续集成**: ubuntu/macos × py3.11-3.13 测试矩阵 + wheel 构建
  (`.github/workflows/ci.yml`); 本地 `bash scripts/check.sh`
- **性能基线**: `python scripts/benchmark.py` (参考: 96×64 网格约
  2800 波包步/秒; NN 每轮约 34 ms, Apple M4)

```bash
bash scripts/check.sh              # 本地全量质量闸门
python scripts/benchmark.py        # 微基准 (含 --quick)
```

## v0.8.0 亮点: 电子结构后端与训练数据生成 (实验性)

- **统一 `Calculator` 协议**: 能量 (Hartree) + 核梯度 (Hartree/Bohr);
  实现 Analytic / XTB (子进程) / PySCF / ASE 适配器 + 内置 LJ 演示;
- **几何采样**: 笛卡尔位移网格 + 最小距离过滤 → 训练集 (npz, 带
  provenance);
- **CLI 闭环**: `autoquantum backends` → `sample` → `fit` → 代理面;
- **诚实边界**: 真实后端在发布环境未安装、未验证; 采样特征无置换
  对称性。详见 [book 第 3.6 节](book/chapters/03-势能面.md)。

```bash
autoquantum backends                              # 后端可用性
autoquantum sample --backend demo --input ref.xyz -o data.npz
autoquantum fit --data data.npz -o model.pkl --force-weight 1.0
```

## v0.7.0 亮点: 能量反卷积、收敛检查与教材

- **多宽度扫描 + 能量反卷积**: 恢复点值 T(E) (病态反问题 — 返回
  条件数与残差两个诊断量, 残差大时曲线仅作诊断);
- **传播收敛检查**: 基线 vs 加密设置对比 |ΔP_react|;
- **教材** `book/`: 10 章正文 + 附录, 面向新手到高手, 与代码逐行
  对应 (见 [book/README.md](book/README.md))。

## v0.6.0 亮点: 力训练

NN 拟合以 ∂V/∂x 为监督目标 (double backprop, 经有限差分校验):

- engine 2D 管线默认启用力训练 (`--nn-force-weight` 可调);
- Eckart 2D: 能量 RMSE 4.6×10⁻³ → 4.5×10⁻⁴ au, 梯度 RMSE
  2.2×10⁻² → 2.1×10⁻³ au/Bohr — 力目标同时正则化能量拟合;
- summary/日志输出能量与梯度双 RMSE 指标。

## v0.5.0 亮点: 多维 NN 势能面拟合与数据驱动管线

补齐 "PES 网格 → NN 代理面 → 量子动力学" 闭环:

- **多维输入** (n, d) 拟合, 二维 PES 网格直接训练;
- **输入/输出标准化**随模型持久化, `predict`/`gradient` 接受物理单位;
- **Adam 优化器** + 修正早停语义 (相对改进判据, 修复慢收敛尾部被截断);
- **解析输入梯度** `gradient(X)` (与有限差分校验一致, 为力训练铺路);
- **engine 二维管线**: 自动拟合 NN 代理面 (RMSE 日志与告警) →
  波包动力学运行在 NN 面上;
- **`AbInitioData.sample_function`**: 从任意势能函数生成带梯度的训练集。

## v0.4.0 亮点: 二维含时波包传播

新增 Jacobi 坐标 (R, r) 下的二维含时波包传播 (Split-Operator FFT):

- 复势对称分裂 (Strang 二阶), 网格边缘二次型 CAP 吸收, 吸收量按格点确定性归因;
- 通道分析: 反应/反射概率随时间的演化, 恒等式 `存活 + Σ吸收 = 1` 精确成立;
- 碰撞能扫描 (E = p²/2μ 语义), 结果与时间无关扫描接口兼容;
- 基态宽度初始化 (谐振子精确/Morse 谐振近似) 与入口通道平衡位置自动对心;
- 初态-CAP 重叠自检 (重叠 > 1e-4 时报错, 防止 v0.3.x 时代 LEPS 被静默污染的问题);
- 可视化: 密度快照 + PES 等高线 + 分界面, 通道概率曲线, GIF 动画。

数值验证 (细节见 RELEASE_VALIDATION.md):

| 验证项 | 结果 |
|---|---|
| 1D Eckart 垒波包 vs 解析谱透射率 (3 能量 × 3 网格) | 误差 < 4×10⁻⁴ |
| 可分离 2D Eckart vs 能量加权 1D 参考 (E=0.02/0.04) | 差 0.002 / 0.004 |
| 谐振子基态静止性 (126 步) | 密度漂移 < 4×10⁻⁶ |
| 概率恒等式 (全部测试) | ~10⁻⁹ |

## 安装

```bash
pip install -r requirements.txt   # numpy / matplotlib / scipy
pip install -e .                  # 或 pip install . 使用发布 wheel
```

## CLI 使用

```bash
# 一维 H₂ 散射 (Morse 势, 时间无关扫描)
autoquantum run -s H2_1D -p morse -o output_h2

# 二维 H + H₂ 反应, 含时波包 (默认: 2D 体系 auto → wavepacket)
autoquantum run -s H3_2D -p eckart -o output_h3
autoquantum run -s H3_2D -p leps -m wavepacket -o output_h3_leps

# 显式选择动力学方法
autoquantum run -s H2_1D -m wavepacket        # 一维含时波包
autoquantum run -s H3_2D -m stationary        # 实验性 2D 时间无关 (仅定性)

autoquantum info
```

输出目录包含 `report.html` (汇总报告)、PES 等高线、反应概率曲线、
波包快照/概率图与 `wavepacket.gif` 动画。

## Python API 示例

```python
import numpy as np
from autoquantum.pes.eckart import EckartBuilder
from autoquantum.dynamics import (
    WavePacket2D, WavePacket2DPropagator, WavePacket2DScan,
    h3_reduced_masses, harmonic_ground_width, eckart_product_mask,
)

mass_R, mass_r = h3_reduced_masses()          # (2m/3, m/2), m = 1836.15 au
builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                         "r0": 1.401, "coupling": 0.08})

R = np.linspace(0.5, 9.0, 192)
r = np.linspace(0.5, 3.5, 144)
prop = WavePacket2DPropagator(builder.evaluate_2d, R, r, mass_R, mass_r,
                              dt=0.5, cap_edges=("R_min", "R_max"))
packet = WavePacket2D(R0=6.0, r0=1.276, sigma_R=0.5,
                      sigma_r=harmonic_ground_width(0.5, mass_r))

scan = WavePacket2DScan(prop, packet, eckart_product_mask(2.0), ("R_min",),
                        reactant_mask=lambda R, r: R >= 2.0, n_steps=3000)
result = scan.run(0.005, 0.05, 5)   # 碰撞能扫描 → P_react(E)
```

完整示例: `python autoquantum/examples/h3_wavepacket.py` (Eckart / LEPS)。

## 项目结构

```
autoquantum/
├── cli.py                      # 命令行接口
├── core/engine.py              # AutoPipeline 编排 (方法路由/守恒检查)
├── pes/                        # Morse/Harmonic/LJ/LEPS/Eckart + 数据容器
├── nn/                         # 纯 NumPy 前馈网络 (线性输出层, 早停恢复最优)
├── dynamics/
│   ├── quantum_1d.py           # 1D 定态散射 (Numerov)
│   ├── quantum_2d.py           # 2D 定态 (实验性, 不作定量结论)
│   ├── wavepacket.py           # 1D 含时波包 (Split-Operator + CAP)
│   └── wavepacket_2d.py        # 2D 含时波包 (复势分裂 + 通道分析)
├── visualization/              # 1D/2D 图, 波包动画, HTML 报告
└── examples/                   # h1d_scattering / h3_reaction / h3_wavepacket
```

## 支持体系

| 体系 | PES | 动力学 | 验证状态 |
|------|-----|--------|----------|
| H₂ 1D | Morse / Harmonic | 透射/反射 (定态扫描) | 有基础测试, 未做解析基准 |
| H₂ 1D | 任意 | 1D 含时波包 | CAP 守恒已验证 |
| H+H₂ 2D | Eckart | 2D 含时波包 | vs 能量加权解析参考已验证 |
| H+H₂ 2D | LEPS (教学参数) | 2D 含时波包 | 通道守恒已验证; 阈值为模型量, 非真实 H+H₂ |
| H+H₂ 2D | Eckart/LEPS | 2D 定态 (实验性) | ⚠️ 未验证, 不可作定量结论 |
| 任意 1D/2D | NN 拟合代理面 | 透射/反射 / 波包 | 梯度 FD 校验; Eckart 2D RMSE ~0.3% V span |

## 功能边界与已知限制

- **从头算 PES (v0.8.0, 实验性)**: `Calculator` 协议 + XTB/PySCF/ASE
  后端 + `sample`/`fit` CLI 闭环已实现; 真实后端在本仓库发布环境
  **未安装、未验证**, 采样特征为笛卡尔坐标 (无置换对称性), 采样
  非自适应。
- **NN 拟合**: 多维输入/标准化/Adam/解析梯度/力训练已具备;
  力训练依赖解析或差分梯度来源, 从头算梯度 (ASE/PySCF) 仍缺失。
- **NN 代理面动力学精度以 RMSE 日志为准**: 拟合误差直接传导进
  反应概率; engine 在 RMSE > 2%·V_span 时告警。
- **LEPS 默认参数是教学模型**: 共线交换 MEP 势垒 ~0.14 au (3.8 eV),
  远高于真实 H+H₂ (~0.4 eV); 其反应阈值不是真实体系的物理量。
  需要真实量标时用 Eckart 模型 (垒高 0.015 au ≈ 0.41 eV)。
- **2D 定态求解器为实验性**: 单通道递推、无流归一化, 引擎会输出警告。
- **波包能量扫描是有限时间估计**: 单一能量宽度、固定传播时长, 未做
  能量反卷积; 定量使用须自行做网格/时长/能量宽度收敛检查。
- `scripts/sync_github.sh` 指向旧仓库且会改写 SSH 配置, **勿运行**。
- 模型保存用 pickle, 只加载可信文件。

## 路线图

- [x] 一维量子散射 (定态 + 含时波包)
- [x] 二维含时波包传播 (Eckart / LEPS, v0.4.0)
- [x] 多维 NN 拟合 + NN 代理面动力学 (v0.5.0)
- [x] 力训练 (double backprop, v0.6.0)
- [x] 能量反卷积 + 收敛检查 (v0.7.0, 病态反问题, 带诊断)
- [x] CLI 与 HTML 报告
- [x] 教材 (book/, v0.7.0)
- [x] 电子结构后端协议 + 数据生成 CLI (v0.8.0, 实验性; 真实后端未验证)
- [ ] 主动学习/对称性特征 (等变性扩展)
- [ ] 四原子以上复杂体系
- [ ] GPU 加速
