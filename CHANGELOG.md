# Changelog

## 0.25.0 — CASSCF 多参考 (键断裂/强关联 PES)

### Added

- **CASSCF** (`PySCFCalculator(method="casscf", active_space=(ncas, nelecas))`):
  RHF/ROHF 参考 + 多组态自洽场 (轨道与 CI 同时优化), 覆盖单参考方法失效的
  键断裂与强关联区。
- **CASSCF 梯度**: 默认**中心有限差分** (正确性优先)。原因: **PySCF 没有
  CASSCF 解析梯度模块** (`pyscf.grad.mcscf` 不存在); `mc.nuc_grad_method()`
  返回的是 **CASCI 型**梯度 (缺轨道响应), 实测对 CAS(2,2)/H₂ 恰好精确
  (vs FD 3.8e-07) 但对 CAS(4,4)/H₂O **偏差 123 Ha/Bohr**。
  `grad_t_mode="casci"` 可取该近似 (仅建议快速预估)。

### Verified (服务器 Slurm 1558715 + 1558721, 日志已归档)

| 检验 | 结果 |
|---|---|
| H₂/cc-pVDZ 解离曲线 | R=4.0 Bohr：**\|RHF−FCI\| = 105.6 mHa → \|CASSCF(2,2)−FCI\| = 0.9 mHa**（R=5.0 时 0.14 mHa）；RHF 在拉伸区严重高估而 CASSCF 恢复正确的解离行为 ✓ |
| LiH/cc-pVDZ CAS(2,2) | RHF−FCI = 31.1/34.1 mHa → CASSCF−FCI = 14.6/8.5 mHa ✓ |
| 自然轨道占据数 | R=4.0 Bohr 前两位占据 = 0.742/0.377（显著部分占据 = 多参考特征）✓ |
| 梯度机制诊断 | CAS(2,2)/H₂ 的 CASCI 型梯度 vs FD = 3.8e-07；CAS(4,4)/H₂O 失效（123 Ha/Bohr）→ 默认改 FD ✓ |

注: 计划中的完整重跑（含 FD 梯度优化交叉验证）因集群被其他作业占满
（xc002 176/192 CPU，`import pyscf` 需 97.7 s）而中止；上表结果来自
已完成并归档的运行。

## 0.24.0 — IRC 内禀反应坐标与 X2C 标量相对论

### Added

- **IRC** (`pes/neb.py: irc_path`): 从过渡态沿虚频方向的质量加权最速下降积分
  (Ishida–Morokuma 平均梯度抑制定步长路径的之字形振荡); 数值 Hessian 的最低
  (负) 本征矢给出初始方向; 返回路径 + 能量/梯度范数/单调性诊断。
- **X2C 标量相对论** (`PySCFCalculator(relativistic="x2c")`): 基于 X2C 单电子
  哈密顿量; 与非相对论共用同一梯度/相关方法路径 (PySCF `mf.x2c()`)。
- **测试**: 本地 +2 (IRC 对称性与单调性回归; 共 16 项于
  `tests/test_optimize_scan.py`), 本地总数 130。

### Fixed (IRC 两处真实 bug, 由解析面快速迭代抓出)

1. **质量加权换算方向错误**: `dx = dq * sqrt(m)` 应为 `dq / sqrt(m)`
   (q_i = √m_i x_i) —— 写错会让 IRC 沿错误方向"飞出"(能量反升)。
2. **初始步向量未带方向符号** + 粗糙的"翻转"修正, 使反向路径被翻回过渡态而
   **停滞** (小步长时 ΔE 恒为 0)。改用 Ishida–Morokuma 平均梯度后,
   双方向 100% 单调且对称反应两方向结果完全一致。

### Verified (服务器 Slurm, exit 0)

| 检验 | 结果 |
|---|---|
| A. LEPS IRC 双方向 (1558653) | ΔE = −0.0900 Ha, **单调 100%**, 两方向**完全一致**（对称反应镜像自洽）✓ |
| B. H₃/CCSD IRC (1558653) | 两端 100% 单调; 片段 H–H 由 TS 的 0.9422 Å 收敛到 0.7956 Å（优化 H₂ 0.7609）✓ |
| C. IRC 顶点 (1558653) | 首步 Δ = −2426 µHa（必须下降）✓ |
| D. **X2C vs 精确 Dirac** (1558690, 94 s) | H(Z=1)：−6.633 vs −6.657 µHa（**0.4%**）；He⁺(Z=2)：−107.535 vs −106.514 µHa（**1.0%**）✓ |
| X2C 梯度 vs FD | H2O/cc-pVDZ：7.9e-07 Ha/Bohr ✓ |
| X2C 重元素位移 | AuH（含 ECP）：−35.8 mHa（显著）✓ |

## 0.23.0 — 过渡态搜索 (CI-NEB) 与两处关键 bug 修复

### Added

- **`pes/neb.py`**: Climbing-Image Nudged Elastic Band — 只依赖解析梯度
  (对解析面/ML 代理面/电子结构后端通用); 镜像用 **FIRE** (自适应步长 +
  力归一化 + 速度上限) 演化; `refine_saddle`, `imaginary_mode_count` 辅助。
- **测试**: `tests/test_optimize_scan.py` 增加 NEB 回归测试 (切向无回绕;
  LEPS 势垒 vs 解析鞍点; 对称反应 ΔE=0) 与**鞍点虚频回归测试**
  (抛物线势垒必须报 1 虚频) — 本地 128 项。

### Fixed (两处关键 bug, 均由严格验证抓出)

1. **NEB 切向的负索引回绕**: `ext[i-1]` 在 `i=0` 时取到路径**终点**,
   使首个镜像切向反向 → 整条路径塌陷、势垒完全错误 (实测 44 eV)。
   改为正确的中心差分 `ext[i+2] - ext[i]`。
2. **谐振频率漏掉虚频**: 原实现取"最大的 nvib 个本征值", 而虚频对应**负**
   本征值, 会排到数值零模之下被丢弃 → 过渡态被误报为 0 虚频。
   改为在**平动/转动正交补空间内**对角化 (`Q^T H Q`)。
   ⚠ 该 bug 会让所有 TS 验证静默失效, 却对极小点完全无害 (故此前未被发现)。
3. 势垒参照改为**反应物端点** (原用第一个镜像, 因镜像位于路径内部而有偏差)。

### Verified (服务器 Slurm 1558518 @ xc016, 2344 s, exit 0, A–D 全绿)

| 检验 | 结果 |
|---|---|
| A. LEPS 解析面 (H+H₂) | CI-NEB 势垒 **2.9961 eV = 解析鞍点 2.9961 (0.00%)**；TS R₁=R₂=2.212 Bohr（解析 2.211）✓ |
| B. H₃ 交换 (CCSD/cc-pVDZ) | 势垒 **10.241 kcal/mol**（CCSD(T)/CBS 9.60；小基组高估 6.7% 属预期），收敛 744 次评估 ✓ |
| C. **鞍点虚频（关键）** | TS **恰 1 虚频 = −1464.6 cm⁻¹**；全谱 −1465/240/908/2080 cm⁻¹（线性三原子 3N−5=4）；对照 H₂ 优化后 0 虚频、ν = 4382.6 cm⁻¹ ✓ |
| D. TS 几何 | 线性残差 1.8e-16；对称偏差 0.000%；R_HH = 0.9422 Å vs 文献 0.93 Å（1.31%）✓ |

## 0.22.0 — PES 工作流: 几何优化、内坐标扫描、谐振频率、ECP

### Added

- **几何优化** (`pes/optimize.py`): BFGS + Armijo 回溯线搜索 (解析梯度);
  可选步长上限防止跳入非物理区; 返回收敛信息与真极小判据。
- **谐振频率 / 数值 Hessian**: 中心差分 Hessian + 质量加权 + 平动/转动投影;
  `core/periodic.py` 收录 36 个元素的最常见同位素质量 (CODATA)。
- **内坐标扫描** (`pes/scan.py`): `scan_bond` / `scan_angle` / `scan_path` /
  `relaxed_scan_bond` (约束优化, 梯度投影) → 直接产出含能量与力的
  `AbInitioData` (含 symbols 与 geometry, 可喂对称函数委员会拟合)。
- **ECP / 赝势**: 混合基组 (如 `{"Au": "cc-pVDZ-PP", "H": "cc-pVDZ"}`) 可用,
  PySCF 自动加载 ECP。
- **CLI**: `autoquantum opt` / `scan` / `freq` 三个子命令 (+ `--frozen-core` 等
  共用后端参数), 对标 Gaussian 的 `opt/freq/scan` 使用方式。
- **测试**: 新增 `tests/test_optimize_scan.py` (12 项解析模型测试, 无需 pyscf):
  二次型/Morse 极小、谐振子频率 √(k/μ)、同位素位移、3N−6 模式数、
  扫描自洽与松弛约束 —— 本地总数 125。

### Verified (服务器 Slurm 1558433 @ xc002, 800 s, exit 0, A–G 全绿)

- 几何优化: H2O 收敛 |g|=7.8e-05 且逐坐标位移能量升高 (真极小);
  H2/CCSD(T)/cc-pVQZ = 0.7417 Å (文献 0.7417); N2/CH4 亦通过。
- 谐振频率: H2O/MP2/cc-pVDZ 1679/3854/3974 vs 实验谐振 1649/3832/3943 cm⁻¹
  (最大偏差 1.86%), 虚频数 0。
- 内坐标扫描: 键长扫描极小 vs 优化极小差 0.284%; 键角差 0.015°;
  数据容器 (points/gradients/symbols/geometry) 完整。
- 松弛扫描: 每点键长保持 2.2e-16 Bohr, 垂直梯度 9.1e-04 Ha/Bohr。
- ECP: AuH/cc-pVDZ-PP 梯度 vs 有限差分 = 4.5e-07 Ha/Bohr。
- 端到端: 81 点 MP2/cc-pVDZ 双坐标扫描 → NN 力训练 → 留出集 (17 点)
  能量 RMSE 0.070 mHa = **0.178% of span** (判据 <1%), 力 RMSE 1.07%。

### Fixed (由服务器测试抓出)

- `relaxed_scan_bond` 的约束梯度投影在 3 原子模型下的广播错误 (测试用势已改为
  通用成对势); 键长扫描的抛物拟合需按非谐性放宽容差 (谐振子用严格判据)。
- 端到端拟合的梯度张量形状 (n, N, 3) → (n, 3N) 训练器约定。

## 0.21.0 — 相关方法后端 (MP2/CCSD/CCSD(T)) 与商用软件能力矩阵

### Added

- **相关波函数方法** (`pes/calculators.py`): `method` 支持 `mp2` / `ccsd` /
  `ccsd(t)`, 返回总能量与**相关核梯度** (MP2 → `grad.mp2`; CCSD/(T) → `grad.ccsd`);
  `frozen_core=True` 使用化学冻结核 (`pyscf.data.elements.chemcore`)。
- **CCSD(T) 的 (T) 项梯度修正** (`grad_t_mode="fd"`): PySCF 的 `grad.ccsd`
  **不含 (T) 项** (源码无 `ccsd_t` 路径), 直接用会造成"能量 CCSD(T)、力只有 CCSD"
  的不一致。默认以 (T) 增量的中心有限差分补足 (成本 6N 次 CCSD(T)),
  验证: 解析梯度 vs FD = 3.5e-07 Ha/Bohr (修复前 1.6e-03)。
- **`energy()` 与梯度解耦** (`_run(..., need_grad=False)`): 避免纯能量调用触发
  6N 次 CCSD(T) 的 (T) 梯度 (实测慢 13 倍以上, 作业因此卡住)。
- **`CAPABILITY_VS_COMMERCIAL.md`**: 与 Molpro/Gaussian 的完整能力矩阵
  (含 ✓/✗ 证据与"可替代"的可度量判据)。
- **`scripts/validate_correlated_methods.py`** + `scripts/sbatch_corr_validate.sbatch`:
  服务器 (Slurm) 验证套件 — RHF vs 文献、**CCSD ≡ FCI**(H2 严格 0.0000 mHa;
  LiH 0.011 mHa)、方法阶梯一致性、四方法梯度 FD 校验、H2 键长基组收敛阶梯、
  冻结核开关。**6/6 全绿** (节点 xc002, 约 22 min)。
- Mock 测试 +3 (共 17): 相关方法分发、(T) 梯度级别 provenance、`energy()` 不触发梯度。

### Fixed

- **重复的 `_run` 定义**: 类中存在两个 `_run` (后者覆盖前者), 导致相关分支从未
  执行 — 由服务器验证抓出 (所有方法返回同一 SCF 能量)。
- **`frozen` API 误用**: `mp.MP2`/`cc.CCSD` 的 `frozen` 是**构造参数**,
  不是 `kernel()` 参数 (原实现抛 `TypeError`)。
- 测试判据修正: 2 电子体系的 (T) 恒为零属物理正确 (非缺陷); H2 键长必须用
  **基组收敛阶梯**而非实验值直接比对; 冻结核判据符号。

## 0.20.3 — 参考曲线误差归因与 Γ 模型化验证

### Findings (决定性诊断, 均可复现)

- **参考离子曲线的出处被识别**: `scripts/ref_basis_id.py` + `ref_method_id.py` 用
  基组/方法阶梯复现参考 `HeLip.txt`, 结论为 **cc-pVQZ 级别 + 未做 counterpoise 校正**
  的 CCSD/CCSD(T) (19 点 MAE 0.71 meV, 最大 2.4 meV, R=3.2 Bohr 处 0.03 meV);
  混合基组 (cc-pVQZ/cc-pVTZ 互换)、MP2 (偏差 70 meV)、SCF (15.6 meV)、
  Cartesian 基函数均显著更差。
- **参考曲线在阱底含 ~10 meV 未校正 BSSE** (占其 85.91 meV "阱深" 的 12%),
  且在 R>5 Å 被拉平为渐近平台 (偏离物理 −C₄/R⁴ 尾 0.5 meV)。
  → "全区间 MAE 2.09%" 的主因是参考数据自身的系统误差, 非本工作精度。
- **本工作 BSSE 自由阱深确定**: 4 个 QZ 级基组的 CP 值收敛至
  cc-pVQZ 77.67 / aug-cc-pVQZ 78.36 / 自建 cc-pVQZ+3diff (nao=208) 78.80 meV
  → 最佳估计 **78.3 ± 0.6 meV (自身不确定度 0.7%, < 1%)**;
  def2-QZVPPD-CP (75.50) 与 aVTZ-CP (73.42) 为欠收敛 (基组极化不足 / BSSE 过大)。

### Added

- **`scripts/width_model_validation.py`**: Γ(R) 模型化验证 — log-三次样条 + 指数尾
  (k = 3.28 /Å ²Σ, 4.38 /Å ²Π) + ICD C₆; 对参考 MRCI 全部 17/16 个数据点的
  **最大相对误差 0.0002%** ✓ (即生产管线使用的 Γ(R) 输入与参考数据完全一致)。
- `scripts/large_basis_ion.py` / `moderate_basis_check.py` / `ref_basis_id.py` /
  `ref_method_id.py` / `ref_frozen_test.py`; validation/ 同步收录全部 npz 证据。

### 误差表 (v0.20.3 终版)

| 指标 | 基准 | 结果 | 误差 | 达标 |
|---|---|---|---|---|
| He* 激发能 | NIST 19.8196 eV | FCI/aVQZ | +0.27% | ✓ |
| Li 电离能 | NIST 5.3917 eV | FCI/aVQZ | −0.40% | ✓ |
| 离子阱区 R=5–9 bohr | 参考样条 | CCSD(T)+CP | 0.81–0.85% | ✓ |
| Γ(R) 模型 (管线输入) | 参考 MRCI 数据 | log 样条+指数尾 | 0.0002% | ✓ |
| 离子阱深 | 参考(去BSSE) 76–78 meV | 78.3 ± 0.6 meV | 0.8%① | ⚠ |
| 离子全区间 MAE | 参考样条(原始) | CCSD(T)+CP/aVQZ | 2.09% | ✗ |
| Γ 第一性原理 | 参考 MRCI | CAP-CI (机制已验证) | 不可分辨 | ✗ |

① 参考方法出处未记录 → 其 BSSE 自由值只能定到 76–78 meV, 故一致性为 0.8–3%。

## 0.20.2 — 误差 < 1% 迭代审计与 CAP-CI 实现

### Added

- **`scripts/target_1pct_audit.py`**: 误差审计与迭代工具 — 原子量 (RHF/ROHF→FCI,
  基组阶梯 aVTZ→aVQZ) + He+Li⁺ 曲线 (RHF→CCSD(T), counterpoise 校正),
  自动计算相对 NIST 与参考样条的误差并判定 <1%。
- **`scripts/error_budget_report.py`**: 汇总误差表 (含 CP 序列 CBS X⁻³ 外推)。
- **`scripts/cap_ci.py` + `scripts/cap_ci_search.py`**: **CAP-CI 第一性原理共振
  计算** — DFT 数值网格构造复吸收势算符矩阵; 用 PySCF 实数 FCI 引擎逐列构造
  复 FCI 矩阵 (活性空间小, 稠密复对角化); η 轨迹 + 平台判据搜索真实共振。
  **机制已在束缚态上验证正确**: He+Li⁺ 的 Γ 严格正比于 η 且 → 0
  (η=2e-4…4e-3 → Γ=0.0002…0.0042 meV), 能量与 SCF 一致 (0.1 mHa)。

### Results (误差 < 1% 迭代)

| 指标 | 迭代前 | 迭代后 | 达标 |
|---|---|---|---|
| He* 激发能 (vs NIST) | ROHF −5.0% | FCI/aVQZ **+0.27%** | ✓ |
| Li 电离能 (vs NIST) | ROHF −0.9% | FCI/aVQZ **−0.40%** | ✓ |
| HeLi⁺ 阱区 (R=5–9 bohr) | RHF 2.41% | CCSD(T)+CP/aVQZ **0.85%** | ✓ |
| HeLi⁺ 全区间 | RHF 2.8% | CCSD(T)+CP/aVQZ 2.09% | ✗ |
| HeLi⁺ 阱深 | — | 78.4 (CP) vs 85.9 (参考) | ✗ |
| Γ(R) | 模型 | CAP-CI 机制验证; 共振不可分辨 | ✗ |

- 阱区 MAE 在 **aVTZ / aVQZ / def2-QZVPPD 三个独立基组一致** (0.69/0.73/0.70 meV),
  说明该区间已收敛; def2-QZVPPD 的 BSSE 仅 2.8 meV (aVTZ 21.3/aVQZ 33.9),
  其未校正阱深 78.3 meV 为最可靠无偏估计。
- 全区间/阱深未达标的成因已定位并记录: 阱底 CP 过校正 (离子体系已知问题) +
  参考数据在 7.6 Å 后拉平为渐近平台 (非本工作误差)。

### Known limitation (诚实记录)

- He*+Li 的 ²Σ⁺ 自电离共振 (Γ≈10 meV, 出射电子动能 14.4 eV) 在
  aug-cc-pVTZ (+额外弥散) + ≤11 轨道活性空间下**不可分辨**: 谱中只有
  连续谱赝态 (Γ ∝ η 线性增长, 无平台)。定量 Γ 需要专门的连续谱基组
  (高角动量弥散函数) 或 Feshbach 投影实现, 超出当前框架。

## 0.20.1 — He*+Li 结果校核与生产参考势交叉验证

### Fixed (科学结论更正)

- **⁴Σ⁺ "范德华阱 54 meV" 证伪**: counterpoise (CP) 校正显示 R=8/10/12 Bohr 处
  BSSE 分别为 30/11/3 meV, 恰好构成 0.20.0 报告的"阱"; CP 校正后 |V| < 1 meV。
  ⁴Σ⁺ 在 SCF/CASCI 层面无真实束缚。
- **彭宁电子能量标注更正**: 正确值 $E_e = 14.4279\ \text{eV} + [V_{2\Sigma}-V^+]$
  (14.4279 = He\*(2³S) 激发能 19.8196 − Li 电离能 5.3917); 0.20.0 中
  "18.8 eV 垂直激发能隙 = 彭宁电子可用能量" 的概念混淆已更正 (差 4.4 eV)。
- **解析宽度模型量级错误**: $\Gamma=0.04e^{-1.1R}$ Ha (R/bohr) 在 R=6 Bohr
  处比参考 MRCI 数据小 5 个数量级且衰减过陡; 应改用 MRCI $\Gamma(R)$ 数据。

### Added

- **`scripts/validate_vs_reference_pes.py`**: ⁴Σ⁺/²Σ⁺/离子曲线 + CP 校正 +
  片段状态自检 (全部 gto.M 显式 `unit="Bohr"`; 记录 PySCF 默认 Å 单位踩坑)。
- **`scripts/plot_heli_vs_reference.py`**: 与生产参考数据 (Pro_HeLi_Enhanced:
  MLR 4 通道 + HeLi⁺ 离子势 + MRCI 宽度) 的四联对比图, 含修正后的彭宁电子能量。
- **CASCI(5e,12o) 轨道匹配方法**: 四重态轨道基组解 ⁴Σ⁺ (解离极限偏差 −0.00 mHa),
  双重态轨道基组解基态 (偏差 −0.01 mHa); 揭示 R≈6 Bohr 处 ~330 meV 避免交叉势垒
  (与 He(2³P)+Li 的 ⁴Σ⁺ 通道耦合), 解释短程 SCF 分支歧义的根源。

### Findings (对参考生产数据的核对, 供上游确认)

- **HeLi⁺ 离子通道互证通过**: 本工作 RHF 最小值 −68 meV @ 3.6 Bohr vs 参考
  HeLip.txt 曲线 −86.5 meV @ 3.65 Bohr (差 ~20 meV, 符合 RHF 缺相关能预期)。
- **参考 `potentials.py` 单位约定存在不一致**: 参数注释为 `R_e`/bohr、
  `C6/C8/C10`/a.u., 代码却按 Angstrom 代入; 仅当解读为 "R_e 为 Angstrom +
  C 为 eV·Angstrom⁶" 时, 四个通道的 MLR 指数才同为正值 (0.65–1.51, 物理自洽,
  C6(³S)=3498 a.u.); 按字面混合单位则该指数全负 (−1.79~−2.65), 长程尾偏强
  27 倍 (R=50 Bohr: −0.19 vs −0.006 meV), 短程排斥墙偏陡 (R=4 Bohr: +67 eV)。
  建议核对原始拟合的单位约定。

## 0.20.0 — 亚稳态体系从头算: 高自旋锁定、MOM 与真实集群算例

### Added

- **`PySCFCalculator` 高级电子结构能力** (`pes/calculators.py`):
  - **高自旋约束与自旋锁定**: 支持 ROHF/ROHF/ROKS/UHF/UKS 方法分发;
    `spin_lock=True` 时收敛后用 `mf.spin_square()` 审计自旋纯度, 偏差超过
    `spin_tol` 抛出 `CommandBackendError`, 从物理上杜绝激发态变分塌陷;
  - **最大重叠法 (MOM)**: `use_mom=True` + `mom_reference={'prev','initial'}`,
    按 PySCF 2.x API `mom_occ(mf, occorb, setocc)` 注入 alpha/beta 分离占据
    (ROHF 的 1 维 {2,1,0} 占据自动展开为 (2, nmo)), 沿几何路径跟踪指定组态;
  - **复共振势接口**: `resonance_width(coords)` (指数/盒式模型) 与
    `complex_energy(coords)` 返回 $W(R)=V(R)-i\Gamma(R)/2$, 供非厄米波包传播;
  - `_mol()` 显式 `mol.build()`; `conv_tol`/`max_cycle`/`xc` 可配置;
    `provenance` 完整记录 method/basis/charge/spin/spin_lock/use_mom/xc/cap。
- **CLI**: `autoquantum sample` 新增 `--spin`, `--method`, `--basis`, `--xc`,
  `--spin-lock`, `--spin-tol`, `--mom` (修复此前 PySCF 自旋参数无法从命令行
  传入的断层)。
- **真实集群算例** `scripts/calc_metastable_heli.py` + `scripts/sbatch_he_li.sbatch`:
  He\*(2³S)+Li 四重态/双重态势能面 (aug-cc-pVTZ, 30 点 ROHF+spin_lock+MOM 链),
  含原子渐近 FCI 校验、垂直能隙 $\Delta E(R)$、自电离宽度 $\Gamma(R)$ 与四联图;
  `--replot` 支持从 npz 离线重绘 (无需 pyscf)。已在 c211→liquid_high 实测:
  FCI ³S–¹S = 19.88 eV (实验 19.82), 渐近一致性 0.9 mHa, 全程 30.6 s。
  **注: 早期报告的 vdW 阱 54 meV 已于 0.20.1 校核中证伪 (BSSE 赝像)。**
- **测试**: 新增 8 项 PySCF Mock 测试 (方法分发/自旋锁定拦截/MOM setocc 格式/
  CAP 宽度/复能量), 共 109 项; 无需安装 pyscf 即可在 CI 全绿。

### Fixed

- **MOM 调用签名**: 改为 PySCF 2.x 的位置参数 `mom_occ(mf, occorb, setocc)`
  并构造 ROHF/UHF 所需的 alpha/beta 占据数组 (原 `set_occ=` 关键字在
  PySCF 2.14 不存在)。
- **`remote.sh` 默认主机/解释器**: `c211` + `~/aqd-env` (原 target-server/aqd-venv),
  并抑制 macOS 扩展属性导致的 tar 噪音。
- **教材配图 CJK 字体**: 跨平台字体探测 (Noto CJK/Fandol/Droid/Songti),
  对数轴改用纯文本指数格式器, 避免 CJK 字体缺 U+2212 导致负号丢失。

### Documentation

- 教材第 4 章新增"亚稳态体系与自电离共振 (He\*+Li)"与"最大重叠法"两节,
  并附集群实测四联图 (`book/figures/ch04_he_li_metastable.png`, 原始数据
  `book/data/he_li_metastable_pes.npz`); README/CHANGELOG 同步。

## 0.19.0 — QCT 集成、量子对比与教材全书编译

### Added

- **QCT 核心导出**: 在 `autoquantum/dynamics/__init__.py` 中正式公开导出 `QCTTrajectory`、`QCTEnsemble`、`QCTResult`、`wigner_sample`。
- **物理 Wigner 采样与反射早停**: `QCTEnsemble.run` 自动按双原子约化质量与基态振动宽度计算真实物理频率 $\omega = 1/(\mu_r \sigma_r^2)$；`QCTTrajectory.propagate` 增加 $R_{\rm refl\_threshold}$ 遇阻反射早期退出逻辑，大幅提升非反应轨线积分性能。
- **集群生产管线深度集成 QCT**: `scripts/cluster_scan.py` 与 `scripts/production.py` 增加 `--method {wavepacket, qct}` 与 `--n-traj` 支持，支持大规模 Slurm 阵列并行化轨线系综生产。
- **QM vs QCT 比较诊断框架**: 新增 `scripts/compare_qct_quantum.py`，实现同一势能面网格下量子波包动力学与准经典轨线动力的双向交叉基准，输出反应阈值、量子隧穿增强因子与玻尔对应极限定量对比表。
- **亚稳态体系与高级电子结构接口**:
  - `PySCFCalculator` 增加高自旋约束（ROHF/ROKS/UHF）、自旋纯度审计与 `spin_lock` 截断机制，消除如 $\text{He}^* + \text{Li}$ 四重态变分塌陷；
  - 增加最大重叠法（MOM，`use_mom=True`）轨道占据跟踪，避免扫描过程中激发态根翻转；
  - 增加复势能提取接口 `resonance_width` 与 `complex_energy`，支持自电离衰变宽度 $\Gamma(R)$；
  - `autoquantum sample` CLI 支持 `--spin`, `--method`, `--basis`, `--xc`, `--spin-lock`, `--spin-tol`, `--mom` 命令行参数。
- **教材新编两章与全量配图**:
  - 第 14 章《准经典轨线动力学 (QCT)》：哈密顿正则方程、辛积分、Wigner 采样、经典极限与量子隧穿对比。
  - 第 15 章《热速率常数与集群生产》：微观反应几率到宏观速率常数 $k(T)$ 的微正则/正则系综积分、Arrhenius 活化能拟合、Slurm 自动化生产管线。
  - 新增图 `ch14_qct_vs_quantum.png`、`ch15_thermal_rates.png`，配图总数扩充至 14 幅真实计算图。
- **教材编译升级**: 编译生成《从势能面到波包-分子反应动力学-v0.19.0.pdf》（134 页，15 章正文 + 4 附录，全面升级教学与高阶研发指引）。

### Fixed

- **Arrhenius 拟合自然对数转换**: 修正 `rates.py:plot_arrhenius` 中 $\log_{10} A$ 到自然对数的换算因子（$\ln A = \log_{10} A \times \ln 10$）。
- **`QCTResult.energy_drift_max` 类型转换**: 避免特定 numpy array 格式化触发的 `TypeError`。

## 0.18.0 — QCT 准经典轨线基础实现

### Added

- **`dynamics/qct.py`**: QCT 模块 — Velocity Verlet 经典轨迹在
  Born-Oppenheimer 势能面上的传播 (2D 共线 H+H₂); Wigner 分布采样
  振动态初始条件; `QCTEnsemble` 对每个碰撞能运行 N_traj 条轨迹。
- **力符号修复**: Velocity Verlet 中 F = −∂V/∂R (原实现误用 +∂V/∂R)。
- **`QCTEnsemble`**: 批量轨迹运行 → P_react(E), 与波包扫描接口兼容。
- **`wigner_sample`**: 谐振子基态 Wigner 分布采样 (正确的量子-经典对应)。
- **QCT vs 量子验证**: 高能端一致性 + 低能端 QCT 不应高于量子 (物理合理性)。
- 测试新增 5 项, 共 102 项。

### 物理预期

- 高能端 (E >> barrier): QCT ≈ 量子 (经典极限)。
- 低能端: QCT 可能低于量子 (缺隧穿) 或高于 (Wigner 采样含 classically
  forbidden 初始条件) — 两者均为 QCT 方法的已知系统偏差。

## 0.17.0 — 代码质量与文档完善 — 代码质量与文档完善

### Added

- **83 个中文 docstring** 补充至 17 个核心模块的公开函数和类
  (含物理单位 Hartree/Bohr/au)；覆盖率从 69% 提升到 ~90%。
- **`Makefile`**: 常用命令入口 (`make test/check/benchmark/book/clean/
  sync/submit`)。
- **`knowledge_base/README.md`**: 标注历史文档可能滞后。
- **`autoquantum/py.typed`**: 类型检查标记。

### Removed

- 过时 `scripts/sync_github.sh` (指向错误仓库且有 SSH 配置修改风险)。
- 旧 v0.9.1 PDF 从根目录清理 (最新版随 Release 分发)。
- `results/` 从 git 移除 (加入 `.gitignore`，通过管线重现)。

### Changed

- `.gitignore` 完善: `results/`、`book/build/`、`从势能面到波包*.pdf`。
- `production.py` 修复 3 个 bug (重复 --pes、grid_r 引用、numpy import)。

## 0.16.0 — 热速率常数与多势能面生产管线 — 热速率常数与多势能面生产管线

### Added

- **`analysis/rates.py`**: 热速率常数 k(T) — 从微观反应概率 P(E) 经
  Boltzmann 加权积分计算约化速率常数; Arrhenius 拟合 (Ea, log₁₀A, R²);
  Arrhenius 图生成。完成 "势能面 → NN 代理 → 散射 → P(E) → k(T) →
  Arrhenius" 全链路。
- **`cluster_scan.py --pes leps|eckart|morse`**: 多势能面通用接口,
  生产管线不再硬编码 LEPS。
- **`production.py --rates`**: 集群扫描完成后自动计算 k(T) 并生成
  Arrhenius 图。
- `analysis/__init__.py` 新子包。
- 测试新增 7 项 (Boltzmann 加权/阈值行为/温度单调性/Arrhenius 恢复),
  共 97 项。

### Fixed

- `ArrheniusFit.log_a` 统一为 log₁₀(A) (化学标准, 非自然对数)。
- 阶梯函数测试阈值修正 (截断能级/kBT = 0.48 时 P ≈ 0.28 物理正确)。

## 0.15.0 — 端到端集群生产管线 — 端到端集群生产管线

### Added

- **`scripts/production.py`**: 一条命令完成 配置→同步→Slurm 提交→
  监控→取回→合并→出图。支持 `--partition liquid_high/air`、
  `--torch` (GPU)、`--chunks N` (并行度)、`--max-wait` (0=只提交)。
- `scripts/remote.sh` 新增 `test` (服务器测试) 和 `run` (服务器命令)
  子命令; 修复 SSH 远端变量展开。
- `autoquantum/py.typed` 类型检查标记。
- `knowledge_base/README.md` (标注历史文档可能滞后)。
- 移除过时 `scripts/sync_github.sh` (指向错误仓库且有 SSH 配置修改
  风险); 旧 v0.9.1 PDF 从根目录清理。

### Changed

- `.gitignore` 新增 `results/`、`book/build/`、`从势能面到波包*.pdf`。
- 集群作业输出自动按 Job ID 命名; 结果通过 NFS 共享存储全集群可见。

## 0.14.0 — 教材第 13 章 + 符号修复 — 教材第 13 章 + 符号修复

### Added

- **第 13 章《对称函数势能面与主动学习》**: BP 对称函数（径向+角度）、
  共享原子能量委员会架构（置换不变按构造成立）、OOD 不确定性量化、
  主动学习闭环；9 节结构含新手路径/推导/代码对应/实验/陷阱/高手
  专栏/自测；所有 API 和数值对照源码验证（RMSE 1.4e-3, OOD 75×,
  置换不变 0.0, 池 RMSE 比 0.34）。
- 正文符号表新增 ≲/≳ 映射；代码块 Unicode 框线/箭头自动替换为
  ASCII 等价物。

### Fixed

- 教材编译器 `md2tex_book.py` 代码块中的 Unicode 框线字符 (─│┌┐└┘┤►▼)
  在等宽字体中缺字形 → 添加 `CODE_SYM` 替换映射。

## 0.13.0 — 主动学习闭环 — 主动学习闭环

委员会分歧驱动的采样-标注-再训练闭环 — 路线图"主动学习"项落地。

### Added

- **`nn/active_learning.py`**: `run_active_learning` — 每轮在已标注集
  训练共享原子能量委员会 → 对候选池计算标准差 (OOD 信号) → 挑选最
  不确定的 batch_size 个新点 → 用 Calculator 后端标注 → 加入训练集。
  记录每轮标注数/池 RMSE/std 统计/挑点索引; 池耗尽提前结束;
  池内真值仅作诊断, 委员会不可见。
- **测试**: 主动学习闭环降低池 RMSE (>30%)、挑点无重复、池耗尽停止,
  共 89 项。

### 说明

- 这是 D-optimality/GHOST 等采样策略的最简基座 (acquisition = 委员会
  标准差); 接入真实电子结构后端 (xtb/pyscf/ase) 即可用于生产采样。
- 梯度接力 (标注梯度 → 力训练) 是下一步。

## 0.12.0 — GPU 加速 (A100 实测)

### Added

- **`dynamics/wavepacket_2d_torch.py`**: 二维波包传播的 PyTorch 后端
  (API 兼容 NumPy 版子集), 支持 CUDA GPU 与 float32/float64; CAP 归因、
  通道记账、能量监测与 NumPy 版逐项一致。
- **A100-40GB 实测** (集群 liquid/air 节点, LEPS 势, 2500 步): 
  网格越大加速越明显 — 384×288: NumPy 6.4ms/步 → A100 fp32 1.4ms/步
  (4.6×); 768×576: 21.0 → 4.8ms/步 (4.4×)。
- **对拍验证** (tests/test_gpu_parity.py, 本地 CPU torch 也可跑):
  torch 后端与 NumPy 传播子的透射率/反射率/概率恒等式/⟨H⟩ 逐项一致
  (fp64 ~1e-6, fp32 ~1e-3), A100 上复测通过。
- 集群部署: `~/aqd-env` 已装 torch 2.5.1+cu121 (A100 = sm80)。
- 测试新增 3 项, 共 88 项。

### 说明

- 小网格 (192×144) GPU 加速有限 (~2×): FFT 为主的传播子在数据传输
  开销主导时收益缩水; 建议网格 ≥ 384×288 或批量扫描时使用。
- torch 为可选依赖, 未安装时所有现有路径不变。

## 0.11.0 — 对称函数势能面与委员会不确定性

补齐多原子 NN 势能面的两大短板: 置换对称性与数据外 (OOD) 不确定性
量化 — 这是走向四原子以上体系与主动学习的必经基础。

### Added

- **`nn/symmetry.py` 对称函数** (Behler-Parrinello 型, 逐原子中心,
  向量化): 径向 exp(−η[(r−r_c)²])f_c + 角度 2^{1−ζ}(1+λcosθ)f_c f_c
  exp(...); 严格平移/旋转不变, 原子置换只置换中心维。
- **`nn/ensemble.py` 共享原子能量委员会**: E = Σ_i E_atom(Φ_i) —
  原子置换按构造严格不变, 同种原子共享参数; n 个种子成员给出
  总能量标准差作为 OOD/主动学习信号 (`predict_with_uncertainty`,
  `std`, save/load 含归一化状态)。
- **`nn/optim.py`**: 共享 Adam 实现 (训练器与委员会共用)。
- **CLI `fit --symmetry`**: 数据集含 (n,3N) 笛卡尔坐标与 symbols
  即得置换不变委员会势能面; `--committee` 控制成员数。
- `AbInitioData` npz 支持 `symbols` 字段。
- 测试 +10 项 (对称性物理正确性/委员会学习/OOD/CLI), 共 85 项。

### Fixed (实现过程中暴露的真实缺陷)

- **近常数特征标准化爆炸**: 无邻居的角度项 std ~1e-10, 直接
  标准化为 ±1e10 → tanh 饱和、训练完全停滞 (委员会 RMSE 等于
  标签标准差)。标准化尺度加下限阈值 (委员会与主训练器)。
- **委员会归一化状态丢失**: 特征/能量标准化常数未随模型保存,
  训练损失收敛但预测 RMSE 高 3 个量级 — 与 v0.10 训练卡同源问题。
- **默认对称函数 η 网格不合理**: η≥4, r_c=5 时 1-3 bohr 键全部
  落在高斯尾部 (特征 ~1e-10, 死特征); 默认改为 (0.02, 0.2, 1.0)。
- 委员会保存避开 lambda 属性 (pickle 原始数组); `fit` 梯度
  (n,N,3)→(n,3N) 对齐; `fit --epochs` 参数统一。

## 0.10.0 — 工程硬化: 验证、可复现与 CI

把项目从"能跑的研究原型"推向科学计算软件标准: 输入验证、结果可信度
诊断、运行可复现凭证、持续集成与性能基线。

### Added

- **`core/validation.py` 输入验证层**: `validate_grid` (严格等距/有限/
  单调)、`validate_energy_window`、`validate_pes_values` (NaN/发散墙)、
  `validate_positive`; 失败抛 `ValidationError` 并附修复提示 (fail fast)。
- **传播健康诊断** `check_propagation`: 概率记账恒等式、通道和、末态
  未吸收比例 (传播不足/CAP 太弱)、NaN、能量漂移; engine 每次波包运行
  后自动执行 (`wp_health_check`, 默认开)。
- **能量监测**: `WavePacket2DPropagator.energy_expectation` + 
  `propagate(track_energy=True)` → `energy_track / energy_drift_rel`;
  自由演化守恒到 1e-8 (测试固定)。
- **运行清单** `run_manifest.json` (每次运行自动): 完整配置 + 环境快照
  (python/numpy/scipy/matplotlib/autoquantum/git commit) + 结果摘要 +
  PES 数据 SHA-256 指纹 — 结果可复现的凭证。
- **训练卡**: NN 训练自动记录数据指纹/规模/RMSE/超参/环境, 随 `.pkl`
  持久化 (`PESNN.training_card`), 模型文件自解释训练来源。
- **`suggest_dt`**: c/E_max 步长相位精度建议, engine 在 dt 过大时告警。
- **实验模块护栏**: `allow_experimental=False` 时显式拒绝 2D 定态求解器。
- **CI** (`.github/workflows/ci.yml`): ubuntu+macos × py3.11-3.13 测试
  矩阵 + 导入冒烟 + 配图测试 + wheel 构建; TeX Live 教材构建作业。
- **`scripts/check.sh`** 本地质量闸门; **`scripts/benchmark.py`** 可复现
  微基准 (参考: 96×64 网格 ~2800 波包步/秒, NN 每轮 ~34 ms, Apple M4)。
- 测试新增 19 项, 共 75 项。

### Fixed

- `write_run_manifest` 对 Python 3.14 局部类的反射兼容 (3.14 下
  `isinstance(C, type)` 与 `bool(C.__dict__)` 均不可靠)。
- `PESNN` 重复包装护栏 (CLI 双重包装导致 save 报错)。

## 0.9.1 — 教材正式定名与 PDF 出版

- **书名确定**: 《从势能面到波包——分子反应动力学的量子理论、数值验证与代码实践》(AutoQuantum 项目组, v0.9.1); 书名落实于封面、索引与前言扉页。
- **教材构建管线** `scripts/md2tex_book.py`: Markdown → ctexbook (xelatex), 零额外 Python 依赖 (系统 TeX Live 即可)。处理标题/嵌套列表/数学感知分列的 GFM 表格/代码块/图片/折叠答案/行内行间公式 (含 \tag)/链接/中英符号; 大表按估算行高分块 (续表) 保证断页与可读字号; 初态-CAP 等既有代码不变。
- **PDF 版教材** (105 页, 封面+可点击目录+页眉页脚+页码, 12 张真实计算配图) 输出至仓库根目录。
- 独立视觉验收两轮: 修复了页眉/页脚命令被 f-string 破坏、封面字号参数丢失、代码块环境丢失、表格因 `|psi|` 误分列与不可断页导致的 38 行术语丢失、1.2pt 不可读字号、长路径越界等阻断项; 终版 0 错误/0 缺字形/0 空白页/0 越界。

## 0.9.0 — 教材: 动力学基础章与真实计算配图

回应两条反馈: 动力学缺少基础章节、教材无图。

### Added

- **第 06 章《分子振动与能量转移》**: 核振动 Hamiltonian 与质量加权
  坐标、谐振子本征值与零点运动、Morse 非谐性能级与有限束缚态数、
  ZPE 为何不属于电子力 (与第 04 章衔接)、光谱间隔/冷分子基态占据、
  V→T/V→V 能量转移、束缚态到散射态的衔接。
- **`scripts/make_book_figures.py` + `book/figures/` (12 张图)**:
  全部由真实代码计算生成 (Morse 能级、谐振子本征函数、ZPE 阈值、
  sech² 垒 T(E)、波包演化、通道记账、二维快照、PES 地形、NN 力
  训练对比、多宽度反卷积、BSSE 示意); 可一条命令复现
  (`python scripts/make_book_figures.py`); 概念图在标题中标注
  schematic; 图中轴标签为英文, 中文图注在正文。
- 图渲染冒烟测试 (`tests/test_book_figures.py`): 真实计算 + 落盘 +
  物理质量护栏 + 配图齐全性 — 共 56 项测试。

### Fixed (配图独立审阅发现的 5 处物理错误)

1. 谐振子本征函数: Legendre/概率型 Hermite → **物理学家 Hermite**
   与 exp(−x²/2) 配套;
2. ZPE 热占据: 温度范围 0.1-2.0 K 误标为千 K 级、v=0 占据随温度
   **上升** → 50-3000 K、物理方向单调下降;
3. LEPS 势能面图: 键长坐标 (r,R,r+R) 却标为 Jacobi 并叠加 R=1.5r
   分界面 → 改用真正的 `leps_jacobi_pes` 包装, 势垒标注改为沿分界面
   实测值 ~0.13 au;
4. NN 力训练图: 标题宣称"能量与梯度同时提升"与自身数据矛盾
   (某次运行梯度变差) → 标题改为数据驱动, 与回归测试同配置, 改善
   不复现时生成即告警;
5. Morse 能级图: 画的是谐振值却标为 Morse 能级 → 改用精确 Morse
   公式 E_v = ħω(v+1/2) − [ħω(v+1/2)]²/4D_e;
6. 术语: sech² 垒在图题中诚实标为"eckart.py 的实现模型"而非严格的
   Eckart 势; 2D 快照补密度色标与"有限时间通道概率"说明。

### Changed

- 章节再编排: 06-11 → 07-12 (为分子振动章让位), 交叉引用与索引
  同步; 图注与图内标题术语对齐。

## 0.8.1 — 教材: 电子结构基础章与章节重编排

### Added

- **第 04 章《电子结构基础》** (教材新增, 回应"基组等内容缺失"的反馈):
  Born-Oppenheimer 三层近似、Slater 行列式与变分原理、高斯基函数/
  收缩/弥散/BSSE、Hartree-Fock 与 Roothaan-Hall 方程、SCF 不动点、
  DFT (Hohenberg–Konn/Kohn-Sham/LDA-GGA)、GFN-xTB 半经验方法、
  Hellmann-Feynman 与 Pulay 力、核排斥能与零点能的边界; 并把
  `basis/charge/spin/uhf/method` 等**每个后端参数**逐一映射到物理含义
  与 `pes/calculators.py` 的真实 API。
- 附录 A 新增 A.3 电子结构术语补充 (22 条术语 + 与代码的对应)。
- 前言学习路线 A 加入电子结构选读; 符号表补充核力/梯度约定。

### Changed

- **教材章节重编排**: 新第 4 章插在势能面与 NN 之间 (知识顺序:
  量子力学 → 电子结构 → 势能面 → ML → 动力学), 原 04-10 章顺延
  为 05-11, 全书交叉引用 (标题式与"第 N 章"式) 与索引表同步更新。
- 本版仅文档/教材变更, 代码与数值验证结论不变 (v0.8.0 的 53 项
  测试与诚实边界继续有效)。

## 0.8.0 — 电子结构后端与训练数据生成 (实验性)

完成"从头算 → 拟合 → 动力学"闭环的数据入口: 从电子结构程序
采集 (能量, 梯度) 训练集, 训练力训练代理面。

### Added

- **`pes/calculators.py` 统一后端协议** `Calculator` (能量 Hartree +
  核梯度 Hartree/Bohr, 与动力学模块同一单位制):
  - `AnalyticCalculator` (含中心差分回退) 与内置 `demo_calculator`
    (LJ 演示, 明确非量子化学; 让整条 CLI 闭环在零依赖环境可验证);
  - `XTBCommandCalculator` (子进程 GFN-xTB `--grad` + 正则解析),
    `PySCFCalculator` (RHF/UHF/ROHF/KS), `ASECalculatorAdapter`
    (任意 ASE calculator, ∇E = -F) — 均为**实验性**;
  - `available_calculators()` 可用性报告, `make_calculator()` 工厂;
    provenance (程序/版本/基组/电荷/自旋) 随数据集保存。
- **几何采样** `AbInitioData.sample_geometries`: 笛卡尔位移网格
  (active_atoms × axes, ranges 单元素广播, 最小原子间距过滤,
  max_points 子采样) → (points, energies, gradients) + provenance;
  `save_npz`/`load_npz` 数据集持久化。
- **CLI**: `autoquantum backends` (后端可用性), `autoquantum sample`
  (XYZ 参考几何 → 数据集), `autoquantum fit` (数据集 → NN 代理面,
  报告能量/梯度 RMSE)。
- `nn_pes_2d` 提升为公共 API: 训练好的 2D 代理面可直接喂给
  `WavePacket2DPropagator` (engine 内部改用同一包装器)。
- 测试: 后端协议/FD 梯度/采样器/CLI 闭环 — 共 53 项。

### 诚实边界

- 真实量子化学后端 (xtb/pyscf/ase) 在本仓库发布验证环境**未安装、
  未经端到端验证**; 其数值正确性由所调程序与设置决定, 本仓库不做
  认证。可验证部分是协议、采样管线与 demo 闭环 (演示实测: 能量
  RMSE 0.32% span, 梯度 RMSE 2.4×10⁻³ Hartree/Bohr)。
- 采样特征为**展平笛卡尔坐标** (n, 3N): 无置换/旋转等变性, 同核
  体系需自行扩展 (见 3.6 节与第 10 章)。
- 采样为均匀位移网格, 非自适应/主动学习 (接口位置已预留)。

## 0.7.0 — 能量反卷积、收敛检查与配套教材

### Added

- **多宽度扫描** `multi_width_scan`: 多个包宽度重复扫描, 输出能量平均
  响应矩阵 (每个扫描点的波包以其能量为中心, p₀ = -√(2μ_R·E))。
- **能量反卷积** `deconvolve_reaction`: 分段线性基 + 曲率正则化
  (D₂ᵀD₂) 的增广最小二乘, 从多宽度数据恢复 T(E)。实现要点:
  **不能走正规方程** —— AᵀA 会把条件数平方 (cond(A)~1e9 时触及
  float64 噪声底并产生 0/1 振荡), 必须对增广系统 [A; √λ·D₂] 做
  SVD 最小二乘。返回 (E, T, cond(A), 相对残差), 两个诊断量各司其职:
  条件数刻画病态性, 残差刻画数据与包络模型的失配 (如有限传播时间
  导致的低能截断)。
- **传播收敛检查** `check_convergence`: 基线 vs 加密 (网格/时间步/
  时长) 对比, 返回 |ΔP_react|。
- engine/CLI: `--wp-deconvolve` / `--wp-convergence`; 收敛偏差
  >0.02 或反卷积残差 >0.1 时告警; 反卷积曲线进入 dashboard
  (`wavepacket_deconv.png`)。
- **教材** `book/`: 10 章正文 + 前言 + 2 个附录 (~134KB, 面向新手
  与高手双层结构), 全部 API 与基准数值对照源码核验。
- 测试: 能量包络归一化、解析阈值合成数据恢复 (RMSD<0.05)、病态
  与失配诊断、收敛检查一致性 — 共 41 项。

### 实测行为 (诚实的病态性)

- 宽能窗 (0.005-0.045) + 传播充分 (6000 步): cond(A)=1.3e3,
  残差 0.019, 恢复的 T(E) 与 1D 解析参考平均差 ~0.11 (误差集中在
  阈值跳变处)。
- 窄窗 (0.025-0.045): cond(A)=1.9e9, 残差 0.38 — 曲线无意义,
  被诊断量正确标记。
- engine CLI 冒烟: 1200 步设置下收敛检查报告 ΔP=0.25 (未收敛
  告警), 反卷积残差 0.17 (告警) — 两个工具都按设计工作。

## 0.6.0 — 力训练 (double backprop)

NN 拟合以 ∂V/∂x 为监督目标, 显著提升代理面的能量与梯度保真度。

### Added

- **力训练**: `NNTrainer.train(X, y, dY=..., force_weight=...)` —
  联合损失 `MSE_V + w·MSE_F`; `dY (n, d)` 为 ∂V/∂x 监督目标
  (标准化空间自动换算)。
- **`FeedForwardNN.input_gradient_backward`**: 反向的反向
  (reverse-over-reverse double backprop), 计算 Σ adjG·(输入梯度) 对
  全部权重/偏置的解析梯度 — 经有限差分校验 (误差 ~10⁻¹⁰, 覆盖
  tanh/sigmoid/relu)。注意 pass-1 映射 T = G·Wᵀ 的转置方向与
  前向 backprop 相反 (dW = barTᵀ·G)。
- **二阶激活导数**: tanh/sigmoid/relu 的 f″ (力训练经 f″ 项把
  梯度依赖传入偏置)。
- **engine 2D 拟合默认启用力训练** (`nn_force_weight=1.0`, CLI
  `--nn-force-weight`): 训练数据由 `sample_function` 生成 (含中心
  差分梯度), 日志与 summary 输出能量 RMSE 与梯度 RMSE 双指标。

### 效果 (Eckart 2D, [40,40], Adam lr=0.005, 1200 轮, 固定种子)

| 配置 | V_RMSE (au) | 梯度 RMSE (au/Bohr) |
|---|---|---|
| 纯能量拟合 (w=0) | 4.6×10⁻³ | 2.2×10⁻² |
| 力训练 (w=0.5) | 1.1×10⁻³ | 4.4×10⁻³ |
| 力训练 (w=2.0) | 4.6×10⁻⁴ | 2.3×10⁻³ |

CLI 实测 (w=1.0, 800 轮): V_RMSE 4.5×10⁻⁴ au (0.03% V span)。
力目标同时正则化能量拟合 — 两者同步改善。

### Changed

- `PipelineConfig.nn_force_weight` 默认 1.0 (2D), CLI 对 1D 体系
  默认 0; 朴素梯度下降路径 (adam_beta1=0) 不支持力训练, 会输出告警。

### 验证与限制

- 36 项测试通过 (新增: double-backprop FD 门控、力训练提升梯度
  精度断言)。
- 力训练目前依赖解析/差分梯度来源; 从头算梯度 (ASE/PySCF) 仍属
  路线图。能量反卷积与收敛自动化未在本版。

## 0.5.0 — 多维 NN 势能面拟合与数据驱动管线

补齐 "任意 1D/2D PES → NN 代理 → 动力学" 的数据驱动闭环 (从头算
后端仍缺失, 见限制)。

### Added

- **多维 NN 拟合**: `FeedForwardNN`/`NNTrainer` 支持任意维输入
  (n, d); 二维 PES 网格 (R, r, V) 可直接训练。
- **输入/输出标准化**: 训练划分上拟合、随模型 save/load 持久化;
  `predict`/`gradient` 接受物理单位数据 (兼容无归一化的旧模型文件)。
- **Adam 优化器**: 训练器内置 (β₁=0.9, β₂=0.999); 早停改用相对
  改进判据 (1e-3) 并放大默认 patience 至 100 — 修复全批量训练
  慢收敛尾部被过早截断的问题。
- **解析输入梯度** `gradient(X)`: dŷ/dX 经反标准化链式修正, 与
  中心差分校验一致 (rtol 1e-4) — 为后续力训练提供基础。
- **`AbInitioData.sample_function`**: 从任意可调用势能函数在规则
  网格上生成带中心差分梯度的训练集 (数据生成接口, 非电子结构计算)。
- **2D 数据驱动管线**: engine 对二维体系先拟合 NN 代理面 (RMSE
  日志 + >2%·V_span 告警), 波包动力学运行在 NN 面上; `nn_lr`
  默认调整为 0.005 (Adam), 新增 `nn_max_train_points` 子采样。
- 测试: 多维拟合精度、梯度 FD 校验、归一化持久化、函数采样器
  梯度对齐、2D NN-PES 管线集成 — 共 34 项。

### Changed

- `PipelineConfig.nn_lr` 0.001 → 0.005 (Adam); `NNTrainer` 早停
  语义见上。1D 流程同样受益 (Morse 面拟合 loss ~5e-6)。

### 验证

- 34 项测试通过; 2D NN 拟合 Eckart 面 RMSE ≈ 3×10⁻³ au (0.26% V
  span); NN 面上的波包传播满足全部守恒恒等式。
- **注意**: NN 代理面上的反应概率继承了拟合误差, 精度以日志中的
  RMSE 为准; 力训练、能量反卷积、从头算后端仍属路线图。

## 0.4.0 — 二维含时波包传播

本版聚焦路线图首要项 "二维含时波包传播", 并修复独立代码评审确认的
全部数值缺陷。**发布范围: 一维定态/含时与二维含时路径; 二维定态
求解器标注为实验性; 从头算后端仍缺失。**

### Added

- `dynamics/wavepacket_2d.py` — 二维含时波包:
  - Split-Operator (FFT) 复势对称分裂传播子 (Strang 二阶), 支持
    非对角动能 g 矩阵 (共线键长坐标);
  - 网格边缘二次型 CAP, 吸收量按格点确定性归因 (角点归 η 最大的
    边), 恒等式 `P_grid + Σ L_edge = 1` 精确成立;
  - 通道分析: product/reactant 掩码互斥校验, `P_react + P_refl = 1`
    当掩码划分网格时精确成立;
  - `WavePacket2DScan` 碰撞能扫描 (E = p²/2μ), 兼容 `ScatteringResult2D`;
  - 初态-CAP 重叠自检 `check_initial_overlap`;
  - 基态宽度辅助: 谐振子精确 `1/√(μω)`, Morse 谐振近似。
- `visualization/wavepacket2d.py` — 波包快照 / 通道概率 / GIF 动画,
  dashboard 与 HTML 报告集成。
- `examples/h3_wavepacket.py` — Eckart (默认) / LEPS 波包示例。
- engine `method` 配置 (`auto`/`stationary`/`wavepacket`; 2D auto → 波包),
  CLI `-m/--method`、`--wp-points/--wp-steps/--no-gif`。
- engine 初态-CAP 重叠检查 (>1e-4 报错, >1e-5 告警)。
- 测试: 1D/2D 波包幺正性、CAP 守恒、角点归因确定性、初态重叠、
  LEPS 阈值行为、engine 端到端集成、NN 早停恢复 — 共 26 项。

### Fixed

- **高斯宽度约定**: 初始化用振幅宽度 σ (|ψ|² ∝ exp(-(Δx/σ)²)),
  基态宽度辅助函数原返回位置标准差 1/√(2μω) (差 √2, 制备挤压激发态),
  修正为振幅宽度 1/√(μω)。
- **分裂对称性**: 重构中 CAP 曾被整体附加在步末 (破坏二阶精度),
  改为复势 W = V − iη 对半分裂。
- **1D 波包通道标签**: 波包自右向左传播时透射/反射方向标反, 修正为
  透射=左侧区域+左侧吸收。
- **`quantum_2d.py` 递推符号**: 中心差分应为 `(2 − h²k²)` 而非
  `(2 + h²k²)`; 去除禁阻区钳位 (允许虚指数); 去除对结果的 [0,1]
  裁剪 (不再掩盖无效输出); 模块标注实验性并加引擎告警。
- **engine 能量窗口**: 越界截断不再静默 — 记录日志并保证升序
  (原先可能产生倒序区间)。
- **Eckart 入口态**: 波包中心对准耦合致移的入口通道振动平衡位置
  (原固定在 r0)。
- **`wavepacket.py` (1D)**: 复数初始化类型错误 (原模块完全不可运行)、
  numpy 2.x `np.trapz` 兼容、保存调度 off-by-one、新增可选 CAP。
- **NN**: 输出层线性化使 PES 拟合不受 tanh 值域限制, 反向传播即
  0.5·MSE 的精确梯度 (0.3.1 已发布, 本分支包含); 早停恢复验证损失
  最低的权重。
- 示例脚本 `sys.path` 少一层目录导致直接运行失败。

### Changed

- `requirements.txt` 移除未使用的 torch; 版本号统一 0.4.0。
- engine 二维体系默认走含时波包; 时间无关路径需显式 `-m stationary`。

### 验证与限制

见 [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md)。要点: 26 项测试
通过; 1D Eckart 波包 vs 解析谱 <4×10⁻⁴; 可分离 2D vs 能量加权参考
差 0.002/0.004。**LEPS 教学参数的阈值、2D 定态输出、从头算功能
均不在本版验证范围内。**

## 0.3.1 — NN 回归输出修复

- 隐藏层保留激活函数, 输出层线性化; PES 预测不再受 tanh 值域限制。
- 版本号统一 (原 setup.py 0.2.0 与 `__init__` 0.3.0 不一致)。
- 回归测试: 线性输出 / 超范围拟合 / 有限差分梯度。
- 注意: 旧权重加载后预测改变, 需重新验证或重训。
