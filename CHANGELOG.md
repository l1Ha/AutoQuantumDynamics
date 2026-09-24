# Changelog

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
