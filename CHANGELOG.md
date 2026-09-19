# Changelog

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
