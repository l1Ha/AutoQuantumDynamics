# Changelog

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
