# AutoQuantum

**分子反应动力学量子计算研究原型** — 势能面构建 → 神经网络拟合 → 量子动力学 → 可视化的自动化流程。

> ⚠️ 这是研究/教学原型，不是经过认证的科学计算软件。定量使用前请阅读
> [功能边界与已知限制](#功能边界与已知限制) 与 [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md)。

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

- **从头算 PES 未实现**: `AbInitioData` 为数据容器 + 可调用函数采样器
  (`sample_function`, 带中心差分梯度), 无 ASE/PySCF 后端。
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
- [x] CLI 与 HTML 报告
- [ ] 波包能量反卷积与收敛自动化
- [ ] 四原子以上复杂体系
- [ ] GPU 加速
- [ ] ASE/PySCF 自动数据生成 (含从头算梯度)
