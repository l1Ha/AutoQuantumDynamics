# v0.4.0 release validation

## Scope

二维含时波包传播 (新)、一维定态/含时、NN 拟合修复、CLI/引擎重构。
二维定态求解器为实验性 (引擎告警), 不在验证承诺范围内; 从头算后端
仍缺失。LEPS 教学参数的阈值是模型量, 非真实 H+H₂ 物理量。

## 测试

- 全量 26 项测试通过 (macOS arm64, Python 3.14.5, NumPy 2.4.4)。
  开发环境首跑 310 s (含 matplotlib 冷启动); 隔离 wheel 环境 41 s。
- 覆盖: 1D/2D 波包幺正性与平移、CAP 吸收守恒、角点归因确定性、
  初态-CAP 重叠、可分离体系能量加权参考、LEPS 阈值行为、
  engine 端到端集成 (含 GIF)、NN 梯度有限差分/线性输出/早停恢复。

## 数值基准 (固定参数, 预设判据)

| 基准 | 结果 |
|---|---|
| 1D Eckart 波包 vs 解析谱透射率加权平均 (E∈{0.5,1,2}, 网格/时长收敛) | 误差 < 3.7×10⁻⁴; 细化后变化 < 2×10⁻⁵ |
| 可分离 2D Eckart (R 4096×r 32, dt=0.02, t=160, E=1) vs 同判据解析解 | 误差 1.24×10⁻⁵ |
| 可分离 2D vs 能量加权 1D 参考, E_coll=0.02 (阈值区内) | 0.7451 vs 0.7471 (差 0.002) |
| 可分离 2D vs 能量加权 1D 参考, E_coll=0.04 (阈值上方) | 0.9955 vs 参考 (测试容差 0.03 内) |
| 谐振子基态静止性 (126 步, 一个振荡周期的 1/10) | 密度漂移 3.4×10⁻⁶ |
| 概率恒等式 P_grid + Σ L_edge = 1 (全部含 CAP 算例) | ~10⁻⁹ |
| 通道和 P_react + P_refl = 1 (互补掩码算例) | ~10⁻⁹ |

仲裁记录: E=0.02 处波包结果 (0.745) 与单能量 1D 点值 (0.9835) 差异
曾达 0.24; 经能量分解仲裁 (波包动量分布 × 细网格逐能量 T(E) 加权
积分 = 0.7471) 确认波包正确 — 该参数化阈值很宽, 点值与包络平均本
质不同。测试据此采用能量加权参考 (test_wavepacket_2d.TestEckartSeparable)。

## CLI 冒烟 (开发环境)

- `H2_1D -p morse` (auto→stationary): 完成, max T = 0.9683。
- `H3_2D -p eckart` (auto→wavepacket): 完成, 扫描 max 0.9717。
- `H3_2D -p leps -m wavepacket`: 完成, 扫描 max 0.7692 (E=0.25 au)。
- `H3_2D -m stationary` (实验性): 完成并输出实验性告警。
- `examples/h3_wavepacket.py --pes eckart`: 完成, 恒等式 1.0000000000。

## wheel 隔离验证

全新 venv (无系统 site-packages) 安装 `autoquantum-0.4.0-py3-none-any.whl`:

- 运行时版本与发行元数据均为 0.4.0;
- 26 项测试通过 (挂载源码 tests 目录);
- `autoquantum --version` / `info` 正常; `pip check` 无损坏依赖;
- 依赖实测: NumPy 2.5.3, Matplotlib 3.11.2, SciPy 1.18.1。

Wheel SHA-256:

```text
7bf09d8d62cfb418112b62ec325f9362759b6738b70d81fa3bfc0a02b8481c96
```

## 不在本版验证/修复范围内

- 2D 定态求解器输出 (实验性, 仅供定性参考);
- LEPS 教学参数下的势垒/阈值/反应概率的物理真实性;
- 从头算 PES 生成、多维 NN 拟合、力训练、能量反卷积;
- 跨平台/多 Python 版本矩阵; pickle 模型向后兼容性 (0.3.1 线性输出
  层改变旧模型预测, 旧模型需重训)。
