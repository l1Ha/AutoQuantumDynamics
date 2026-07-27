# AutoQuantum

**分子反应动力学全维量子动力学计算自动实现平台**

从势能面构建 → 神经网络拟合 → 量子动力学计算 → 结果可视化的全自动流程。

## 快速安装

```bash
pip install -r requirements.txt
pip install -e .
```

## CLI 使用

```bash
# 一维 H₂ 散射 (Morse 势)
autoquantum run -s H2_1D -p morse -o output_h2

# 二次散射
autoquantum run -s H2_1D -p harmonic -o output_h2_harm

# 二维 H + H₂ 反应 (Eckart 垒)
autoquantum run -s H3_2D -p eckart -o output_h3

# 查看系统信息
autoquantum info
```

## 示例

```bash
# 一维 H₂ Morse 势散射
python autoquantum/examples/h1d_scattering.py

# 二维 H + H₂ 反应散射 (Eckart 垒 + 最小能量路径)
python autoquantum/examples/h3_reaction.py
```

## 项目结构

```
AutoQuantum/
├── autoquantum/
│   ├── cli.py               # 命令行接口
│   ├── core/
│   │   ├── engine.py         # AutoPipeline 自动编排引擎
│   │   └── base.py           # 基类
│   ├── pes/
│   │   ├── analytic.py       # Morse / Harmonic / LJ
│   │   ├── builder.py        # PES 构建器
│   │   ├── abinitio.py       # 从头算数据接口
│   │   ├── leps.py           # LEPS (H+H₂)
│   │   └── eckart.py         # Eckart 垒 (H+H₂)
│   ├── nn/
│   │   ├── model.py          # 前馈神经网络 (纯 NumPy)
│   │   ├── train.py          # 训练管线
│   │   └── dataset.py        # 数据集
│   ├── dynamics/
│   │   ├── quantum_1d.py     # 1D 量子散射 (Numerov)
│   │   ├── quantum_2d.py     # 2D 量子反应框架
│   │   ├── wavepacket.py     # 含时波包 (Split-Operator)
│   │   └── observables.py    # 可观测量
│   ├── visualization/
│   │   ├── plot.py           # 1D 图
│   │   ├── contour.py        # 2D 等高线 + 3D
│   │   └── dashboard.py      # HTML 报告
│   └── examples/
│       ├── h1d_scattering.py # H₂ 一维散射
│       └── h3_reaction.py    # H+H₂ 反应散射
├── knowledge_base/           # 理论文档
├── tests/                    # 单元测试
└── setup.py
```

## 流程说明

```
[PES 构建] → [NN 拟合] → [动力学计算] → [可视化]
```

### 支持体系

| 体系 | PES | 动力学 | 说明 |
|------|-----|--------|------|
| H₂ 1D | Morse | 透射/反射 | 双原子振动散射 |
| H₂ 1D | Harmonic | 透射/反射 | 谐振近似 |
| H+H₂ 2D | LEPS | MEP 反应概率 | 三原子反应 |
| H+H₂ 2D | Eckart | MEP 反应概率 | 简化反应模型 |
| 任意 1D | NN 拟合 | 透射/反射 | 数据驱动 |

## 路线图

- [x] 一维 H₂ 量子散射 (Morse / Harmonic)
- [x] 二维 H + H₂ 反应散射 (Eckart / LEPS + MEP)
- [x] 神经网络 PES 拟合
- [x] CLI 命令行接口
- [x] HTML 报告生成
- [ ] 二维含时波包传播
- [ ] 四原子以上复杂体系
- [ ] GPU 加速
- [ ] ASE/PySCF 自动数据生成
