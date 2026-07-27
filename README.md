# AutoQuantum

**分子反应动力学全维量子动力学计算自动实现平台**

从势能面构建 → 神经网络拟合 → 量子动力学计算 → 结果可视化的全自动流程。

## 项目结构

```
AutoQuantum/
├── autoquantum/           # 核心代码
│   ├── core/              # 流程引擎 & 基类
│   │   └── engine.py      # AutoPipeline 自动编排引擎
│   ├── pes/               # 势能面构建
│   │   ├── analytic.py    # 解析势函数 (Morse, Harmonic, LJ)
│   │   ├── builder.py     # PES 构建器
│   │   └── abinitio.py    # 从头算数据接口
│   ├── nn/                # 神经网络拟合
│   │   ├── model.py       # FFNN 前馈神经网络
│   │   ├── train.py       # 训练管线
│   │   └── dataset.py     # 数据集
│   ├── dynamics/          # 量子动力学
│   │   ├── quantum_1d.py  # 一维量子散射 (Numerov)
│   │   ├── wavepacket.py  # 含时波包传播 (Split-Operator)
│   │   └── observables.py # 可观测量
│   ├── visualization/     # 可视化
│   │   ├── plot.py        # 绘图工具
│   │   └── dashboard.py   # HTML 报告生成
│   └── examples/          # 示例
│       └── h1d_scattering.py  # H₂ 一维散射
├── knowledge_base/        # 知识库文档
├── tests/                 # 单元测试
└── setup.py
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
pip install -e .
```

### 运行一维 H₂ 散射示例

```bash
python autoquantum/examples/h1d_scattering.py
```

输出在 `output_h2_1d/` 目录:
- `pes_original.png` — 原始势能面
- `pes_nn_fit.png` — 神经网络拟合结果对比
- `transmission.png` — 透射/反射概率
- `report.html` — 完整报告

### 编程使用

```python
from autoquantum.core.engine import AutoPipeline, PipelineConfig

config = PipelineConfig(system_name="H2_1D")
pipeline = AutoPipeline(config)
results = pipeline.run()
print(pipeline.summary())
```

## 流程说明

```
[PES 构建] → [NN 拟合] → [动力学计算] → [可视化]
```

1. **PES 构建**: 生成势能面数据 (解析或从头算)
2. **NN 拟合**: 神经网络学习 PES 数据点
3. **动力学计算**: 求解薛定谔方程获得透射/反射系数
4. **可视化**: 生成图表和 HTML 报告

## 路线图

- [x] 一维 H₂ 量子散射 (Morse 势)
- [ ] 二维 H + H₂ 反应散射
- [ ] 全维三原子反应 (Jacobi 坐标)
- [ ] 四原子以上复杂体系
- [ ] GPU 加速动力学计算
- [ ] 自动从头算数据生成接口 (ASE/PySCF)
- [ ] AIMD 数据导入
