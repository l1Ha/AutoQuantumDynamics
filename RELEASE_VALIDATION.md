# v0.8.0 release validation

## Scope

电子结构后端协议与实现、几何采样与数据集持久化、`sample`/`fit`
CLI 闭环、`nn_pes_2d` 公共 API, 以及教材第 3.6 节。v0.4.0-v0.7.0
的验证结论继续有效。

## 诚实边界 (本版最重要的一条)

**真实量子化学后端在本仓库发布验证环境中未安装、未经端到端验证**:
发布机器上无 `xtb` 可执行文件, 无 pyscf / ase / geometric 包。
因此本版验证的是: (a) 后端协议的实现正确性 (通过 Analytic 与
demo-LJ 后端); (b) 采样管线的几何/滤波/持久化正确性; (c) CLI
闭环端到端可运行。`XTBCommandCalculator` / `PySCFCalculator` /
`ASECalculatorAdapter` 的数值正确性由所调外部程序与参数决定,
本仓库不做认证, 首次使用前应自行做基准校验。

## 已验证项

| 项 | 结果 |
|---|---|
| demo-LJ 解析梯度 vs 中心差分 | rtol 1e-5 内一致 |
| `AnalyticCalculator` FD 回退 vs 解析梯度 | rtol 1e-5 内一致 |
| 采样器: 形状 (n,3N) 特征 + (n,N,3) 几何梯度、ranges 广播、最小距离过滤、max_points 子采样 | 单元测试通过 |
| npz 往返 (points/energies/gradients/provenance/geometry) | 逐数组一致 |
| CLI sample→fit 闭环 (demo 后端, 216 点, 力训练 1500 轮) | 能量 RMSE 9.2e-4 Hartree (0.32% span), 梯度 RMSE 2.4e-3 Hartree/Bohr |
| 缺失后端的报错路径 (`xtb` 未安装) | 抛出含安装指引的 `CommandBackendError` |
| `available_calculators()` | 正确报告 analytic/demo 可用, xtb/pyscf/ase 不可用, 不抛异常 |

## 局限 (已写入 README 与教材 3.6)

- 采样特征为展平笛卡尔坐标: 无置换/旋转等变性 (同核体系需对称性
  扩展, 见第 10 章);
- 采样为均匀位移网格, 非主动学习 (接口位置已预留);
- 后端 provenance (程序/版本/基组/电荷/自旋) 随数据集保存, 是
  可复现性的前提。

## 测试与 wheel 验证

- 全量 53 项测试通过 (macOS arm64, Python 3.14.5), 65 s。
- 全新 venv 安装 `autoquantum-0.8.0-py3-none-any.whl`: 版本元数据
  0.8.0; 53 项测试通过; 隔离环境内 `sample`/`fit`/`backends` CLI
  闭环通过; `pip check` 干净。

Wheel SHA-256:

```text
dd1092aaad1b5109697eba4d3fabb46da13ab57605668fe5f84607b89864fcf7
```

## 遗留

主动学习采样与对称性特征; 四原子以上体系; GPU 加速; 真实后端的
基准验证套件 (待环境具备后补充)。
