# v0.5.0 release validation

## Scope

多维 NN 势能面拟合 (新)、NN 代理面直通波包动力学、Adam 优化器、
解析梯度接口、函数采样器。v0.4.0 的波包验证结论继续有效; 从头算
后端与力训练仍缺失 (路线图)。

## 测试

- 全量 34 项测试通过 (macOS arm64, Python 3.14.5, NumPy 2.4.4);
  隔离 wheel 环境 44 s。
- 新增覆盖: 多维拟合精度 (RMSE < 2e-2)、解析梯度 vs 中心差分
  (rtol 1e-4, 含归一化链式路径)、归一化 save/load 逐点一致、
  `sample_function` 梯度对齐解析解、2D NN-PES 管线集成
  (RMSE < 0.01 au + 守恒恒等式)。

## 数值结果 (固定种子, 可复现)

| 项 | 结果 |
|---|---|
| 2D NN 拟合 Eckart 面 (150×150 网格→3000 子采样, [40,40], Adam lr=0.005, 1200 轮) | RMSE ≈ 3.3×10⁻³ au (0.26% V span) |
| NN 代理面上波包传播 | 全部守恒恒等式 ~10⁻⁹ |
| 1D Morse 面 NN 拟合 (engine CLI) | final MSE ≈ 5.6×10⁻⁶ |
| 多维测试函数 (sin·exp + 线性项, 2000 轮) | RMSE ≈ 1.3×10⁻² |
| 解析梯度 vs 中心差分 (12 点 × 2 维) | rtol 1e-4 内一致 |
| `sample_function` 梯度 vs 解析 (1D/2D) | rtol 1e-4 内一致 |

优化器记录: 朴素全批量梯度下降在慢收敛尾部被 patience=50 早停截断
(平台 1.4e-2); 换 Adam + 相对改进判据 (1e-3) + patience=100 后,
同结构同轮数 RMSE 降至 8×10⁻³–3×10⁻³ 量级。

## CLI 冒烟 (开发环境)

- `H2_1D -p morse` (NN 拟合 + 定态扫描): 完成, max T = 0.9684
  (与解析面 0.9683 一致 — NN 面未引入可见偏差)。
- `H3_2D -p eckart` (2D NN 拟合 RMSE 3.3e-3 → NN-PES 波包): 完成,
  summary 输出 NN PES fit RMSE。

## wheel 隔离验证

全新 venv 安装 `autoquantum-0.5.0-py3-none-any.whl`:
版本元数据 0.5.0; 34 项测试通过; `autoquantum --version` 正常;
`pip check` 无损坏依赖。

Wheel SHA-256:

```text
726aa4f57ad943d22de9663eaa8e2004dfaa24ee46c45a90b060f882c3115178
```

## 不在本版验证/修复范围内

- 力训练 (以 dV/dx 为监督目标) 未实现 — `gradient()` 仅为接口基础;
- NN 代理面上的反应概率继承拟合误差 (RMSE 日志为准, >2%·V_span 告警);
- 从头算电子结构后端 (ASE/PySCF) 缺失;
- 2D 定态求解器仍为实验性; LEPS 教学参数阈值非真实物理量;
- 跨平台/多 Python 版本矩阵。
