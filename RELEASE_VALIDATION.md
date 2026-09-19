# v0.6.0 release validation

## Scope

力训练 (double backprop, 以 ∂V/∂x 为监督目标) 及其在 engine 二维
管线的默认接入。v0.4.0/v0.5.0 的验证结论继续有效; 从头算梯度
(ASE/PySCF) 与能量反卷积仍属路线图。

## 正确性门控

- **double-backprop 有限差分校验**: `input_gradient_backward` 的解析
  梯度 vs Φ = Σ adjG·input_gradient 的中心差分, 三种激活函数
  (tanh/sigmoid/relu) 全部权重/偏置逐元素一致, 最大相对误差
  ~10⁻¹⁰ (test_nn.test_force_gradient_backward_fd, tanh)。
- 实现要点: pass-1 映射 T = G·Wᵀ 的转置方向与前向 backprop 相反
  (dW = barTᵀ·G, barG = barT·W); 梯度对偏置的依赖经 f″(z) 项进入
  前向图伴随。

## 数值结果 (固定种子, 可复现)

Eckart 2D ([40,40], Adam lr=0.005, 1200 轮, 150×150 网格 → 6000 子采样):

| force_weight | V_RMSE (au) | 梯度 RMSE (au/Bohr) |
|---|---|---|
| 0 (纯能量) | 4.6×10⁻³ | 2.2×10⁻² |
| 0.5 | 1.1×10⁻³ | 4.4×10⁻³ |
| 1.0 (默认) | ~10⁻³ | ~2–6×10⁻³ |
| 2.0 | 4.6×10⁻⁴ | 2.3×10⁻³ |

- CLI 实测 (w=1.0, 800 轮): V_RMSE 4.5×10⁻⁴ au (0.03% V span),
  梯度 RMSE 2.1×10⁻³; NN 面上波包传播守恒恒等式保持。
- 多维测试函数力训练断言: 梯度 RMSE 降至纯能量拟合的 50% 以下,
  能量拟合不劣化 (test_force_training_improves_gradient)。

## 测试与 wheel 验证

- 全量 36 项测试通过 (macOS arm64, Python 3.14.5); 隔离 wheel 环境
  58 s, 集成测试 (力训练 1200 轮) RMSE 1.4×10⁻³ au (0.07% V span)。
- 全新 venv 安装 `autoquantum-0.6.0-py3-none-any.whl`: 版本元数据
  0.6.0; `autoquantum --version` 正常; `pip check` 无损坏依赖。

Wheel SHA-256:

```text
7478d0299b815eec3030d286e3728f59005e6f0d484af7c5055e0e3f7d7b5a9c
```

## 不在本版验证/修复范围内

- 从头算梯度来源 (ASE/PySCF); 力训练目前依赖解析/差分梯度;
- 波包能量反卷积与收敛自动化;
- 2D 定态求解器仍为实验性; LEPS 教学参数阈值非真实物理量;
- 朴素梯度下降路径 (adam_beta1=0) 不支持力训练 (有告警);
- 跨平台/多 Python 版本矩阵。
