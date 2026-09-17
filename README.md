# AutoQuantum

分子势能面与量子动力学的 Python 研究原型。项目目标是串联势能面构建、神经网络拟合、动力学计算和可视化；**目前尚未实现或验证完整的从头算到全维反应动力学流程**。

## v0.3.1：NN 回归修复补丁

本版仅修复神经网络回归输出层，并统一软件包版本号。未纳入开发中的二维含时波包、CLI 和动力学重构。详见 [CHANGELOG.md](CHANGELOG.md)。

- 隐藏层保留所选激活函数，输出层改为线性，能量预测不再受 tanh 的 `[-1, 1]` 范围限制。
- 现有反向传播与线性输出下的半均方误差梯度一致。
- 增加线性输出、超出激活范围的拟合和有限差分梯度回归测试。

**兼容性提示：**旧模型权重加载后也采用新的线性输出，预测会改变。旧拟合模型必须重新验证，建议重新训练；本版不保证旧模型预测兼容。

## 安装与检查

```bash
python -m pip install .
autoquantum --version
autoquantum info
python -m unittest discover -s tests -v
```

开发安装可使用 `python -m pip install -e .`。无需为本版的 NumPy 神经网络额外安装 PyTorch；旧 `requirements.txt` 仍包含该多余依赖。

## 最小 NN 示例

```python
import numpy as np
from autoquantum.nn.model import FeedForwardNN

x = np.linspace(0, 1, 60).reshape(-1, 1)
y = (3 * x + 1).ravel()
model = FeedForwardNN([1, 32, 1], activation="tanh", seed=42)
for _ in range(500):
    model.train_step(x, y, lr=0.01)
print("MSE:", np.mean((model.predict(x) - y) ** 2))
```

## 功能边界与已知限制

| 模块 | 当前状态 |
|---|---|
| 解析 PES | 已有 Morse、谐振子、LJ、LEPS、Eckart 代码；不代表真实体系精度已经验证 |
| 从头算 PES | `AbInitioData` 为数组数据容器；没有 ASE/PySCF 自动计算后端 |
| NN 拟合 | 基础一维训练流程；本版修复输出层；尚无完整归一化、力训练或多维训练管线 |
| 一维定态动力学 | 已有原型实现和基础测试，缺少充分的物理基准与收敛验证 |
| 二维定态动力学 | 现有递推、禁阻区处理和通量提取存在正确性问题，不应用其输出作科学结论 |
| 二维含时波包 | 开发中，本版不包含 |
| H3 CLI | 已知 `eckart`/`leps` 进入一维 PES 注册器后可能产生 `KeyError`，暂不推荐使用 |
| 一维含时波包 | 已知复数初始化错误及 NumPy 2.x `np.trapz` 兼容问题，本补丁未修复 |

其他限制：早停未恢复最佳权重；保存格式使用 pickle，仅加载可信模型；历史质量、长度单位与能量零点需由使用者核查。`scripts/sync_github.sh` 仍指向旧仓库并会修改 SSH 配置，**不要运行**；使用常规 git 命令管理远程。

测试通过仅指已有测试覆盖，不是对所有 CLI 路径、反应概率、势垒或物理精度的认证。开发分支的 Eckart 基准与 LEPS 诊断不属于本发布的验证证据。

## 后续目标

- 修复并独立验证动力学与 CLI 路径。
- 建立带单位、数据校验及误差评估的多维 PES 拟合流程。
- 接入可追溯的电子结构计算后端。
- 对二维含时传播、入口态和吸收边界开展收敛验证。
