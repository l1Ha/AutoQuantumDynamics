import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

from autoquantum.nn.model import FeedForwardNN, PESNN


@dataclass
class TrainingConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 32])
    activation: str = "tanh"
    epochs: int = 500
    lr: float = 0.001
    train_split: float = 0.8
    batch_size: int = 0
    seed: int = 42
    # 无相对改进 (阈值 1e-3) 的容忍轮数; 全批量 Adam 尾部收敛慢, 需留足
    early_stop_patience: int = 100
    # 输入/输出标准化 (在训练划分上拟合, 随模型持久化)
    normalize: bool = True
    # >0 时等距子采样到该点数 (大数据网格加速训练)
    max_train_points: int = 0
    # Adam 优化器 (0 时退回朴素梯度下降)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class NNTrainer:
    """多维 PES 回归训练器。

    - 输入 X: (n, d) 或 (n,) [d=1]; 目标 y: (n,)
    - 自动标准化 (x/y 各自 train 划分统计), 随模型保存/加载
    - 早停并恢复验证损失最低的权重
    - history 损失为物理单位 MSE
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[PESNN, Dict[str, List[float]]]:
        cfg = self.config
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if y.size != n:
            raise ValueError(f"X/y 样本数不一致: {n} vs {y.size}")

        # 等距子采样 (保持网格代表性)
        if cfg.max_train_points and n > cfg.max_train_points:
            idx = np.linspace(0, n - 1, cfg.max_train_points).astype(int)
            X, y = X[idx], y[idx]
            n = cfg.max_train_points

        n_train = int(n * cfg.train_split)
        indices = np.random.RandomState(cfg.seed).permutation(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = FeedForwardNN(
            layers=[d] + list(cfg.hidden_layers) + [1],
            activation=cfg.activation,
            seed=cfg.seed,
        )

        if cfg.normalize:
            x_mean = X_train.mean(axis=0)
            x_scale = X_train.std(axis=0)
            x_scale = np.where(x_scale == 0, 1.0, x_scale)
            y_mean = float(y_train.mean())
            y_scale = float(y_train.std()) or 1.0
            model.set_normalization(x_mean, x_scale, y_mean, y_scale)
            Xn_train = (X_train - x_mean) / x_scale
            Xn_val = (X_val - x_mean) / x_scale
            yn_train = (y_train - y_mean) / y_scale
        else:
            Xn_train, Xn_val, yn_train = X_train, X_val, y_train

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        # Adam 状态 (作用于标准化数据)
        use_adam = self.config.adam_beta1 > 0
        t = 0
        mW = [np.zeros_like(w) for w in model.weights]
        vW = [np.zeros_like(w) for w in model.weights]
        mB = [np.zeros_like(b) for b in model.biases]
        vB = [np.zeros_like(b) for b in model.biases]
        b1, b2, eps = (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps)

        def adam_step(Xb, yb):
            nonlocal t
            activations, zs = model.forward(Xb)
            dw, db = model._backward(Xb, yb, activations, zs)
            t += 1
            for i in range(len(model.weights)):
                mW[i] = b1 * mW[i] + (1 - b1) * dw[i]
                vW[i] = b2 * vW[i] + (1 - b2) * dw[i] ** 2
                mB[i] = b1 * mB[i] + (1 - b1) * db[i]
                vB[i] = b2 * vB[i] + (1 - b2) * db[i] ** 2
                mhat = mW[i] / (1 - b1 ** t)
                vhat = vW[i] / (1 - b2 ** t)
                model.weights[i] -= cfg.lr * mhat / (np.sqrt(vhat) + eps)
                mhat_b = mB[i] / (1 - b1 ** t)
                vhat_b = vB[i] / (1 - b2 ** t)
                model.biases[i] -= cfg.lr * mhat_b / (np.sqrt(vhat_b) + eps)

        for epoch in range(cfg.epochs):
            if cfg.batch_size > 0:
                bs = cfg.batch_size
                for i in range(0, len(Xn_train), bs):
                    if use_adam:
                        adam_step(Xn_train[i:i + bs], yn_train[i:i + bs])
                    else:
                        model.train_step(Xn_train[i:i + bs],
                                         yn_train[i:i + bs], cfg.lr)
            else:
                if use_adam:
                    adam_step(Xn_train, yn_train)
                else:
                    model.train_step(Xn_train, yn_train, cfg.lr)

            # 物理单位损失
            train_loss = np.mean((model.predict(X_train) - y_train) ** 2)
            val_loss = np.mean((model.predict(X_val) - y_val) ** 2)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d}/{cfg.epochs} | "
                      f"train_loss: {train_loss:.6e} | val_loss: {val_loss:.6e}")

            if val_loss < best_val_loss * (1.0 - 1e-3):
                best_val_loss = val_loss
                best_state = ([w.copy() for w in model.weights],
                              [b.copy() for b in model.biases])
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # 恢复验证损失最低时的权重 (早停/过拟合时避免返回已退化的末轮模型)
        if best_state is not None:
            model.weights, model.biases = best_state

        pes_nn = PESNN(model)
        return pes_nn, self.history
