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
    early_stop_patience: int = 50


class NNTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[PESNN, Dict[str, List[float]]]:
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        n = X.shape[0]
        n_train = int(n * self.config.train_split)
        indices = np.random.RandomState(self.config.seed).permutation(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        layer_sizes = [1] + self.config.hidden_layers + [1]
        model = FeedForwardNN(
            layers=layer_sizes,
            activation=self.config.activation,
            seed=self.config.seed,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            if self.config.batch_size > 0:
                bs = self.config.batch_size
                for i in range(0, n_train, bs):
                    batch_X = X_train[i:i + bs]
                    batch_y = y_train[i:i + bs]
                    model.train_step(batch_X, batch_y, self.config.lr)
            else:
                model.train_step(X_train, y_train, self.config.lr)

            train_loss = np.mean((model.predict(X_train) - y_train.flatten()) ** 2)
            val_loss = np.mean((model.predict(X_val) - y_val.flatten()) ** 2)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d}/{self.config.epochs} | "
                      f"train_loss: {train_loss:.6e} | val_loss: {val_loss:.6e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = ([w.copy() for w in model.weights],
                              [b.copy() for b in model.biases])
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # 恢复验证损失最低时的权重 (早停/过拟合时避免返回已退化的末轮模型)
        if best_state is not None:
            model.weights, model.biases = best_state

        pes_nn = PESNN(model)
        return pes_nn, self.history
