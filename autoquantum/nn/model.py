import numpy as np
from typing import List, Optional, Callable


class FeedForwardNN:
    """纯 NumPy 前馈网络 (回归: 线性输出层)。

    支持任意维输入 ``layers[0] = d``。内置可选的输入/输出归一化
    (由 ``NNTrainer`` 在训练数据上拟合): ``predict``/``gradient``
    接受**物理单位**数据, 内部自动标准化/反标准化; ``train_step``
    则始终作用于原始 (标准化后) 数据, 由训练器负责喂标准化数组。

    ``gradient(X)`` 返回解析输入梯度 dŷ/dX (经反标准化链式修正),
    与有限差分校验一致 — 为后续力训练提供基础。
    """

    def __init__(self, layers: List[int],
                 activation: str = "tanh",
                 seed: Optional[int] = None):
        self.layers = list(layers)
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)
        self.activation_deriv = self._get_derivative(activation)

        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            lim = np.sqrt(6 / (layers[i] + layers[i + 1]))
            self.weights.append(np.random.uniform(-lim, lim, (layers[i], layers[i + 1])))
            self.biases.append(np.zeros((1, layers[i + 1])))

        # 归一化参数 (默认未启用)
        self.x_mean: Optional[np.ndarray] = None
        self.x_scale: Optional[np.ndarray] = None
        self.y_mean: Optional[float] = None
        self.y_scale: Optional[float] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _get_activation(name: str) -> Callable:
        if name == "tanh":
            return lambda x: np.tanh(x)
        elif name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "linear":
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation: {name}")

    @staticmethod
    def _get_derivative(name: str) -> Callable:
        if name == "tanh":
            return lambda x: 1 - np.tanh(x) ** 2
        elif name == "sigmoid":
            s = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
            return lambda x: s(x) * (1 - s(x))
        elif name == "relu":
            return lambda x: (x > 0).astype(float)
        elif name == "linear":
            return lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {name}")

    # ------------------------------------------------------------------
    def set_normalization(self, x_mean: np.ndarray, x_scale: np.ndarray,
                          y_mean: float, y_scale: float):
        """设置归一化参数; x_scale/y_scale 为 0 时按 1 处理。"""
        self.x_mean = np.asarray(x_mean, dtype=float).ravel()
        self.x_scale = np.asarray(x_scale, dtype=float).ravel()
        self.x_scale = np.where(self.x_scale == 0, 1.0, self.x_scale)
        self.y_mean = float(y_mean)
        self.y_scale = float(y_scale) if float(y_scale) != 0 else 1.0

    def _shape_input(self, X: np.ndarray) -> np.ndarray:
        """(n,) → (n,1) [d=1] 或 (d,) → (1,d) [d>1]。"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if self.layers[0] == 1 else X.reshape(1, -1)
        return X

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None:
            return X
        return (X - self.x_mean) / self.x_scale

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        if self.y_mean is None:
            return y
        return y * self.y_scale + self.y_mean

    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> tuple:
        """原始前向 (不做归一化 — 训练器喂标准化数据)。

        隐藏层施加激活函数, 输出层线性 (PES 值可超出有界激活值域)。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        activations = [X]
        zs = []
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            zs.append(z)
            if i == n_layers - 1:
                # 线性输出层: PES 值为实数, 可能超出 tanh 等有界激活的值域
                activations.append(z)
            else:
                activations.append(self.activation_fn(z))

        return activations, zs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """物理单位预测。输入 (n, d) / (d,) [单点] / (n,) [d=1]。"""
        X = self._shape_input(X)
        Xn = self._normalize_x(X)
        activations, _ = self.forward(Xn)
        return self._denormalize_y(activations[-1]).ravel()

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """解析输入梯度 dŷ/dX, 物理单位, 形状 (n, d)。

        反向传播起始于线性输出 (导数 1), 逐层乘 Wᵀ 与隐藏层激活导数,
        最后经归一化链式修正 y_scale / x_scale。
        """
        X = self._shape_input(X)
        Xn = self._normalize_x(X)

        activations = [Xn]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            zs.append(z)
            activations.append(z if i == len(self.weights) - 1
                               else self.activation_fn(z))

        # d ŷ_std / d x_std: 从线性输出回传
        g = np.ones_like(zs[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            g = g @ self.weights[i].T
            if i > 0:
                g = g * self.activation_deriv(zs[i - 1])

        if self.x_mean is not None:
            g = g * (self.y_scale / self.x_scale)
        return g

    def _backward(self, X: np.ndarray, y: np.ndarray,
                  activations: List[np.ndarray],
                  zs: List[np.ndarray]) -> tuple:
        m = X.shape[0]
        y = y.reshape(-1, 1)

        delta = activations[-1] - y
        dw = [activations[-2].T @ delta / m]
        db = [np.sum(delta, axis=0, keepdims=True) / m]

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * self.activation_deriv(zs[i])
            dw.insert(0, activations[i].T @ delta / m)
            db.insert(0, np.sum(delta, axis=0, keepdims=True) / m)

        return dw, db

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.001):
        activations, zs = self.forward(X)
        dw, db = self._backward(X, y, activations, zs)

        for i in range(len(self.weights)):
            self.weights[i] -= lr * dw[i]
            self.biases[i] -= lr * db[i]

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "layers": self.layers,
                "activation": self.activation_name,
                "weights": self.weights,
                "biases": self.biases,
                "x_mean": self.x_mean,
                "x_scale": self.x_scale,
                "y_mean": self.y_mean,
                "y_scale": self.y_scale,
            }, f)

    @classmethod
    def load(cls, path: str) -> "FeedForwardNN":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layers"], data["activation"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        # 兼容无归一化字段的旧模型文件
        model.x_mean = data.get("x_mean")
        model.x_scale = data.get("x_scale")
        model.y_mean = data.get("y_mean")
        model.y_scale = data.get("y_scale")
        return model


class PESNN:
    """势能面代理模型: 支持一维 (n,) 与二维 (n, d) 物理单位调用。"""

    def __init__(self, model: FeedForwardNN):
        self.model = model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """解析梯度 dV/dx, 形状 (n, d)。"""
        return self.model.gradient(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "PESNN":
        return cls(FeedForwardNN.load(path))
