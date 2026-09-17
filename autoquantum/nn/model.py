import numpy as np
from typing import List, Optional, Callable


class FeedForwardNN:
    def __init__(self, layers: List[int],
                 activation: str = "tanh",
                 seed: Optional[int] = None):
        self.layers = layers
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

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
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
                # Linear output layer: PES values are real-valued and may lie
                # outside the range of bounded activations (e.g. tanh in
                # [-1, 1]), so the final layer must not be squashed.
                activations.append(z)
            else:
                activations.append(self.activation_fn(z))

        return activations, zs

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1].flatten()

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
            }, f)

    @classmethod
    def load(cls, path: str) -> "FeedForwardNN":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layers"], data["activation"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        return model


class PESNN:
    def __init__(self, model: FeedForwardNN):
        self.model = model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "PESNN":
        return cls(FeedForwardNN.load(path))
