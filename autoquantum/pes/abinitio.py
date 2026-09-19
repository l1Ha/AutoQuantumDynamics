import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional, Sequence


class AbInitioData:
    """PES 训练数据容器 (点 / 能量 / 可选梯度)。

    注意: 本类**不执行**电子结构计算 — 它只承载外部来源 (解析面、
    文件、量子化学程序) 给出的数据。用 ``sample_function`` 可从任意
    可调用势能函数生成带有限差分梯度的训练集。
    """

    def __init__(self):
        self.points: Optional[np.ndarray] = None
        self.energies: Optional[np.ndarray] = None
        self.gradients: Optional[np.ndarray] = None

    def from_arrays(self, points: np.ndarray, energies: np.ndarray,
                    gradients: Optional[np.ndarray] = None):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        self.points = points
        self.energies = np.asarray(energies, dtype=float).ravel()
        self.gradients = (np.asarray(gradients, dtype=float)
                          if gradients is not None else None)

    def from_analytic_pes(self, builder, grid: np.ndarray,
                          noise_level: float = 0.0) -> "AbInitioData":
        values = builder.evaluate(grid)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.max(values), size=values.shape)
            values = values + noise
        self.points = grid
        self.energies = values
        return self

    @classmethod
    def sample_function(cls, func: Callable,
                        ranges: Sequence[Tuple[float, float]],
                        n_per_dim: int,
                        noise_level: float = 0.0,
                        grad_h: float = 1e-5,
                        seed: Optional[int] = None) -> "AbInitioData":
        """在规则网格上采样标量函数 V(x)，生成带中心差分梯度的训练集。

        Parameters
        ----------
        func : Callable
            接受 (n, d) 点数组、返回 (n,) 能量的可调用对象,
            例如 ``lambda pts: pes(pts[:, 0], pts[:, 1])``。
        ranges : [(lo, hi), ...]
            每维采样范围, d = len(ranges)。
        n_per_dim : int
            每维网格点数 (总点数 n_per_dim**d)。
        noise_level : float
            加在能量上的高斯噪声幅度 (乘以 max|V|)。
        grad_h : float
            中心差分步长 (能量单位)。
        """
        if n_per_dim < 2:
            raise ValueError("n_per_dim 必须 >= 2 (差分需要)")
        d = len(ranges)
        axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in ranges]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([m.ravel() for m in mesh])
        values = np.asarray(func(points), dtype=float).ravel()

        gradients = np.zeros_like(points)
        for j in range(d):
            hp = np.zeros(d); hp[j] = grad_h
            vp = np.asarray(func(points + hp), dtype=float).ravel()
            vm = np.asarray(func(points - hp), dtype=float).ravel()
            gradients[:, j] = (vp - vm) / (2.0 * grad_h)

        if noise_level > 0:
            rng = np.random.RandomState(seed)
            values = values + rng.normal(0.0, noise_level * np.max(np.abs(values)),
                                         size=values.shape)

        data = cls()
        data.from_arrays(points, values, gradients)
        return data

    @property
    def n_points(self) -> int:
        return len(self.points) if self.points is not None else 0

    def split(self, train_ratio: float = 0.8) -> Tuple["AbInitioData", "AbInitioData"]:
        n = self.n_points
        indices = np.random.permutation(n)
        n_train = int(n * train_ratio)

        train = AbInitioData()
        test = AbInitioData()

        train.points = self.points[indices[:n_train]]
        train.energies = self.energies[indices[:n_train]]
        if self.gradients is not None:
            train.gradients = self.gradients[indices[:n_train]]
        test.points = self.points[indices[n_train:]]
        test.energies = self.energies[indices[n_train:]]
        if self.gradients is not None:
            test.gradients = self.gradients[indices[n_train:]]

        return train, test
