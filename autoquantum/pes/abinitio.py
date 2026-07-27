import numpy as np
from typing import Dict, Any, Tuple, Optional


class AbInitioData:
    def __init__(self):
        self.points: Optional[np.ndarray] = None
        self.energies: Optional[np.ndarray] = None
        self.gradients: Optional[np.ndarray] = None

    def from_arrays(self, points: np.ndarray, energies: np.ndarray,
                    gradients: Optional[np.ndarray] = None):
        self.points = np.asarray(points)
        self.energies = np.asarray(energies)
        self.gradients = np.asarray(gradients) if gradients is not None else None

    def from_analytic_pes(self, builder, grid: np.ndarray,
                          noise_level: float = 0.0) -> "AbInitioData":
        values = builder.evaluate(grid)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.max(values), size=values.shape)
            values = values + noise
        self.points = grid
        self.energies = values
        return self

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
        test.points = self.points[indices[n_train:]]
        test.energies = self.energies[indices[n_train:]]

        return train, test
