import numpy as np
from typing import Dict, Any, Optional

class QuantumSystem:
    def __init__(self, name: str, mass: float, n_dim: int = 1):
        self.name = name
        self.mass = mass
        self.n_dim = n_dim

    def __repr__(self):
        return f"QuantumSystem(name={self.name}, mass={self.mass}, dim={self.n_dim})"


class PES:
    def __init__(self, name: str, system: QuantumSystem):
        self.name = name
        self.system = system

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)


class DynamicsResult:
    def __init__(self, energy: np.ndarray, transmission: np.ndarray,
                 reflection: np.ndarray, grid: Optional[np.ndarray] = None,
                 wavefunction: Optional[np.ndarray] = None):
        self.energy = energy
        self.transmission = transmission
        self.reflection = reflection
        self.grid = grid
        self.wavefunction = wavefunction

    def summary(self) -> Dict[str, Any]:
        return {
            "energy_range": [float(self.energy.min()), float(self.energy.max())],
            "n_energy_points": len(self.energy),
            "max_transmission": float(self.transmission.max()),
            "max_reflection": float(self.reflection.max()),
        }
