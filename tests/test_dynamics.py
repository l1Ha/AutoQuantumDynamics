import unittest
import numpy as np
from autoquantum.dynamics.quantum_1d import QuantumScattering1D


class TestDynamics1D(unittest.TestCase):
    def test_scattering_solve(self):
        def flat_pes(x):
            return np.zeros_like(x)

        solver = QuantumScattering1D(
            mass=1.0, pes=flat_pes,
            n_grid=100, grid_min=0, grid_max=5,
        )
        result = solver.solve(0.01, 0.1, 10)
        self.assertEqual(len(result.energy), 10)
        self.assertEqual(len(result.transmission), 10)


if __name__ == "__main__":
    unittest.main()
