import numpy as np

try:
    _trap = np.trapezoid
except AttributeError:  # numpy < 2.0
    _trap = np.trapz


def TransmissionProbability(psi: np.ndarray, grid: np.ndarray,
                            barrier_center: float) -> float:
    transmitted = grid > barrier_center
    return _trap(np.abs(psi[transmitted]) ** 2, grid[transmitted])


def ReflectionProbability(psi: np.ndarray, grid: np.ndarray,
                          barrier_center: float) -> float:
    reflected = grid < barrier_center
    return _trap(np.abs(psi[reflected]) ** 2, grid[reflected])
