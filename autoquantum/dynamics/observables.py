import numpy as np


def TransmissionProbability(psi: np.ndarray, grid: np.ndarray,
                            barrier_center: float) -> float:
    transmitted = grid > barrier_center
    return np.trapz(np.abs(psi[transmitted]) ** 2, grid[transmitted])


def ReflectionProbability(psi: np.ndarray, grid: np.ndarray,
                          barrier_center: float) -> float:
    reflected = grid < barrier_center
    return np.trapz(np.abs(psi[reflected]) ** 2, grid[reflected])
