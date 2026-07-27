import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


class PESPlotter:
    @staticmethod
    def plot_pes(grid: np.ndarray, values: np.ndarray,
                 fitted: Optional[np.ndarray] = None,
                 save_path: str = "pes_plot.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grid, values, "b-", label="Original PES", linewidth=2)
        if fitted is not None:
            ax.plot(grid, fitted, "r--", label="NN Fitted PES", linewidth=2)
        ax.set_xlabel("Bond Length (Å)", fontsize=12)
        ax.set_ylabel("Potential Energy (au)", fontsize=12)
        ax.set_title("Potential Energy Surface", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")


class DynamicsPlotter:
    @staticmethod
    def plot_transmission(energy: np.ndarray, transmission: np.ndarray,
                          reflection: Optional[np.ndarray] = None,
                          save_path: str = "transmission_plot.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(energy, transmission, "g-", label="Transmission", linewidth=2)
        if reflection is not None:
            ax.plot(energy, reflection, "r-", label="Reflection", linewidth=2)
        ax.set_xlabel("Energy (au)", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title("Transmission / Reflection Probability", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")


class WavePacketPlotter:
    @staticmethod
    def plot_wavepacket_evolution(times: np.ndarray, grid: np.ndarray,
                                  psi: np.ndarray,
                                  save_path: str = "wavepacket_evo.png"):
        n_times = len(times)
        n_show = min(5, n_times)
        indices = np.linspace(0, n_times - 1, n_show, dtype=int)

        fig, axes = plt.subplots(n_show, 1, figsize=(8, 2 * n_show),
                                 sharex=True)
        if n_show == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            density = np.abs(psi[idx]) ** 2
            axes[i].plot(grid, density, "b-", linewidth=1.5)
            axes[i].set_ylabel(f"t={times[idx]:.2f}", fontsize=10)
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Position (au)", fontsize=12)
        fig.suptitle("Wavepacket Evolution", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
