"""二维 PES 可视化 — 等高线 + 3D 图。"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


class PESContourPlotter:
    """二维 PES 等高线与 3D 曲面绘图工具。"""

    @staticmethod
    def plot_contour(R: np.ndarray, r: np.ndarray, V: np.ndarray,
                     save_path: str = "pes_contour.png",
                     levels: int = 50,
                     title: str = "Potential Energy Surface"):
        """绘制二维 PES 填充等高线图 (Jacobi 坐标 R-r 平面)。

        Args:
            R: R 方向网格 (Bohr)。
            r: r 方向网格 (Bohr)。
            V: (n_R, n_r) 势能网格 (Hartree, 'ij' 索引)。
            save_path: 输出 PNG 文件路径。
            levels: 填充等高线层数。
            title: 图标题。
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        RR, rr = np.meshgrid(R, r, indexing="ij")
        cs = ax.contourf(RR, rr, V, levels=levels, cmap="viridis")
        ax.contour(RR, rr, V, levels=20, colors="white", linewidths=0.5, alpha=0.5)
        ax.set_xlabel("R (Bohr)", fontsize=12)
        ax.set_ylabel("r (Bohr)", fontsize=12)
        ax.set_title(title, fontsize=14)
        cbar = fig.colorbar(cs, ax=ax, label="Energy (Hartree)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    @staticmethod
    def plot_3d(R: np.ndarray, r: np.ndarray, V: np.ndarray,
                save_path: str = "pes_3d.png",
                title: str = "Potential Energy Surface (3D)"):
        """绘制二维 PES 三维曲面图 (R, r, V 轴均以 au 标注)。

        Args:
            R: R 方向网格 (Bohr)。
            r: r 方向网格 (Bohr)。
            V: (n_R, n_r) 势能网格 (Hartree)。
            save_path: 输出 PNG 文件路径。
            title: 图标题。
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        RR, rr = np.meshgrid(R, r, indexing="ij")
        surf = ax.plot_surface(RR, rr, V, cmap="viridis",
                               linewidth=0, antialiased=True, alpha=0.9)
        ax.set_xlabel("R (Bohr)", fontsize=11)
        ax.set_ylabel("r (Bohr)", fontsize=11)
        ax.set_zlabel("Energy (Hartree)", fontsize=11)
        ax.set_title(title, fontsize=14)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label="Energy (Hartree)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")


class ReactionPathPlotter:
    """反应路径 (最小能量剖面) 绘图工具。"""

    @staticmethod
    def plot_reaction_profile(rc: np.ndarray, V: np.ndarray,
                              save_path: str = "reaction_profile.png"):
        """绘制沿反应坐标的能量剖面, 红虚线标注最高势垒位置与数值。

        Args:
            rc: 反应坐标数组 (Bohr)。
            V: 对应能量 (Hartree)。
            save_path: 输出 PNG 文件路径。
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rc, V, "b-", linewidth=2)
        ax.axvline(x=rc[np.argmax(V)], color="r", linestyle="--",
                   alpha=0.5, label=f"Barrier: {V.max():.4f} au")
        ax.set_xlabel("Reaction Coordinate", fontsize=12)
        ax.set_ylabel("Energy (Hartree)", fontsize=12)
        ax.set_title("Minimum Energy Path", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
