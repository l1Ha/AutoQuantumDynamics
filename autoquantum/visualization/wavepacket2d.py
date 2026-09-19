"""二维含时波包可视化 — 快照、通道概率、GIF 动画。"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class WavePacket2DPlotter:
    """WavePacket2DResult 的可视化工具集。"""

    @staticmethod
    def plot_snapshots(result, save_path: str = "wavepacket_snapshots.png",
                       n_show: int = 4, V_clip=(-0.2, 0.3),
                       ds_line=None, title: str = "2D Wavepacket Evolution"):
        """波包密度 |ψ(R, r)|² 快照，叠加 PES 等高线与分界面。"""
        if result.snapshots is None:
            raise ValueError("结果未保存密度快照 (save_density=False)")

        n_times = len(result.times)
        n_show = min(n_show, n_times)
        idx = np.linspace(0, n_times - 1, n_show, dtype=int)
        ncol = 2
        nrow = (n_show + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.2 * nrow))
        axes = np.atleast_1d(axes).ravel()

        RR, rr = np.meshgrid(result.R_grid, result.r_grid, indexing="ij")
        Vc = np.clip(result.V_grid, *V_clip)
        dmax = max(float(s.max()) for s in result.snapshots[idx])

        for ax, k in zip(axes, idx):
            im = ax.pcolormesh(result.R_grid, result.r_grid,
                               result.snapshots[k].T, cmap="magma",
                               shading="auto", vmin=0.0, vmax=dmax)
            ax.contour(RR, rr, Vc, levels=16, colors="cyan",
                       linewidths=0.5, alpha=0.55)
            if ds_line is not None:
                ax.plot(ds_line[0], ds_line[1], "w--", linewidth=1.2, alpha=0.8)
            ax.set_xlabel("R (Bohr)", fontsize=11)
            ax.set_ylabel("r (Bohr)", fontsize=11)
            ax.set_title(f"t = {result.times[k]:.0f} au", fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046, label="|ψ|²")

        for ax in axes[n_show:]:
            ax.axis("off")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    @staticmethod
    def plot_probabilities(result, save_path: str = "wavepacket_probability.png"):
        """反应/反射/存活概率随时间演化。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result.times, result.reaction_prob, "g-",
                linewidth=2, label=f"Reaction (final: {result.reaction_prob[-1]:.3f})")
        ax.plot(result.times, result.reflection_prob, "r-",
                linewidth=2, label=f"Reflection (final: {result.reflection_prob[-1]:.3f})")
        ax.plot(result.times, result.norm_t, "b--",
                linewidth=1.5, alpha=0.8, label="Surviving norm")
        ax.set_xlabel("Time (au)", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        e = result.energy
        e_str = f"E = {e[0]:.4f} au" if e is not None else ""
        ax.set_title(f"Channel Probabilities {e_str}", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    @staticmethod
    def save_animation(result, save_path: str = "wavepacket.gif",
                       fps: int = 8, max_frames: int = 48,
                       V_clip=(-0.2, 0.3), ds_line=None):
        """波包传播 GIF 动画 (密度 + PES 等高线 + 分界面)。"""
        if result.snapshots is None:
            raise ValueError("结果未保存密度快照 (save_density=False)")

        n_times = len(result.times)
        n_frames = min(max_frames, n_times)
        idx = np.linspace(0, n_times - 1, n_frames, dtype=int)

        RR, rr = np.meshgrid(result.R_grid, result.r_grid, indexing="ij")
        Vc = np.clip(result.V_grid, *V_clip)
        dmax = max(float(result.snapshots[k].max()) for k in idx[:max(1, n_frames // 4)])

        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.contour(RR, rr, Vc, levels=16, colors="cyan",
                   linewidths=0.6, alpha=0.6)
        if ds_line is not None:
            ax.plot(ds_line[0], ds_line[1], "w--", linewidth=1.2, alpha=0.8)
        mesh = ax.pcolormesh(result.R_grid, result.r_grid,
                             result.snapshots[idx[0]].T, cmap="magma",
                             shading="auto", vmin=0.0, vmax=dmax)
        time_text = ax.set_title(f"t = {result.times[idx[0]]:.0f} au", fontsize=13)
        ax.set_xlabel("R (Bohr)", fontsize=11)
        ax.set_ylabel("r (Bohr)", fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.046, label="|ψ|²")

        def update(fi):
            mesh.set_array(result.snapshots[idx[fi]].T.ravel())
            time_text.set_text(f"t = {result.times[idx[fi]]:.0f} au")
            return mesh, time_text

        anim = FuncAnimation(fig, update, frames=n_frames,
                             interval=1000 / fps, blit=False)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  Saved: {save_path}")
