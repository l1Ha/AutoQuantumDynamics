import os
import numpy as np
from typing import Dict, Any

from autoquantum.visualization.plot import PESPlotter, DynamicsPlotter, WavePacketPlotter
from autoquantum.visualization.contour import PESContourPlotter, ReactionPathPlotter
from autoquantum.visualization.wavepacket2d import WavePacket2DPlotter

try:
    from autoquantum import __version__ as AQ_VERSION
except ImportError:  # 包初始化期间的兜底
    AQ_VERSION = "0.4.0"


class DashboardGenerator:
    def __init__(self, results: Dict[str, Any], config):
        self.results = results
        self.config = config
        self.output_dir = config.output_dir
        self.dim = results.get("dynamics_dim", "1d")

    def generate_all(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if self.dim == "2d":
            self._plot_pes_2d()
        else:
            self._plot_pes()
            self._plot_nn_fit()
        self._plot_dynamics()
        self._plot_wavepacket()
        self._generate_html_report()

    def _plot_pes(self):
        grid = self.results.get("pes_grid")
        values = self.results.get("pes_values")
        if grid is not None and values is not None:
            PESPlotter.plot_pes(
                grid, values,
                save_path=os.path.join(self.output_dir, "pes_original.png"),
            )

    def _plot_nn_fit(self):
        grid = self.results.get("pes_grid")
        values = self.results.get("pes_values")
        fitted = self.results.get("nn_fitted_values")
        if all(x is not None for x in [grid, values, fitted]):
            PESPlotter.plot_pes(
                grid, values, fitted,
                save_path=os.path.join(self.output_dir, "pes_nn_fit.png"),
            )

    def _plot_pes_2d(self):
        builder = self.results.get("pes_2d")
        if builder is not None:
            R_grid, r_grid, V_grid = builder.generate_grid(
                R_range=(0.5, 9.0), r_range=(0.5, 6.0),
                n_R=150, n_r=150,
            )
            PESContourPlotter.plot_contour(
                R_grid, r_grid, V_grid,
                save_path=os.path.join(self.output_dir, "pes_contour.png"),
                title=f"{self.config.system_name} PES",
            )
            min_path = np.min(V_grid, axis=1)
            ReactionPathPlotter.plot_reaction_profile(
                R_grid, min_path,
                save_path=os.path.join(self.output_dir, "reaction_profile.png"),
            )

    def _plot_dynamics(self):
        result = self.results.get("dynamics_result")
        if result is not None:
            DynamicsPlotter.plot_transmission(
                result.energy, result.transmission,
                getattr(result, "reflection", None),
                save_path=os.path.join(self.output_dir, "transmission.png"),
            )

    def _plot_wavepacket(self):
        if "wavepacket_result" in self.results:
            wp = self.results["wavepacket_result"]
            ds_line = self.results.get("wp_ds_line")
            save_gif = getattr(self.config, "wp_save_gif", True)
            WavePacket2DPlotter.plot_snapshots(
                wp,
                save_path=os.path.join(self.output_dir, "wavepacket_snapshots.png"),
                n_show=4, ds_line=ds_line,
            )
            WavePacket2DPlotter.plot_probabilities(
                wp,
                save_path=os.path.join(self.output_dir, "wavepacket_probability.png"),
            )
            if save_gif:
                try:
                    WavePacket2DPlotter.save_animation(
                        wp,
                        save_path=os.path.join(self.output_dir, "wavepacket.gif"),
                        ds_line=ds_line,
                    )
                except Exception as exc:  # GIF 生成失败不阻塞主流程
                    print(f"  [warn] wavepacket.gif 生成失败: {exc}")

        if "wavepacket_result_1d" in self.results:
            times, psi_all = self.results["wavepacket_result_1d"]
            grid = self.results["dynamics_result"].grid
            WavePacketPlotter.plot_wavepacket_evolution(
                times, grid, psi_all,
                save_path=os.path.join(self.output_dir, "wavepacket_1d.png"),
            )

    def _generate_html_report(self):
        html_path = os.path.join(self.output_dir, "report.html")
        result = self.results.get("dynamics_result")

        lines = [
            "<!DOCTYPE html><html><head>",
            "<title>AutoQuantum Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5}",
            ".container{max-width:900px;margin:0 auto;background:white;padding:30px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}",
            "h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}",
            "h2{color:#2980b9;margin-top:30px}",
            ".stat{display:inline-block;background:#ecf0f1;padding:10px 20px;margin:5px;border-radius:5px}",
            ".stat-value{font-size:1.4em;font-weight:bold;color:#2c3e50}",
            ".stat-label{font-size:0.85em;color:#7f8c8d}",
            "img{max-width:100%;margin:15px 0;border:1px solid #ddd;border-radius:4px}",
            "footer{text-align:center;color:#95a5a6;margin-top:40px;font-size:0.85em}",
            "</style></head><body>",
            "<div class='container'>",
            f"<h1>AutoQuantum: {self.config.system_name}</h1>",
            "<h2>System Parameters</h2>",
            f"<div class='stat'><div class='stat-value'>{self.config.mass}</div><div class='stat-label'>Mass (au)</div></div>",
            f"<div class='stat'><div class='stat-value'>{self.config.pes_type}</div><div class='stat-label'>PES Type</div></div>",
            f"<div class='stat'><div class='stat-value'>{self.dim.upper()}</div><div class='stat-label'>Dimension</div></div>",
            f"<div class='stat'><div class='stat-value'>{getattr(self.config, 'method', 'auto')}</div><div class='stat-label'>Method</div></div>",
        ]

        if result is not None:
            lines.extend([
                "<h2>Dynamics Results</h2>",
                f"<div class='stat'><div class='stat-value'>{result.energy[0]:.4f} - {result.energy[-1]:.4f}</div><div class='stat-label'>Energy Range (au)</div></div>",
                f"<div class='stat'><div class='stat-value'>{result.transmission.max():.3f}</div><div class='stat-label'>Max Reaction Prob</div></div>",
            ])

        if "wp_final_reaction" in self.results:
            lines.append(
                f"<div class='stat'><div class='stat-value'>{self.results['wp_final_reaction']:.3f}</div>"
                "<div class='stat-label'>Wavepacket Final P_react</div></div>"
            )

        img_dir = self.output_dir
        candidates = ["pes_original.png", "pes_nn_fit.png",
                      "pes_contour.png", "reaction_profile.png",
                      "transmission.png",
                      "wavepacket_snapshots.png", "wavepacket_probability.png",
                      "wavepacket_1d.png", "wavepacket.gif"]
        for img in candidates:
            path = os.path.join(img_dir, img)
            if os.path.exists(path):
                name = img.replace(".png", "").replace(".gif", "").replace("_", " ").title()
                lines.append(f"<h2>{name}</h2>")
                lines.append(f"<img src='{img}' alt='{img}'>")

        lines.extend([
            f"<footer>Generated by AutoQuantum v{AQ_VERSION}</footer>",
            "</div></body></html>",
        ])

        with open(html_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved: {html_path}")
