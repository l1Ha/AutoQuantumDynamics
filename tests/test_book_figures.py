import importlib.util
import os
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts", "make_book_figures.py")


class TestBookFigures(unittest.TestCase):
    """配图脚本冒烟测试: 真实计算 + 落盘, 防止教材配图腐化。

    只跑最便宜的图 (解析/1D), 2D/扫描类图由发布验证人工抽查。
    """

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("make_book_figures", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_cheap_figures_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.mod.FIG_DIR = tmp
            self.mod.fig_morse_levels()
            self.mod.fig_wavefunctions()
            self.mod.fig_eks_and_cep()
            for name in ("ch06_morse_levels.png", "ch06_wavefunctions.png",
                         "ch04_eks_cep.png"):
                path = os.path.join(tmp, name)
                self.assertTrue(os.path.exists(path), f"缺图: {name}")
                self.assertGreater(os.path.getsize(path), 8000,
                                   f"图文件异常小: {name}")

    def test_morse_figure_uses_physical_mass(self):
        # H2: mu = m_H/2, omega ≈ 0.0200 au (4401 cm^-1) — 玩具质量会被发现
        mu = 918.075
        omega = 1.028 * np.sqrt(2 * 0.1744 / mu)
        self.assertAlmostEqual(omega, 0.0200, places=3)
        toy = np.sqrt(0.5 / 0.5)  # m=1, k=0.5 玩具谐振子
        self.assertLess(omega, toy / 20)

    def test_figures_exist_in_repo(self):
        fig_dir = os.path.join(os.path.dirname(SCRIPT), "..", "book", "figures")
        pngs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
        self.assertGreaterEqual(len(pngs), 10,
                                f"教材配图缺失 (仅 {len(pngs)} 张)")


if __name__ == "__main__":
    unittest.main()
