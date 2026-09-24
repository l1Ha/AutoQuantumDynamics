import shutil
import unittest
import numpy as np

from autoquantum.pes.calculators import (
    Calculator, AnalyticCalculator, demo_calculator,
    available_calculators, make_calculator, CommandBackendError,
)


class TestAnalyticBackend(unittest.TestCase):
    def test_analytic_gradient_vs_finite_difference(self):
        # 解析梯度与中心差分一致
        eps = 0.01, 0.0, 0.0
        coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        calc = demo_calculator()
        g = calc.gradient(coords)
        for i in range(2):
            for j in range(3):
                cp, cm = coords.copy(), coords.copy()
                cp[i, j] += 1e-6
                cm[i, j] -= 1e-6
                num = (calc.energy(cp) - calc.energy(cm)) / 2e-6
                self.assertAlmostEqual(g[i, j], num, places=5)

    def test_fd_fallback_matches_analytic(self):
        # 未提供解析梯度时中心差分回退应逼近解析梯度
        calc = demo_calculator()
        fd = AnalyticCalculator(calc.energy_fn, gradient_fn=None, grad_h=1e-5)
        coords = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
        np.testing.assert_allclose(fd.gradient(coords), calc.gradient(coords),
                                   rtol=1e-5, atol=1e-7)

    def test_energy_and_gradient(self):
        calc = demo_calculator()
        r_min = 2 ** (1 / 6) * 3.4      # LJ 极小点 → E = -ε
        coords = np.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]])
        e, g = calc.energy_and_gradient(coords)
        self.assertAlmostEqual(e, -0.01, places=6)
        self.assertEqual(g.shape, coords.shape)
        # 极小点处力为零
        np.testing.assert_allclose(g, 0.0, atol=1e-6)
        # 力的反作用: ∇_A E = -∇_B E
        far = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        g2 = calc.gradient(far)
        np.testing.assert_allclose(g2[0], -g2[1], rtol=1e-12)
        self.assertEqual(calc.provenance["backend"], "demo-lj")


class TestBackendRegistry(unittest.TestCase):
    def test_available_calculators_reports_status(self):
        status = available_calculators()
        for key in ("analytic", "demo", "xtb", "pyscf", "ase"):
            self.assertIn(key, status)
            self.assertIn(status[key]["available"], ("yes", "no"))
        self.assertEqual(status["xtb"]["available"],
                         "yes" if shutil.which("xtb") else "no")

    def test_make_calculator_demo_and_unknown(self):
        calc = make_calculator("demo")
        self.assertIsInstance(calc, Calculator)
        with self.assertRaises(ValueError):
            make_calculator("no-such-backend")

    def test_xtb_missing_gives_informative_error(self):
        if shutil.which("xtb"):
            self.skipTest("xtb 已安装, 错误路径不可测")
        with self.assertRaises(CommandBackendError) as ctx:
            make_calculator("xtb", symbols=["H", "H"])
        self.assertIn("xtb", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
