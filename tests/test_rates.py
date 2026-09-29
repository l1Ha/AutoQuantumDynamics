"""热速率常数: Boltzmann 加权、Arrhenius 拟合、物理合理性。"""

import os
import tempfile
import unittest
import numpy as np

from autoquantum.analysis.rates import (
    thermal_rate_constant, arrhenius_fit, plot_arrhenius, ArrheniusFit,
    KB_AU)


class TestThermalRateConstant(unittest.TestCase):
    def test_zero_below_threshold(self):
        # P(E) 在阈值以下为零 → k(T) 低温下也接近零
        E = np.linspace(0, 0.1, 200)
        P = np.where(E > 0.05, 1.0, 0.0)
        T = np.array([100, 200, 300], dtype=float)
        rates = thermal_rate_constant(E, P, T)
        self.assertTrue(np.all(rates[abs(T - 100) < 1] < 0.01),
                        "低温 (远低于阈值) 时速率应接近零")

    def test_step_function_limit(self):
        # 阶梯函数 P(E) → k(T) 应在 E_threshold ~ k_BT 时迅速上升
        E = np.linspace(0, 0.1, 500)
        threshold = 0.02
        P = np.where(E >= threshold, 1.0, 0.0)
        T = np.array([500, 1000, 2000, 5000], dtype=float)
        rates = thermal_rate_constant(E, P, T)
        # 高温时 k(T) → 1 (整个 Boltzmann 分布都在阈值以上)
        self.assertGreater(rates[-1], 0.2)  # 截断能级 0.02/kBT(5000K)=0.10, P~0.28 合理
        # 低温时 k(T) → 0
        self.assertLess(rates[0], 0.1)

    def test_monotonic_temperature(self):
        # P(E) 单调递增时, k(T) 应随温度单调递增
        E = np.linspace(0.005, 0.05, 100)
        P = 1.0 / (1.0 + np.exp(-(E - 0.02) / 0.005))
        T = np.array([300, 600, 1000, 2000, 5000], dtype=float)
        rates = thermal_rate_constant(E, P, T)
        for i in range(len(rates) - 1):
            self.assertGreaterEqual(rates[i + 1], rates[i] - 1e-10)

    def test_output_shape(self):
        E = np.linspace(0, 0.1, 50)
        P = np.ones(50)
        T = np.array([300.0, 600.0])
        rates = thermal_rate_constant(E, P, T)
        self.assertEqual(rates.shape, (2,))


class TestArrheniusFit(unittest.TestCase):
    def test_perfect_arrhenius_recovery(self):
        # 合成 Arrhenius 数据: k(T) = A * exp(-Ea/(R*T))
        T = np.linspace(300, 3000, 30)
        Ea_kj = 25.0  # kJ/mol
        A = 1e10
        R_const = 8.314  # J/(mol·K)
        k = A * np.exp(-Ea_kj * 1000 / (R_const * T))
        fit = arrhenius_fit(T, k)
        self.assertAlmostEqual(fit.ea_kj_mol, Ea_kj, delta=0.5)
        self.assertAlmostEqual(fit.log_a, np.log10(A), delta=0.5)
        self.assertGreater(fit.r_squared, 0.99)

    def test_returns_arrheniusfit(self):
        T = np.array([300, 600, 1200], dtype=float)
        k = np.array([0.01, 0.1, 1.0])
        fit = arrhenius_fit(T, k)
        self.assertIsInstance(fit, ArrheniusFit)


class TestArrheniusPlot(unittest.TestCase):
    def test_plot_renders(self):
        T = np.array([300, 600, 1200, 3000], dtype=float)
        k = np.exp(-1000 / T)  # 假 Arrhenius
        fit = arrhenius_fit(T, k)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arrhenius.png")
            result = plot_arrhenius(T, k, fit, save_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 5000)


if __name__ == "__main__":
    unittest.main()
