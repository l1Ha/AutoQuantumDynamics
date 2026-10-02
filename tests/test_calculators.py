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


class TestPySCFCalculatorMock(unittest.TestCase):
    """测试 PySCFCalculator 的高自旋约束、spin_lock、MOM 及复共振势能接口 (Mock 隔离)。"""

    def _get_mock_modules(self):
        import sys
        from unittest.mock import MagicMock
        mock_pyscf = MagicMock()
        mock_pyscf.__version__ = "2.4.0"
        mock_gto = MagicMock()
        mock_scf = MagicMock()
        mock_dft = MagicMock()
        mock_addons = MagicMock()
        mock_lib = MagicMock()
        mock_lib.asarray.side_effect = lambda x: np.array(x)

        mock_pyscf.gto = mock_gto
        mock_pyscf.scf = mock_scf
        mock_scf.addons = mock_addons
        mock_pyscf.dft = mock_dft
        mock_pyscf.lib = mock_lib

        mods = {
            "pyscf": mock_pyscf,
            "pyscf.gto": mock_gto,
            "pyscf.scf": mock_scf,
            "pyscf.scf.addons": mock_addons,
            "pyscf.dft": mock_dft,
            "pyscf.lib": mock_lib,
        }
        return mods, mock_pyscf

    def test_pyscf_missing_gives_informative_error(self):
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"pyscf": None}):
            with self.assertRaises(CommandBackendError) as ctx:
                make_calculator("pyscf", symbols=["He", "Li"])
            self.assertIn("未安装 pyscf", str(ctx.exception))

    def test_rhf_energy_and_gradient(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.235
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.RHF.return_value = mock_mf

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["He", "Li"], basis="def2-svp", method="rhf")
            coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            e, g = calc.energy_and_gradient(coords)

            self.assertAlmostEqual(e, -7.235)
            self.assertEqual(g.shape, (2, 3))
            self.assertAlmostEqual(g[0, 0], 0.01)
            self.assertEqual(calc.provenance["method"], "rhf")
            self.assertEqual(calc.provenance["basis"], "def2-svp")

    def test_high_spin_rohf_and_uhf(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.150
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.ROHF.return_value = mock_mf

        with patch.dict(sys.modules, mods):
            # 四重态 (He* + Li, S=3/2, 2S=3)
            calc = make_calculator("pyscf", symbols=["He", "Li"], spin=3, method="rohf")
            coords = np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
            e = calc.energy(coords)
            self.assertAlmostEqual(e, -7.150)
            mock_pyscf.scf.ROHF.assert_called_once()
            self.assertEqual(calc.provenance["spin"], "3")

    def test_spin_lock_audit_pass_and_fail(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.120
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.UHF.return_value = mock_mf

        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        with patch.dict(sys.modules, mods):
            # 期望 S=1.5, S(S+1)=3.75
            # 情况 1: 自旋污染在容差内 (3.77 vs 3.75, diff=0.02 <= 0.1) -> 成功
            mock_mf.spin_square.return_value = (3.77, 1.506)
            calc_pass = make_calculator("pyscf", symbols=["He", "Li"], spin=3,
                                        method="uhf", spin_lock=True, spin_tol=0.1)
            e = calc_pass.energy(coords)
            self.assertAlmostEqual(e, -7.120)

            # 情况 2: 自旋污染超标 / 态翻转 (2.50 vs 3.75, diff=1.25 > 0.1) -> 触发拦截
            mock_mf.spin_square.return_value = (2.50, 1.1)
            calc_fail = make_calculator("pyscf", symbols=["He", "Li"], spin=3,
                                        method="uhf", spin_lock=True, spin_tol=0.1)
            with self.assertRaises(CommandBackendError) as ctx:
                calc_fail.energy(coords)
            self.assertIn("自旋锁定失败", str(ctx.exception))
            self.assertIn("自旋污染偏差", str(ctx.exception))

    def test_dft_and_xc_dispatch(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.300
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.dft.UKS.return_value = mock_mf

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["He", "Li"], spin=1,
                                   method="uks", xc="pbe")
            coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
            e = calc.energy(coords)
            self.assertAlmostEqual(e, -7.300)
            mock_pyscf.dft.UKS.assert_called_once()
            self.assertEqual(mock_mf.xc, "pbe")
            self.assertEqual(calc.provenance["xc"], "pbe")

    def test_mom_tracks_reference_orbitals(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        # 模拟两步计算
        fake_mo_1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        fake_occ_1 = np.array([2.0, 0.0])

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.100
        mock_mf.mo_coeff = fake_mo_1
        mock_mf.mo_occ = fake_occ_1
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.RHF.return_value = mock_mf
        mock_pyscf.scf.addons.mom_occ.return_value = mock_mf

        coords_1 = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        coords_2 = np.array([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]])

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["He", "Li"], method="rhf", use_mom=True)
            # 第 1 步: 无先验参考, mom_occ 不应调用
            calc.energy(coords_1)
            mock_pyscf.scf.addons.mom_occ.assert_not_called()

            # 第 2 步: 存在第 1 步轨道, mom_occ 必须被注入调用
            calc.energy(coords_2)
            mock_pyscf.scf.addons.mom_occ.assert_called_once()

            # 重置 MOM
            calc.reset_mom()
            self.assertIsNone(calc._ref_mo_coeff)

    def test_resonance_width_and_complex_energy(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol
        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.000
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.RHF.return_value = mock_mf

        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        with patch.dict(sys.modules, mods):
            # 1. 默认无 CAP: 衰减宽度为 0, 复能量虚部为 0
            calc_real = make_calculator("pyscf", symbols=["He", "Li"])
            self.assertAlmostEqual(calc_real.resonance_width(coords), 0.0)
            self.assertEqual(calc_real.complex_energy(coords), complex(-7.000, 0.0))

            # 2. 指数型自电离宽度: Gamma(R) = A * exp(-beta * R)
            cap_exp = {"type": "exponential", "A": 0.05, "beta": 0.5, "r_index": (0, 1)}
            calc_exp = make_calculator("pyscf", symbols=["He", "Li"], cap_params=cap_exp)
            expected_gamma = 0.05 * np.exp(-0.5 * 4.0)
            self.assertAlmostEqual(calc_exp.resonance_width(coords), expected_gamma, places=7)
            z = calc_exp.complex_energy(coords)
            self.assertAlmostEqual(z.real, -7.000)
            self.assertAlmostEqual(z.imag, -0.5 * expected_gamma)

            # 3. 盒式 CAP 宽度: r_cap=3.0, R=4.0 > 3.0 -> Gamma = 2 * eta * (4 - 3)^2
            cap_box = {"type": "box", "eta": 0.01, "r_cap": 3.0, "r_index": (0, 1)}
            calc_box = make_calculator("pyscf", symbols=["He", "Li"], cap_params=cap_box)
            self.assertAlmostEqual(calc_box.resonance_width(coords), 2.0 * 0.01 * (1.0 ** 2))


if __name__ == "__main__":
    unittest.main()
