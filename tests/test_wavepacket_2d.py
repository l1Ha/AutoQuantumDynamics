import os
import tempfile
import unittest
import numpy as np

from autoquantum.dynamics.wavepacket_2d import (
    WavePacket2D,
    WavePacket2DPropagator,
    WavePacket2DScan,
    h3_reduced_masses,
    leps_jacobi_pes,
    leps_exchange_mask,
    eckart_product_mask,
    morse_ground_width,
    harmonic_ground_width,
)
from autoquantum.dynamics.quantum_1d import QuantumScattering1D
from autoquantum.pes.eckart import EckartBarrierPES
from autoquantum.pes.leps import LEPSBuilder


class TestFreePacket(unittest.TestCase):
    """无势场、无吸收: 严格幺正性检验。"""

    def test_norm_conservation_and_translation(self):
        R = np.linspace(0, 8, 64)
        r = np.linspace(0, 8, 64)
        prop = WavePacket2DPropagator(
            lambda R, r: np.zeros_like(R), R, r,
            mass_R=100.0, mass_r=100.0, dt=0.02, cap_edges=(),
        )
        packet = WavePacket2D(R0=3.0, r0=4.0, sigma_R=0.4, sigma_r=0.4, p_R0=5.0)
        psi = packet.initialize(R, r)
        dV = (R[1] - R[0]) * (r[1] - r[0])

        n0 = np.sum(np.abs(psi) ** 2) * dV
        R0 = prop.expectation_R(psi)
        for _ in range(100):  # t = 2.0
            psi = prop.step(psi)
        n1 = np.sum(np.abs(psi) ** 2) * dV
        R1 = prop.expectation_R(psi)

        self.assertAlmostEqual(n1, n0, places=10)
        # <R>(t) = <R>(0) + p/m * t
        self.assertAlmostEqual(R1 - R0, 5.0 / 100.0 * 2.0, places=6)

    def test_ground_state_is_stationary(self):
        # 谐振子基态宽度约定: sigma = 1/sqrt(mu*omega) 为振幅宽度,
        # 以此初始化的波包在纯谐振势下必须保持静止 (形状与质心不变)
        mass = 50.0
        k = 0.5
        omega = np.sqrt(k / mass)
        sigma = harmonic_ground_width(k, mass)
        R = np.linspace(-3, 3, 96)
        r = np.linspace(-3, 3, 96)
        prop = WavePacket2DPropagator(
            lambda R, r: 0.5 * k * (R ** 2 + r ** 2), R, r,
            mass_R=mass, mass_r=mass, dt=0.05, cap_edges=(),
        )
        packet = WavePacket2D(R0=0.0, r0=0.0, sigma_R=sigma, sigma_r=sigma)
        psi = packet.initialize(R, r)
        dV = (R[1] - R[0]) * (r[1] - r[0])
        rho0 = np.abs(psi) ** 2
        com0 = (rho0[..., None] * 0).sum() + np.sum(
            rho0 * (R[:, None] + r[None, :])) * dV
        for _ in range(126):  # 一个周期 T = 2*pi/omega ≈ 62.9
            psi = prop.step(psi)
        rho1 = np.abs(psi) ** 2
        com1 = np.sum(rho1 * (R[:, None] + r[None, :])) * dV
        # 分裂误差量级 ~ (ωdt)³·t; 允许 1e-5
        self.assertLess(np.abs(rho1 - rho0).max(), 1e-5)
        self.assertAlmostEqual(com1 - com0, 0.0, places=8)


class TestAbsorbingBoundary(unittest.TestCase):
    """CAP 吸收: 总概率 (存活 + 各边吸收) 严格守恒。"""

    def test_absorption_conserves_total_probability(self):
        R = np.linspace(0, 8, 128)
        r = np.linspace(0, 8, 128)
        prop = WavePacket2DPropagator(
            lambda R, r: np.zeros_like(R), R, r,
            mass_R=10.0, mass_r=10.0, dt=0.05,
            cap_edges=("R_max",), cap_width_frac=0.3, cap_height=1.0,
        )
        packet = WavePacket2D(R0=2.0, r0=4.0, sigma_R=1.0, sigma_r=0.4, p_R0=3.0)
        result = prop.propagate(packet.initialize(R, r), n_steps=900,
                                save_every=50, save_density=False)
        total = result.norm_t[-1] + sum(v[-1] for v in result.absorbed.values())
        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertGreater(result.absorbed["R_max"][-1], 0.95)

    def test_corner_attribution_deterministic_and_conserving(self):
        # 四边全部启用且角点重叠: 归因确定, 总量仍守恒
        R = np.linspace(0, 10, 96)
        r = np.linspace(0, 10, 96)
        prop = WavePacket2DPropagator(
            lambda R, r: np.zeros_like(R), R, r,
            mass_R=10.0, mass_r=10.0, dt=0.05,
            cap_edges=("R_min", "R_max", "r_min", "r_max"),
            cap_width_frac=0.3, cap_height=1.0,
        )
        packet = WavePacket2D(R0=5.0, r0=5.0, sigma_R=0.8, sigma_r=0.8,
                              p_R0=2.0, p_r0=1.0)
        result = prop.propagate(packet.initialize(R, r), n_steps=800,
                                save_every=100, save_density=False)
        total = result.norm_t[-1] + sum(v[-1] for v in result.absorbed.values())
        self.assertAlmostEqual(total, 1.0, places=8)
        # 角点只归因一条边: 各边损失之和 = 总阻尼损失 (自动成立),
        # 且对同一构造重复传播, 各边吸收完全一致
        result2 = prop.propagate(packet.initialize(R, r), n_steps=800,
                                 save_every=100, save_density=False)
        for e in result.absorbed:
            self.assertAlmostEqual(float(result.absorbed[e][-1]),
                                   float(result2.absorbed[e][-1]),
                                   places=12)

    def test_initial_overlap_check(self):
        mass_R, mass_r = h3_reduced_masses()
        pes = leps_jacobi_pes(LEPSBuilder())

        # 良好配置: CAP 避开初始波包
        R = np.linspace(0.5, 10.0, 192)
        r_good = np.linspace(0.2, 9.5, 144)
        prop_good = WavePacket2DPropagator(
            pes, R, r_good, mass_R, mass_r, dt=0.5,
            cap_edges=("R_min", "R_max", "r_min", "r_max"),
            cap_width_frac=0.05, cap_height=0.15,
        )
        packet = WavePacket2D(R0=6.7, r0=1.401, sigma_R=0.5,
                              sigma_r=morse_ground_width(0.1744, 1.028, mass_r))
        # 良好配置: 重叠 ~1e-6 (engine 告警阈值 1e-5, 报错阈值 1e-4)
        self.assertLess(prop_good.check_initial_overlap(
            packet.initialize(R, r_good)), 1e-5)

        # 有缺陷配置: CAP 覆盖初始波包 (v0.4.0 修复前 LEPS 的实际状态)
        r_bad = np.linspace(0.8, 8.0, 144)
        prop_bad = WavePacket2DPropagator(
            pes, R, r_bad, mass_R, mass_r, dt=0.5,
            cap_edges=("R_min", "R_max", "r_min", "r_max"),
            cap_width_frac=0.15, cap_height=0.15,
        )
        self.assertGreater(prop_bad.check_initial_overlap(
            packet.initialize(R, r_bad)), 0.9)


class TestEckartSeparable(unittest.TestCase):
    """无耦合 Eckart 模型: 2D 波包对比能量加权的 1D 参考解 (可分离体系)。

    coupling=0 时 r 方向为精确谐振子, 基态宽度初始化的波包在 r 方向
    完全静止, 2D 问题严格约化为 1D Eckart 散射。

    重要的比较语义: 波包测的是其动量分布对逐能量透射率 T(E) 的加权
    平均, 而非中心能量的点值。该参数化 (m_H 物理质量, β=1.5) 的阈值
    很宽 (T 从 E≈0.004 的 0 升到 E≈0.03 的 1), 点值与包络平均差异
    巨大 (E=0.02 处点值 0.98 vs 包络平均 0.75)。正确的参考量是
    ∫|ã(k)|² T(k²/2μ) dk — 这才是可分离体系的精确预言。
    """

    @staticmethod
    def _packet_averaged_transmission(E, packet_sigma_R, mass_R, pes1d):
        """波包动量分布加权的 1D 透射率 (细网格逐能量计算)。"""
        energies = np.linspace(0.002, 0.06, 59)
        solver = QuantumScattering1D(
            mass=mass_R, pes=pes1d, n_grid=4000,
            grid_min=0.5, grid_max=9.0,
        )
        ref = solver.solve(energies[0], energies[-1], len(energies))

        p0 = -np.sqrt(2.0 * mass_R * E)
        # |ã(k)|² ∝ exp(-σ_R²Δk²) → 动量密度标准差 σ_k = 1/(√2·σ_R)
        sigma_k = 1.0 / (np.sqrt(2.0) * packet_sigma_R)
        ks = np.linspace(p0 - 8 * sigma_k, p0 + 8 * sigma_k, 4001)
        w = np.exp(-0.5 * ((ks - p0) / sigma_k) ** 2)
        w /= np.trapezoid(w, ks)
        E_k = ks ** 2 / (2.0 * mass_R)
        T_k = np.interp(E_k, ref.energy, ref.transmission)
        return float(np.trapezoid(w * T_k, ks))

    def test_matches_energy_weighted_reference(self):
        mass_R, mass_r = h3_reduced_masses()
        params = {"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                  "r0": 1.401, "coupling": 0.0}
        pes2d = EckartBarrierPES(params)
        pes1d = lambda x: pes2d.evaluate(x, params["r0"])

        R_grid = np.linspace(0.5, 9.0, 256)
        r_grid = np.linspace(0.5, 3.5, 64)
        prop = WavePacket2DPropagator(
            pes2d.evaluate, R_grid, r_grid, mass_R, mass_r,
            dt=0.5, cap_edges=("R_min", "R_max"),
            cap_width_frac=0.15, cap_height=0.15,
        )
        packet = WavePacket2D(
            R0=6.0, r0=params["r0"], sigma_R=0.5,
            sigma_r=harmonic_ground_width(params["k_r"], mass_r),
        )
        reactant = lambda R, r: np.asarray(R) >= 2.0

        for E, n_steps in ((0.02, 5000), (0.04, 4000)):
            T_pred = self._packet_averaged_transmission(
                E, packet.sigma_R, mass_R, pes1d)

            packet.p_R0 = -np.sqrt(2.0 * mass_R * E)
            result = prop.propagate(
                packet.initialize(R_grid, r_grid), n_steps=n_steps,
                save_every=n_steps // 8,
                product_mask=eckart_product_mask(2.0),
                product_edges=("R_min",), reactant_mask=reactant,
                save_density=False,
            )
            # 互补掩码 → 通道和恒等于 1; 存活 + 全部吸收 = 1
            self.assertAlmostEqual(
                result.reaction_prob[-1] + result.reflection_prob[-1], 1.0,
                places=9, msg=f"通道概率守恒失败 (E={E})",
            )
            self.assertAlmostEqual(
                result.norm_t[-1] + sum(v[-1] for v in result.absorbed.values()),
                1.0, places=9, msg=f"总概率恒等式失败 (E={E})",
            )
            # 与能量加权的 1D 参考一致 (容差覆盖有限传播时间的慢尾)
            self.assertAlmostEqual(
                result.reaction_prob[-1], T_pred, delta=0.03,
                msg=f"2D 波包与能量加权参考偏差过大 (E={E}: "
                    f"{result.reaction_prob[-1]:.4f} vs {T_pred:.4f})",
            )


class TestLEPSWavepacket(unittest.TestCase):
    """LEPS H + H₂ 交换反应: 通道记账与阈值行为。

    该参数化为教学模型: MEP 势垒 ~0.14 au, 故低碰撞能全反射、
    足够能量 (~0.25 au) 才发生显著交换反应。数值为该模型的回归
    基线, 不是真实 H+H₂ 的物理量。
    """

    def test_threshold_behavior_and_bookkeeping(self):
        builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028,
                               "r0": 1.401, "sato": 0.05})
        pes = leps_jacobi_pes(builder)
        mass_R, mass_r = h3_reduced_masses()

        R_grid = np.linspace(0.5, 10.0, 160)
        r_grid = np.linspace(0.2, 9.5, 112)
        prop = WavePacket2DPropagator(
            pes, R_grid, r_grid, mass_R, mass_r,
            dt=0.5, cap_width_frac=0.05, cap_height=0.15,
        )
        packet = WavePacket2D(
            R0=6.7, r0=1.401, sigma_R=0.5,
            sigma_r=morse_ground_width(0.1744, 1.028, mass_r),
        )
        reactant = lambda R, r: np.asarray(R) >= 1.5 * np.asarray(r)

        # 低于 MEP 势垒 (0.139): 几乎全反射
        packet.p_R0 = -np.sqrt(2.0 * mass_R * 0.10)
        result = prop.propagate(
            packet.initialize(R_grid, r_grid), n_steps=2500,
            save_every=250,
            product_mask=leps_exchange_mask(),
            product_edges=("r_max",), reactant_mask=reactant,
            save_density=False,
        )
        self.assertAlmostEqual(
            result.reaction_prob[-1] + result.reflection_prob[-1], 1.0,
            places=9, msg="通道概率守恒失败 (E=0.10)")
        self.assertLess(result.reaction_prob[-1], 0.02,
                        "低于势垒时应几乎全反射")

        # 高于势垒: 显著交换反应
        packet.p_R0 = -np.sqrt(2.0 * mass_R * 0.25)
        result = prop.propagate(
            packet.initialize(R_grid, r_grid), n_steps=2500,
            save_every=250,
            product_mask=leps_exchange_mask(),
            product_edges=("r_max",), reactant_mask=reactant,
            save_density=False,
        )
        self.assertAlmostEqual(
            result.reaction_prob[-1] + result.reflection_prob[-1], 1.0,
            places=9, msg="通道概率守恒失败 (E=0.25)")
        self.assertGreater(result.reaction_prob[-1], 0.3,
                           "高于势垒时应发生显著反应")

    def test_energy_scan(self):
        builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028,
                               "r0": 1.401, "sato": 0.05})
        pes = leps_jacobi_pes(builder)
        mass_R, mass_r = h3_reduced_masses()

        R_grid = np.linspace(0.5, 10.0, 128)
        r_grid = np.linspace(0.2, 9.5, 96)
        prop = WavePacket2DPropagator(
            pes, R_grid, r_grid, mass_R, mass_r,
            dt=0.5, cap_width_frac=0.05, cap_height=0.15,
        )
        packet = WavePacket2D(
            R0=6.7, r0=1.401, sigma_R=0.5,
            sigma_r=morse_ground_width(0.1744, 1.028, mass_r),
        )
        scan = WavePacket2DScan(prop, packet, leps_exchange_mask(),
                                ("r_max",),
                                reactant_mask=lambda R, r: np.asarray(R) >= 1.5 * np.asarray(r),
                                n_steps=2000)
        result = scan.run(0.10, 0.30, 3)

        self.assertEqual(len(result.energy), 3)
        self.assertTrue(np.all(result.reaction_prob >= 0.0))
        self.assertTrue(np.all(result.reaction_prob <= 1.0))
        # 阈值行为: 低能反射, 高能反应
        self.assertLess(result.reaction_prob[0], 0.05)
        self.assertGreater(result.reaction_prob[-1], 0.1)
        self.assertGreater(result.reaction_prob[-1], result.reaction_prob[0])


class TestEngineIntegration(unittest.TestCase):
    """engine 全流程冒烟: Eckart 波包方法端到端。"""

    def test_eckart_wavepacket_pipeline(self):
        from autoquantum.core.engine import AutoPipeline, PipelineConfig

        with tempfile.TemporaryDirectory() as out_dir:
            config = PipelineConfig(
                system_name="H3_2D",
                pes_type="eckart",
                method="wavepacket",
                use_nn_fit=False,
                wp_n_R=96, wp_n_r=64,
                wp_n_steps=800,
                wp_scan_points=2,
                wp_save_gif=True,
                output_dir=out_dir,
            )
            pipeline = AutoPipeline(config)
            pipeline.run()

            self.assertIn("dynamics_result", pipeline._results)
            scan_result = pipeline._results["dynamics_result"]
            self.assertEqual(len(scan_result.energy), 2)
            self.assertTrue(np.all(scan_result.reaction_prob >= 0.0))
            self.assertTrue(np.all(scan_result.reaction_prob <= 1.0))

            wp = pipeline._results["wavepacket_result"]
            self.assertAlmostEqual(
                float(wp.reaction_prob[-1] + wp.reflection_prob[-1]),
                1.0, places=9)
            self.assertAlmostEqual(
                float(wp.norm_t[-1] + sum(v[-1] for v in wp.absorbed.values())),
                1.0, places=9)

            for fname in ("report.html", "transmission.png",
                          "wavepacket_snapshots.png",
                          "wavepacket_probability.png", "wavepacket.gif"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, fname)),
                                f"缺少输出文件: {fname}")


if __name__ == "__main__":
    unittest.main()
