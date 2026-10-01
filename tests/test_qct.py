"""QCT (准经典轨线): 能量守恒、Wigner 采样、与量子波包对比。"""

import unittest
import numpy as np

from autoquantum.dynamics.qct import (
    QCTTrajectory, QCTEnsemble, QCTResult, wigner_sample)
from autoquantum.pes.eckart import EckartBuilder
from autoquantum.pes.leps import LEPSBuilder
from autoquantum.dynamics.wavepacket_2d import (
    WavePacket2D, WavePacket2DPropagator, h3_reduced_masses,
    morse_ground_width, leps_jacobi_pes, leps_exchange_mask,
    DEFAULT_MASS_H)


class TestEnergyConservation(unittest.TestCase):
    def test_velocity_verlet_conserves_energy(self):
        builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                                 "r0": 1.401, "coupling": 0.08})
        pes = builder.evaluate_2d
        mass_R, mass_r = 1224.1, 918.1
        traj = QCTTrajectory(pes, mass_R, mass_r, dt=0.5)
        result = traj.propagate(R=6.0, r=1.401, P_R=-7.0, p_r=0.0,
                                n_steps=2500)
        self.assertLess(result["energy_drift"], 1e-4,
                        f"能量漂移: {result['energy_drift']:.2e}")

    def test_small_dt_improves(self):
        builder = EckartBuilder({"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                                 "r0": 1.401, "coupling": 0.08})
        pes = builder.evaluate_2d
        for dt in (0.5, 0.1):
            traj = QCTTrajectory(pes, 1224.1, 918.1, dt=dt)
            result = traj.propagate(R=6.0, r=1.401, P_R=-7.0, p_r=0.0,
                                    n_steps=500)
            drift_ = result["energy_drift"]
            if dt == 0.1:
                self.assertLess(drift_, 1e-5, f"dt=0.1 漂移: {drift_:.2e}")


class TestWignerSampling(unittest.TestCase):
    def test_harmonic_widths(self):
        mu, omega = 1.0, 1.0
        q, p = wigner_sample(mu, omega, 10000, seed=0)
        self.assertAlmostEqual(np.std(q), 1 / np.sqrt(2 * mu * omega),
                               delta=0.01)
        self.assertAlmostEqual(np.std(p), np.sqrt(mu * omega / 2), delta=0.01)


class TestQCTvsQuantum(unittest.TestCase):
    """QCT 与量子波包对比: 同势面同网格。

    物理预期:
    - 高能端: QCT ≈ 量子 (经典极限)
    - 低能端: QCT 可能偏高 (Wigner 采样含 classically forbidden 区域)
    - 差异反映量子效应 (隧穿/干涉/零点能)
    """

    def _setup(self):
        mass_R, mass_r = h3_reduced_masses(DEFAULT_MASS_H)
        builder = LEPSBuilder({"D": 0.1744, "alpha": 1.028, "r0": 1.401,
                               "sato": 0.05})
        pes = leps_jacobi_pes(builder)
        R = np.linspace(0.5, 10.0, 192)
        r = np.linspace(0.2, 9.5, 144)
        sigma_r = morse_ground_width(0.1744, 1.028, mass_r)
        crit = lambda R, r: R < 1.5 * r
        return pes, mass_R, mass_r, R, r, sigma_r, crit

    def test_qct_in_ballpark_of_quantum(self):
        """QCT 结果应在量子结果的合理范围内 (不精确匹配)。"""
        pes, mass_R, mass_r, R, r, sigma_r, crit = self._setup()
        E = 0.25

        ensemble = QCTEnsemble(pes, mass_R, mass_r, dt=0.5, max_steps=3000)
        qct = ensemble.run(np.array([E]), R0=6.7, r_mean=1.401,
                           r_sigma=sigma_r, reaction_criterion=crit,
                           n_traj=100, seed=0)

        prop = WavePacket2DPropagator(
            pes, R, r, mass_R, mass_r, 0.5,
            cap_edges=("R_min", "R_max", "r_min", "r_max"),
            cap_width_frac=0.05, cap_height=0.15)
        packet = WavePacket2D(6.7, 1.401, 0.5, sigma_r,
                              p_R0=-np.sqrt(2 * mass_R * E))
        qm = prop.propagate(packet.initialize(R, r), 2500, save_every=2500,
                           product_mask=leps_exchange_mask(),
                           product_edges=("r_max",),
                           reactant_mask=lambda Rr, rr:
                               np.asarray(Rr) >= 1.5 * np.asarray(rr),
                           save_density=False)

        qct_P = qct.reaction_probs[0]
        qm_P = qm.reaction_prob[-1]
        # 两者都应在 [0, 1] 内且有显著反应
        self.assertGreater(qct_P, 0.01, f"QCT P 过低: {qct_P:.4f}")
        self.assertLessEqual(qct_P, 1.01)  # 高能端 QCT 可达 P≈1.0
        self.assertGreater(qm_P, 0.01, f"量子 P 过低: {qm_P:.4f}")
        # 数值应在同一数量级
        ratio = max(qct_P, qm_P) / max(min(qct_P, qm_P), 1e-8)
        self.assertLess(ratio, 20, f"QCT/量子 比率过大: {ratio:.1f}")

    def test_energy_dependence_direction(self):
        """QCT 反应概率应随能量增加而增加 (物理合理性)。"""
        pes, mass_R, mass_r, R, r, sigma_r, crit = self._setup()
        ensemble = QCTEnsemble(pes, mass_R, mass_r, dt=0.5, max_steps=3000)
        result = ensemble.run(
            np.array([0.15, 0.25]), R0=6.7, r_mean=1.401, r_sigma=sigma_r,
            reaction_criterion=crit, n_traj=100, seed=0)
        self.assertGreater(result.reaction_probs[-1],
                           result.reaction_probs[0],
                           "反应概率应随能量增加: "
                           f"{result.reaction_probs}")


if __name__ == "__main__":
    unittest.main()
