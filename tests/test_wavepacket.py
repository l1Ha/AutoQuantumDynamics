import unittest
import numpy as np

from autoquantum.dynamics.wavepacket import WavePacket1D, SplitOperatorPropagator, trapezoid


class TestWavePacket1D(unittest.TestCase):
    def test_initialize_normalized_complex(self):
        wp = WavePacket1D(x0=3.0, p0=2.0, sigma=0.3)
        g = np.linspace(0, 6, 400)
        psi = wp.initialize(g)
        self.assertTrue(np.iscomplexobj(psi))
        norm = trapezoid(np.abs(psi) ** 2, g)
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_free_propagation_conserves_norm(self):
        g = np.linspace(0, 6, 512)
        prop = SplitOperatorPropagator(
            mass=100.0, pes=lambda x: np.zeros_like(x),
            grid=g, dt=0.02, cap_edges=(),
        )
        psi = WavePacket1D(x0=3.0, p0=5.0, sigma=0.3).initialize(g)
        dx = g[1] - g[0]
        n0 = np.sum(np.abs(psi) ** 2) * dx
        for _ in range(100):  # t = 2.0
            psi = prop.step(psi)
        n1 = np.sum(np.abs(psi) ** 2) * dx
        self.assertAlmostEqual(n1, n0, places=10)

        com0 = np.sum(np.abs(WavePacket1D(x0=3.0, p0=5.0, sigma=0.3)
                             .initialize(g)) ** 2 * g * dx)
        com1 = np.sum(np.abs(psi) ** 2 * g * dx)
        self.assertAlmostEqual(com1 - com0, 5.0 / 100.0 * 2.0, places=6)

    def test_cap_absorption(self):
        g = np.linspace(0, 8, 512)
        prop = SplitOperatorPropagator(
            mass=10.0, pes=lambda x: np.zeros_like(x),
            grid=g, dt=0.05, cap_width_frac=0.3, cap_height=1.0,
        )
        psi = WavePacket1D(x0=2.0, p0=3.0, sigma=0.5).initialize(g)
        dx = g[1] - g[0]
        absorbed = 0.0
        for _ in range(1500):
            psi = prop.step(psi)
            absorbed += sum(prop.last_absorbed.values())
        norm = np.sum(np.abs(psi) ** 2) * dx
        self.assertGreater(absorbed, 0.95)
        self.assertAlmostEqual(norm + absorbed, 1.0, places=8)

    def test_propagate_schedule(self):
        g = np.linspace(0, 6, 256)
        prop = SplitOperatorPropagator(
            mass=100.0, pes=lambda x: np.zeros_like(x),
            grid=g, dt=0.02, cap_edges=(),
        )
        psi = WavePacket1D(x0=3.0, p0=1.0, sigma=0.3).initialize(g)
        times, psi_all = prop.propagate(psi, n_steps=53, save_every=10)
        # 保存点: 0,10,20,30,40,50,53 — 含最后一步
        self.assertEqual(psi_all.shape[0], 7)
        self.assertAlmostEqual(times[-1], 53 * 0.02, places=12)


if __name__ == "__main__":
    unittest.main()
