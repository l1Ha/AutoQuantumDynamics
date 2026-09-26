"""GPU/torch 后端与 NumPy 传播子的对拍验证 (parity tests)。

torch 未安装时全部跳过; CUDA 未安装时在 CPU 上做正确性对拍
(A100 基准在集群上跑, 见 RELEASE_VALIDATION.md)。
"""

import unittest

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from autoquantum.dynamics.wavepacket_2d import (
    WavePacket2D, WavePacket2DPropagator, eckart_product_mask,
    harmonic_ground_width, h3_reduced_masses)
from autoquantum.pes.eckart import EckartBarrierPES


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestTorchParity(unittest.TestCase):
    """可分离 Eckart 基准: torch 后端 vs NumPy 传播子逐项对拍。"""

    def _make(self, dtype="float64"):
        mass_R, mass_r = h3_reduced_masses()
        params = {"V0": 0.015, "beta": 1.5, "k_r": 0.5,
                  "r0": 1.401, "coupling": 0.0}
        pes = EckartBarrierPES(params).evaluate
        R = np.linspace(0.5, 9.0, 128)
        r = np.linspace(0.5, 3.5, 48)
        np_prop = WavePacket2DPropagator(
            pes, R, r, mass_R, mass_r, dt=0.5,
            cap_edges=("R_min", "R_max"), cap_width_frac=0.15,
            cap_height=0.15)
        from autoquantum.dynamics.wavepacket_2d_torch import (
            TorchWavePacket2DPropagator)
        t_prop = TorchWavePacket2DPropagator(
            pes, R, r, mass_R, mass_r, dt=0.5,
            cap_edges=("R_min", "R_max"), cap_width_frac=0.15,
            cap_height=0.15, dtype=dtype)
        packet = WavePacket2D(R0=6.0, r0=params["r0"], sigma_R=0.5,
                              sigma_r=harmonic_ground_width(0.5, mass_r),
                              p_R0=-np.sqrt(2.0 * mass_R * 0.04))
        return np_prop, t_prop, packet, R, r

    def test_transmission_parity(self):
        np_prop, t_prop, packet, R, r = self._make()
        reactant = lambda Rr, rr: np.asarray(Rr) >= 2.0
        kwargs = dict(n_steps=600, save_every=600,
                      product_mask=eckart_product_mask(2.0),
                      product_edges=("R_min",), reactant_mask=reactant,
                      save_density=False)
        np_res = np_prop.propagate(packet.initialize(R, r), **kwargs)
        t_res = t_prop.propagate(t_prop.initialize(packet), **kwargs)
        tol = 1e-6 if t_prop.tdtype == torch.complex128 else 1e-3
        self.assertAlmostEqual(float(t_res.reaction_prob[-1]),
                               float(np_res.reaction_prob[-1]), delta=tol)
        self.assertAlmostEqual(float(t_res.reflection_prob[-1]),
                               float(np_res.reflection_prob[-1]), delta=tol)
        # 逐时刻一致
        np.testing.assert_allclose(t_res.reaction_prob,
                                   np_res.reaction_prob, atol=tol)

    def test_probability_identity_parity(self):
        np_prop, t_prop, packet, R, r = self._make()
        np_res = np_prop.propagate(packet.initialize(R, r), n_steps=200,
                                  save_every=200, save_density=False)
        t_res = t_prop.propagate(t_prop.initialize(packet), n_steps=200,
                                 save_every=200, save_density=False)
        for res in (np_res, t_res):
            total = float(res.norm_t[-1]
                          + sum(v[-1] for v in res.absorbed.values()))
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_energy_expectation_parity(self):
        np_prop, t_prop, packet, R, r = self._make()
        psi = packet.initialize(R, r)
        self.assertAlmostEqual(t_prop.energy_expectation(
            t_prop.initialize(packet)), np_prop_energy(np_prop, psi),
            places=6)


def np_prop_energy(prop, psi):
    """NumPy 版 ⟨H⟩ (与 torch 实现同一公式, 用于对拍)。"""
    kR = 2 * np.pi * np.fft.fftfreq(prop.n_R, prop.dR)[:, None]
    kr = 2 * np.pi * np.fft.fftfreq(prop.n_r, prop.dr)[None, :]
    t_k = 0.5 * (kR ** 2 / prop.mass_R + kr ** 2 / prop.mass_r)
    rho_k = np.abs(np.fft.fft2(psi)) ** 2
    rho_x = np.abs(psi) ** 2
    return (float((t_k * rho_k).sum() / rho_k.sum())
            + float((prop.V * rho_x).sum() / rho_x.sum()))


if __name__ == "__main__":
    unittest.main()
