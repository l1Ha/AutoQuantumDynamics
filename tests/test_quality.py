"""工程质量测试: 输入验证、传播健康诊断、能量监测、训练卡、
实验模块护栏、运行清单。"""

import json
import os
import tempfile
import unittest
import numpy as np

from autoquantum.core.validation import (
    ValidationError, validate_grid, validate_energy_window,
    validate_pes_values, validate_positive, check_propagation,
    suggest_dt, data_fingerprint, environment_info, write_run_manifest,
)


class TestInputValidation(unittest.TestCase):
    def test_grid_accepts_uniform(self):
        g = validate_grid("x", np.linspace(0, 1, 11))
        self.assertEqual(g.size, 11)

    def test_grid_rejects_bad_inputs(self):
        with self.assertRaises(ValidationError):
            validate_grid("x", [0, 1, 0.5])          # 非单调
        with self.assertRaises(ValidationError):
            validate_grid("x", [0, 0.1, 0.5])        # 非等距
        with self.assertRaises(ValidationError):
            validate_grid("x", [0, np.nan])          # NaN
        with self.assertRaises(ValidationError):
            validate_grid("x", [0.0])                # 点数不足
        with self.assertRaises(ValidationError):
            validate_grid("x", np.zeros((2, 5)))     # 非 1D

    def test_energy_window(self):
        self.assertEqual(validate_energy_window(0.01, 0.1), (0.01, 0.1))
        with self.assertRaises(ValidationError):
            validate_energy_window(0.1, 0.01)        # 倒序
        with self.assertRaises(ValidationError):
            validate_energy_window(-0.01, 0.1)       # 负能

    def test_pes_finiteness_and_range(self):
        validate_pes_values("V", np.linspace(-1, 1, 100))
        with self.assertRaises(ValidationError):
            validate_pes_values("V", np.array([0.0, np.nan]))
        with self.assertRaises(ValidationError):
            validate_pes_values("V", np.array([0.0, 1e12]))  # 发散墙

    def test_positive(self):
        self.assertEqual(validate_positive("m", 2.0), 2.0)
        self.assertEqual(validate_positive("m", 0.0, allow_zero=True), 0.0)
        with self.assertRaises(ValidationError):
            validate_positive("m", -1.0)


class TestPropagationHealth(unittest.TestCase):
    def _fake_result(self, norm_final, absorbed, react, refl,
                     nan=False):
        class R:
            pass
        r = R()
        r.norm_t = np.array([1.0, norm_final])
        r.absorbed = {"R_max": np.array([0.0, absorbed])}
        r.reaction_prob = np.array([0.0, react])
        r.reflection_prob = np.array([0.0, refl])
        if nan:
            r.norm_t = np.array([1.0, np.nan])
        return r

    def test_healthy(self):
        h = check_propagation(self._fake_result(1e-6, 1 - 1e-6, 0.5, 0.5))
        self.assertTrue(h.ok, h.warnings)
        self.assertAlmostEqual(h.channel_sum, 1.0)

    def test_unabsorbed_warns(self):
        h = check_propagation(self._fake_result(0.3, 0.2, 0.5, 0.5))
        self.assertFalse(h.ok)
        self.assertTrue(any("留在网格" in w for w in h.warnings))

    def test_identity_violation_warns(self):
        h = check_propagation(self._fake_result(1e-6, 0.4, 0.4, 0.4))
        self.assertFalse(h.ok)
        self.assertTrue(any("记账" in w for w in h.warnings))

    def test_nan_warns(self):
        h = check_propagation(self._fake_result(0, 0, 0, 0, nan=True))
        self.assertTrue(h.nan_detected)

    def test_energy_drift_threshold(self):
        h = check_propagation(self._fake_result(1e-6, 0.5, 0.5, 0.5),
                               energy_drift_rel=0.5)
        self.assertTrue(any("漂移" in w for w in h.warnings))


class TestEnergyTracking(unittest.TestCase):
    def test_free_propagation_energy_conserved(self):
        from autoquantum.dynamics.wavepacket_2d import (
            WavePacket2D, WavePacket2DPropagator)
        R = np.linspace(-8, 8, 64)
        r = np.linspace(-8, 8, 64)
        prop = WavePacket2DPropagator(
            lambda R, r: np.zeros_like(R), R, r, mass_R=1.0, mass_r=1.0,
            dt=0.1, cap_edges=())
        packet = WavePacket2D(-3.0, 0.0, 0.7, 0.7, p_R0=2.0)
        res = prop.propagate(packet.initialize(R, r), n_steps=40,
                             save_every=20, track_energy=True)
        self.assertIsNotNone(res.energy_track)
        self.assertIsNotNone(res.energy_drift_rel)
        # 自由演化时分裂算符精确 (T 已在 k 空间对角), 能量应守恒到舍入
        self.assertLess(res.energy_drift_rel, 1e-8)
        # 动能期望 = 0.5·⟨p_R²⟩ + 0.5·⟨p_r²⟩; 高斯包
        # var(k) = 1/(2σ²) (振幅宽度 σ), 平面波中心 p0
        sigma = 0.7
        expected = 0.5 * (2.0 ** 2 + 1 / (2 * sigma ** 2)) \
            + 0.5 * (0.0 ** 2 + 1 / (2 * sigma ** 2))
        self.assertAlmostEqual(res.energy_reference, expected, delta=1e-6)


class TestReproducibility(unittest.TestCase):
    def test_fingerprint_changes_with_data(self):
        a = np.arange(10.0)
        b = a.copy()
        self.assertEqual(data_fingerprint(a), data_fingerprint(b))
        b[0] += 1
        self.assertNotEqual(data_fingerprint(a), data_fingerprint(b))

    def test_environment_info(self):
        info = environment_info()
        for key in ("python", "platform", "numpy"):
            self.assertIn(key, info)
        self.assertIn("autoquantum", info)

    def test_run_manifest(self):
        class Cfg:
            system_name = "test"
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run_manifest(os.path.join(tmp, "m.json"), Cfg(),
                                      {"answer": 42})
            data = json.load(open(path, encoding="utf-8"))
        self.assertEqual(data["results"]["answer"], 42)
        self.assertEqual(data["config"]["system_name"], "test")
        self.assertIn("numpy", data["environment"])


class TestSuggestDt(unittest.TestCase):
    def test_shrinks_with_potential(self):
        V_small = np.full((32, 32), 0.01)
        V_big = np.full((32, 32), 10.0)
        dt_small = suggest_dt(V_small, 0.1, 0.1, 1000.0, 1000.0)
        dt_big = suggest_dt(V_big, 0.1, 0.1, 1000.0, 1000.0)
        self.assertGreater(dt_small, dt_big)
        self.assertGreater(dt_big, 0)


class TestTrainingCard(unittest.TestCase):
    def test_card_populated_and_persisted(self):
        from autoquantum.nn import NNTrainer, TrainingConfig
        from autoquantum.nn.model import PESNN
        rng = np.random.RandomState(0)
        X = rng.uniform(-1, 1, (60, 2))
        y = np.sin(2 * X[:, 0]) + 0.2 * X[:, 1]
        model, _ = NNTrainer(TrainingConfig(hidden_layers=[16], epochs=30,
                                           seed=0)).train(X, y)
        card = model.training_card
        self.assertIn("data_sha256", card)
        self.assertEqual(len(card["data_sha256"]), 64)
        self.assertIn("rmse_full", card)
        self.assertIn("hyperparameters", card)
        self.assertIn("autoquantum", card["environment"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pkl")
            model.save(path)
            loaded = PESNN.load(path)
        self.assertEqual(loaded.training_card["data_sha256"],
                         card["data_sha256"])
        np.testing.assert_allclose(loaded.predict(X), model.predict(X),
                                   rtol=1e-12)


class TestEngineGuards(unittest.TestCase):
    def test_experimental_stationary_requires_optin(self):
        from autoquantum.core.engine import AutoPipeline, PipelineConfig
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(
                system_name="H3_2D", pes_type="eckart",
                method="stationary", use_nn_fit=False,
                n_energy_points=4, energy_min=0.01, energy_max=0.04,
                allow_experimental=False,
                output_dir=tmp, generate_dashboard=False)
            with self.assertRaises(ValidationError):
                AutoPipeline(cfg).run()

    def test_run_writes_manifest_and_health(self):
        from autoquantum.core.engine import AutoPipeline, PipelineConfig
        with tempfile.TemporaryDirectory() as tmp:
            # 1D 流程: 清单
            cfg = PipelineConfig(
                system_name="H2_1D", pes_type="morse", method="stationary",
                use_nn_fit=False, n_energy_points=5, energy_min=0.01,
                energy_max=0.05, seed=1234,
                output_dir=tmp, generate_dashboard=False)
            results = AutoPipeline(cfg).run()
            manifest = json.load(open(os.path.join(tmp, "run_manifest.json"),
                                     encoding="utf-8"))
            self.assertEqual(manifest["config"]["seed"], 1234)
            self.assertIn("max_transmission", manifest["results"])
            self.assertIn("run_manifest", results)

            # 2D 波包: 健康诊断
            cfg2 = PipelineConfig(
                system_name="H3_2D", pes_type="eckart", method="wavepacket",
                use_nn_fit=False, wp_n_R=64, wp_n_r=48,
                wp_n_steps=400, wp_scan_points=2, wp_save_gif=False,
                wp_health_check=True, seed=7,
                output_dir=tmp, generate_dashboard=False)
            r2 = AutoPipeline(cfg2).run()
            health = r2["wp_health"]
            self.assertIn("wp_health", r2)
            self.assertIsInstance(health.warnings, list)
            self.assertIn("pes_2d_data_hash", r2)
            self.assertIsNotNone(r2["wavepacket_result"].energy_track)

    def test_pes_blowup_fails_fast(self):
        # NN 面爆炸时有限性门控必须立即失败, 而不是传播出 NaN
        from autoquantum.dynamics.wavepacket_2d import (
            WavePacket2DPropagator, WavePacket2D)
        R = np.linspace(0.5, 9.0, 32)
        r = np.linspace(0.5, 3.5, 32)
        bad = lambda Rr, rr: np.full(np.broadcast(Rr, rr).shape, np.nan)
        prop = WavePacket2DPropagator(bad, R, r, 1.0, 1.0, 0.1,
                                      cap_edges=())
        with self.assertRaises(ValidationError):
            validate_pes_values("bad", prop.V)


if __name__ == "__main__":
    unittest.main()
