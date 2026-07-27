import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from autoquantum.pes import PESBuilder
from autoquantum.nn import NNTrainer, TrainingConfig, FeedForwardNN
from autoquantum.dynamics import QuantumScattering1D, QuantumReaction2D
from autoquantum.visualization import DashboardGenerator
from autoquantum.pes.leps import LEPSBuilder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("AutoQuantum")


@dataclass
class PipelineConfig:
    system_name: str = "H2_1D"
    mass: float = 1.0
    pes_type: str = "morse"
    pes_params: Dict[str, float] = field(default_factory=lambda: {
        "D": 0.1744, "alpha": 1.028, "r0": 0.7416
    })
    n_pes_points: int = 100
    pes_range: List[float] = field(default_factory=lambda: [0.3, 5.0])

    use_nn_fit: bool = True
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 32])
    nn_activation: str = "tanh"
    nn_epochs: int = 500
    nn_lr: float = 0.001
    nn_train_split: float = 0.8

    energy_min: float = 0.001
    energy_max: float = 0.15
    n_energy_points: int = 100
    n_grid_points: int = 500
    grid_min: float = 0.3
    grid_max: float = 5.0

    output_dir: str = "output"
    generate_dashboard: bool = True


class AutoPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"AutoQuantum Pipeline: {self.config.system_name}")
        logger.info("=" * 60)

        self._step_build_pes()
        self._step_nn_fit()
        self._step_dynamics()
        self._step_visualize()
        return self._results

    def _step_build_pes(self):
        logger.info("[1/4] Building Potential Energy Surface ...")
        t0 = time.time()

        builder = PESBuilder(
            pes_type=self.config.pes_type,
            params=self.config.pes_params,
            system_name=self.config.system_name,
            mass=self.config.mass,
        )
        grid, values = builder.generate_grid(
            x_min=self.config.pes_range[0],
            x_max=self.config.pes_range[1],
            n_points=self.config.n_pes_points,
        )
        t1 = time.time()
        logger.info(f"    PES built with {len(grid)} points in {t1 - t0:.3f}s")

        self._results["pes_grid"] = grid
        self._results["pes_values"] = values
        self._results["pes_builder"] = builder

    def _step_nn_fit(self):
        if not self.config.use_nn_fit:
            logger.info("[2/4] NN fitting skipped.")
            return

        logger.info("[2/4] Neural Network Fitting ...")
        t0 = time.time()

        grid = self._results["pes_grid"]
        values = self._results["pes_values"]

        config = TrainingConfig(
            hidden_layers=self.config.nn_hidden_layers,
            activation=self.config.nn_activation,
            epochs=self.config.nn_epochs,
            lr=self.config.nn_lr,
            train_split=self.config.nn_train_split,
        )
        trainer = NNTrainer(config)
        model, history = trainer.train(grid, values)

        fitted_values = model.predict(grid)
        t1 = time.time()
        logger.info(f"    NN trained in {t1 - t0:.3f}s, final loss: {history['train_loss'][-1]:.6e}")

        self._results["nn_model"] = model
        self._results["nn_history"] = history
        self._results["nn_fitted_values"] = fitted_values

    def _step_dynamics(self):
        logger.info("[3/4] Quantum Dynamics Calculation ...")
        t0 = time.time()

        system = self.config.system_name.lower()

        if "h3" in system or "leps" in self.config.pes_type:
            self._step_dynamics_2d()
        else:
            self._step_dynamics_1d()

        t1 = time.time()
        logger.info(f"    Dynamics completed in {t1 - t0:.3f}s")
        if "dynamics_result" in self._results:
            r = self._results["dynamics_result"]
            logger.info(f"    Max reaction probability: {r.transmission.max():.4f}")

    def _step_dynamics_1d(self):
        model = self._results.get("nn_model")
        builder = self._results["pes_builder"]
        pes = model if model is not None else builder

        solver = QuantumScattering1D(
            mass=self.config.mass,
            pes=pes,
            n_grid=self.config.n_grid_points,
            grid_min=self.config.grid_min,
            grid_max=self.config.grid_max,
        )
        result = solver.solve(
            energy_min=self.config.energy_min,
            energy_max=self.config.energy_max,
            n_points=self.config.n_energy_points,
        )
        self._results["dynamics_result"] = result
        self._results["dynamics_dim"] = "1d"

    def _step_dynamics_2d(self):
        params = self.config.pes_params
        builder = LEPSBuilder(params)

        solver = QuantumReaction2D(
            mass_H=1.0, pes=builder.evaluate_2d,
            n_R=200, n_r=200,
            R_range=(0.5, 6.0), r_range=(0.5, 6.0),
        )
        result = solver.solve(
            energy_min=self.config.energy_min,
            energy_max=self.config.energy_max,
            n_points=self.config.n_energy_points,
        )
        self._results["dynamics_result"] = result
        self._results["dynamics_dim"] = "2d"
        self._results["pes_2d"] = builder

    def _step_visualize(self):
        logger.info("[4/4] Generating Visualizations ...")
        t0 = time.time()

        dashboard = DashboardGenerator(
            results=self._results,
            config=self.config,
        )
        dashboard.generate_all()
        t1 = time.time()
        logger.info(f"    Visualizations saved in {t1 - t0:.3f}s")

        self._results["dashboard"] = dashboard

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "AutoQuantum Pipeline Summary",
            "=" * 60,
            f"System: {self.config.system_name}",
            f"PES type: {self.config.pes_type}",
            f"NN fit: {self.config.use_nn_fit}",
        ]
        if "dynamics_result" in self._results:
            r = self._results["dynamics_result"]
            dim = self._results.get("dynamics_dim", "1d")
            lines.append(f"Dynamics: {dim.upper()}")
            lines.append(f"Energy range: {r.energy[0]:.4f} - {r.energy[-1]:.4f} au")
            lines.append(f"Max reaction probability: {r.transmission.max():.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
