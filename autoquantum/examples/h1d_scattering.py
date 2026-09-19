#!/usr/bin/env python3
"""1D H2 Quantum Scattering Example — Full AutoQuantum Pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from autoquantum.core.engine import AutoPipeline, PipelineConfig


def run_h2_1d():
    config = PipelineConfig(
        system_name="H2_1D_Morse",
        mass=1.0,
        pes_type="morse",
        pes_params={"D": 0.1744, "alpha": 1.028, "r0": 0.7416},
        n_pes_points=200,
        pes_range=[0.3, 5.0],
        use_nn_fit=True,
        nn_hidden_layers=[64, 64, 32],
        nn_activation="tanh",
        nn_epochs=300,
        nn_lr=0.001,
        energy_min=0.001,
        energy_max=0.25,
        n_energy_points=100,
        n_grid_points=1000,
        grid_min=0.3,
        grid_max=5.0,
        output_dir="output_h2_1d",
        generate_dashboard=True,
    )

    pipeline = AutoPipeline(config)
    results = pipeline.run()

    print()
    print(pipeline.summary())
    return results


if __name__ == "__main__":
    run_h2_1d()
