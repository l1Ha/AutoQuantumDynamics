import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import numpy as np

from autoquantum.pes import PESBuilder
from autoquantum.pes.abinitio import AbInitioData
from autoquantum.nn import NNTrainer, TrainingConfig, FeedForwardNN
from autoquantum.dynamics import (
    QuantumScattering1D,
    QuantumReaction2D,
    WavePacket1D,
    SplitOperatorPropagator,
    WavePacket2D,
    WavePacket2DPropagator,
    WavePacket2DScan,
    MultiWidthScanResult,
    multi_width_scan,
    deconvolve_reaction,
    check_convergence,
    DEFAULT_MASS_H,
    h3_reduced_masses,
    leps_jacobi_pes,
    leps_exchange_mask,
    eckart_product_mask,
    morse_ground_width,
    harmonic_ground_width,
    exchange_dividing_surface_line,
    TransmissionProbability,
    ReflectionProbability,
)
from autoquantum.visualization import DashboardGenerator
from autoquantum.pes.leps import LEPSBuilder
from autoquantum.pes.eckart import EckartBuilder

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

    # 动力学方法: "auto" 按体系选择 (2D→含时波包, 1D→时间无关),
    # "stationary" 强制时间无关扫描, "wavepacket" 强制含时波包
    method: str = "auto"

    use_nn_fit: bool = True
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 32])
    nn_activation: str = "tanh"
    nn_epochs: int = 500
    nn_lr: float = 0.005
    nn_train_split: float = 0.8
    # >0 时对训练网格等距子采样到该点数 (多维网格加速)
    nn_max_train_points: int = 6000
    # 力训练权重 (dY = ∂V/∂x 监督目标)。力目标同时正则化能量拟合:
    # Eckart 2D 上 V_RMSE 4.6e-3 → ~1e-3, 梯度 RMSE 2.2e-2 → ~4e-3
    nn_force_weight: float = 1.0

    energy_min: float = 0.001
    energy_max: float = 0.15
    n_energy_points: int = 100
    n_grid_points: int = 500
    grid_min: float = 0.3
    grid_max: float = 5.0

    # 二维含时波包参数 (物理单位: ħ=1, Hartree, Bohr, m_H 以电子质量计)
    mass_H: float = DEFAULT_MASS_H
    wp_n_R: int = 192
    wp_n_r: int = 144
    wp_dt: float = 0.5
    wp_n_steps: int = 3000
    wp_sigma_R: float = 0.5
    wp_scan_points: int = 6
    wp_save_gif: bool = True
    # 能量反卷积: 多宽度扫描 + 曲率正则最小二乘恢复 T(E)
    # (病态反问题 — 仅在传播充分且残差小时可信; 运行时记录残差)
    wp_deconvolve: bool = False
    # 传播收敛检查: 基线 vs 加密设置, 记录 |ΔP_react|
    wp_check_convergence: bool = False

    output_dir: str = "output"
    generate_dashboard: bool = True


class AutoPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._results: Dict[str, Any] = {}

    @property
    def is_2d(self) -> bool:
        return ("h3" in self.config.system_name.lower()
                or self.config.pes_type in ("leps", "eckart"))

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"AutoQuantum Pipeline: {self.config.system_name} "
                    f"(method={self.config.method})")
        logger.info("=" * 60)

        self._step_build_pes()
        self._step_nn_fit()
        self._step_dynamics()
        self._step_visualize()
        return self._results

    # ------------------------------------------------------------------
    def _step_build_pes(self):
        logger.info("[1/4] Building Potential Energy Surface ...")
        t0 = time.time()

        if self.is_2d:
            self._step_build_pes_2d()
        else:
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
            self._results["pes_grid"] = grid
            self._results["pes_values"] = values
            self._results["pes_builder"] = builder
            logger.info(f"    PES built with {len(grid)} points in {time.time() - t0:.3f}s")

        if "pes_2d" in self._results:
            logger.info(f"    2D PES built in {time.time() - t0:.3f}s")

    def _step_build_pes_2d(self):
        """二维体系直接构建 2D PES (LEPS / Eckart)。

        网格范围与含时波包动力学域一致, 保证 NN 拟合 (若启用) 不外推。
        """
        pes_type = self.config.pes_type
        if pes_type not in ("leps", "eckart"):
            pes_type = self.config.pes_type = "eckart"
            logger.info("    2D system: defaulting PES to 'eckart'")

        if pes_type == "eckart":
            builder = EckartBuilder(self.config.pes_params)
            R_range, r_range = (0.5, 9.0), (0.5, 3.5)
        else:
            builder = LEPSBuilder(self.config.pes_params)
            R_range, r_range = (0.5, 10.0), (0.2, 9.5)

        R_grid, r_grid, V_grid = builder.generate_grid(
            R_range, r_range, n_R=150, n_r=150,
        )
        self._results["pes_2d"] = builder
        self._results["pes_2d_grid"] = (R_grid, r_grid, V_grid)
        self._results["pes_2d_ranges"] = (R_range, r_range)

    def _step_nn_fit(self):
        if not self.config.use_nn_fit:
            logger.info("[2/4] NN fitting skipped.")
            return
        if self.is_2d:
            self._step_nn_fit_2d()
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
            max_train_points=self.config.nn_max_train_points,
        )
        trainer = NNTrainer(config)
        model, history = trainer.train(grid, values)

        fitted_values = model.predict(grid)
        t1 = time.time()
        logger.info(f"    NN trained in {t1 - t0:.3f}s, final loss: {history['train_loss'][-1]:.6e}")

        self._results["nn_model"] = model
        self._results["nn_history"] = history
        self._results["nn_fitted_values"] = fitted_values

    def _step_nn_fit_2d(self):
        """二维 PES 的 NN 拟合 (网格域 = 波包动力学域, 无外推)。

        通过 ``AbInitioData.sample_function`` 生成带中心差分梯度的
        训练集; ``nn_force_weight > 0`` 时启用力训练 (以 ∂V/∂x 为
        监督目标), 显著提升代理面的梯度保真度。
        """
        grid = self._results.get("pes_2d_grid")
        if grid is None:
            logger.info("[2/4] NN fitting skipped (no 2D grid).")
            return

        logger.info("[2/4] Neural Network Fitting (2D PES) ...")
        t0 = time.time()

        R_range, r_range = self._results.get(
            "pes_2d_ranges", ((0.5, 9.0), (0.5, 6.0)))
        builder = self._results["pes_2d"]

        data = AbInitioData.sample_function(
            lambda pts: np.asarray(builder.evaluate_2d(pts[:, 0], pts[:, 1]),
                                   dtype=float).ravel(),
            ranges=[R_range, r_range], n_per_dim=150,
        )
        X, y = data.points, data.energies

        config = TrainingConfig(
            hidden_layers=self.config.nn_hidden_layers,
            activation=self.config.nn_activation,
            epochs=self.config.nn_epochs,
            lr=self.config.nn_lr,
            train_split=self.config.nn_train_split,
            max_train_points=self.config.nn_max_train_points,
            force_weight=self.config.nn_force_weight,
        )
        trainer = NNTrainer(config)
        model, history = trainer.train(
            X, y, dY=data.gradients if self.config.nn_force_weight > 0 else None)

        pred = model.predict(X)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        grad_rmse = float(np.sqrt(np.mean(
            (model.gradient(X) - data.gradients) ** 2)))
        v_span = float(y.max() - y.min())
        t1 = time.time()
        logger.info(f"    2D NN trained in {t1 - t0:.3f}s, "
                    f"RMSE = {rmse:.3e} au ({rmse / v_span * 100:.2f}% of V span), "
                    f"grad RMSE = {grad_rmse:.3e} au/Bohr"
                    + (f" (force_weight={self.config.nn_force_weight})"
                       if self.config.nn_force_weight > 0 else ""))
        if rmse > 0.02 * v_span:
            logger.warning("    NN 拟合 RMSE > 2%·V_span, 拟合面动力学结果不可靠; "
                           "建议增大 nn_epochs/nn_hidden_layers 或关闭 use_nn_fit")

        self._results["nn_model"] = model
        self._results["nn_history"] = history
        self._results["nn_fit_rmse"] = rmse
        self._results["nn_fit_grad_rmse"] = grad_rmse

    @staticmethod
    def _nn_pes_2d(model):
        """把多维 PESNN 包装为波包传播子需要的 V(R, r) 闭包。"""
        def V(R, r):
            Rb = np.asarray(R, dtype=float)
            rb = np.asarray(r, dtype=float)
            Rb, rb = np.broadcast_arrays(Rb, rb)
            pts = np.column_stack([Rb.ravel(), rb.ravel()])
            return model.predict(pts).reshape(Rb.shape)
        return V

    # ------------------------------------------------------------------
    def _step_dynamics(self):
        logger.info("[3/4] Quantum Dynamics Calculation ...")
        t0 = time.time()

        method = self.config.method
        if method == "auto":
            method = "wavepacket" if self.is_2d else "stationary"

        if method == "wavepacket":
            if self.is_2d:
                self._step_dynamics_2d_wavepacket()
            else:
                self._step_dynamics_1d_wavepacket()
        elif method == "stationary":
            if self.is_2d:
                logger.warning("    2D 时间无关求解器为实验性 (无通道耦合/未做流归一化), "
                               "结果仅作定性参考")
                self._step_dynamics_2d()
            else:
                self._step_dynamics_1d()
        else:
            raise ValueError(f"未知动力学方法: {method!r} "
                             "(可选 auto / stationary / wavepacket)")

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
        """时间无关: 定能量扫描 (实验性, 见 quantum_2d 模块警告)。"""
        pes_type = self.config.pes_type if self.config.pes_type in ("leps", "eckart") else "eckart"
        if pes_type == "eckart":
            builder = EckartBuilder(self.config.pes_params)
        else:
            builder = LEPSBuilder(self.config.pes_params)

        solver = QuantumReaction2D(
            mass_H=self.config.mass_H, pes=builder.evaluate_2d,
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

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_energy_window(e_min: float, e_max: float,
                             lo: float, hi: float, tag: str):
        """把请求窗口截断到模型适用范围并保证升序; 越界时记录日志。"""
        a, b = max(e_min, lo), min(e_max, hi)
        if a >= b:
            logger.warning(f"    {tag}: 请求能量窗口 [{e_min}, {e_max}] 与模型适用"
                           f"范围 [{lo}, {hi}] 不相交, 使用默认范围")
            return lo, hi
        if (a, b) != (e_min, e_max):
            logger.info(f"    {tag}: 能量窗口调整为 [{a:.4f}, {b:.4f}] au "
                        f"(模型适用范围 [{lo}, {hi}])")
        return a, b

    def _step_dynamics_2d_wavepacket(self):
        """二维含时波包: 能量扫描 + 单能量含时演化演示。"""
        cfg = self.config
        M = cfg.mass_H
        mass_R, mass_r = h3_reduced_masses(M)

        if cfg.pes_type == "eckart":
            builder = EckartBuilder(cfg.pes_params)
            pes = builder.evaluate_2d
            R_range, r_range = (0.5, 9.0), (0.5, 3.5)
            R0 = 6.0
            # 入口通道振动平衡位置: 耦合项使 r 平衡随 R 偏移
            k_r = cfg.pes_params.get("k_r", 0.5)
            coupling = cfg.pes_params.get("coupling", 0.08)
            beta = cfg.pes_params.get("beta", 1.5)
            r_eq = cfg.pes_params.get("r0", 1.401) - coupling / k_r * np.tanh(
                beta * (R0 - 3.0))
            r0 = r_eq
            sigma_r = harmonic_ground_width(k_r, mass_r)
            product_mask = eckart_product_mask(2.0)
            product_edges = ("R_min",)
            reactant_mask = lambda R, r: np.asarray(R) >= 2.0
            ds_line = ([2.0, 2.0], [r_range[0], r_range[1]])
            e_min, e_max = self._clamp_energy_window(
                cfg.energy_min, cfg.energy_max, 0.002, 0.05, "Eckart")
            cap_edges = ("R_min", "R_max")
            cap_frac = 0.15
        else:
            builder = LEPSBuilder(cfg.pes_params)
            pes = leps_jacobi_pes(builder)
            R_range, r_range = (0.5, 10.0), (0.2, 9.5)
            R0 = 6.7
            r0 = cfg.pes_params.get("r0", 1.401)
            sigma_r = morse_ground_width(
                cfg.pes_params.get("D", 0.1744),
                cfg.pes_params.get("alpha", 1.028), mass_r)
            product_mask = leps_exchange_mask()
            product_edges = ("r_max",)
            reactant_mask = lambda R, r: np.asarray(R) >= 1.5 * np.asarray(r)
            ds_line = exchange_dividing_surface_line(np.linspace(0.5, 10.0, 2))
            # 该 LEPS 参数化交换势垒 ~0.14 au (MEP); 窗口覆盖阈值两侧
            e_min, e_max = self._clamp_energy_window(
                cfg.energy_min, cfg.energy_max, 0.10, 0.35, "LEPS")
            # CAP 必须避开初始波包 (r0≈1.4): r_min CAP 区 [0.2, ~0.65]
            cap_edges = ("R_min", "R_max", "r_min", "r_max")
            cap_frac = 0.05

        # 数据驱动管线: 若启用 NN 拟合, 动力学运行在 NN 代理面上
        nn_model = self._results.get("nn_model")
        if nn_model is not None:
            pes = self._nn_pes_2d(nn_model)
            logger.info(f"    Dynamics on NN-fitted PES "
                        f"(RMSE = {self._results.get('nn_fit_rmse', float('nan')):.3e} au)")

        R_grid = np.linspace(*R_range, cfg.wp_n_R)
        r_grid = np.linspace(*r_range, cfg.wp_n_r)
        prop = WavePacket2DPropagator(
            pes, R_grid, r_grid, mass_R, mass_r, dt=cfg.wp_dt,
            cap_edges=cap_edges, cap_width_frac=cap_frac, cap_height=0.15,
        )
        packet = WavePacket2D(R0=R0, r0=r0,
                              sigma_R=cfg.wp_sigma_R, sigma_r=sigma_r)

        # 初态-CAP 重叠检查: 过大说明网格/CAP 配置不当, 结果不可信
        overlap = prop.check_initial_overlap(packet.initialize(R_grid, r_grid))
        if overlap > 1e-4:
            raise RuntimeError(
                f"初始波包与 CAP 重叠概率 {overlap:.2e} 过大 (>1e-4): "
                "请增大网格范围或减小 cap_width_frac")
        if overlap > 1e-5:
            logger.warning(f"    初始波包与 CAP 重叠概率 {overlap:.2e} (建议 <1e-5)")

        logger.info(f"    2D wavepacket: grid {cfg.wp_n_R}x{cfg.wp_n_r}, "
                    f"dt={cfg.wp_dt}, steps={cfg.wp_n_steps}")

        # 能量扫描 → 与时间无关扫描接口兼容
        scan = WavePacket2DScan(
            prop, packet, product_mask, product_edges,
            reactant_mask=reactant_mask,
            n_steps=cfg.wp_n_steps,
        )
        scan_result = scan.run(e_min, e_max, cfg.wp_scan_points)

        # 取扫描区间偏上能量做单能量含时演化演示 (波包越过垒的过程更完整)
        E_mid = float(e_min + 0.75 * (e_max - e_min))
        packet.p_R0 = -np.sqrt(2.0 * mass_R * E_mid)
        psi0 = packet.initialize(R_grid, r_grid)
        save_every = max(1, cfg.wp_n_steps // 12)
        wp_result = prop.propagate(
            psi0, cfg.wp_n_steps, save_every=save_every,
            product_mask=product_mask, product_edges=product_edges,
            reactant_mask=reactant_mask, save_density=True,
        )
        wp_result.energy = np.array([E_mid])

        self._results["dynamics_result"] = scan_result
        self._results["dynamics_dim"] = "2d"
        self._results["pes_2d"] = builder
        self._results["wavepacket_result"] = wp_result
        self._results["wp_ds_line"] = ds_line
        self._results["wp_final_reaction"] = wp_result.final_reaction_probability
        logger.info(f"    Single-packet E={E_mid:.4f} au: "
                    f"P_react={wp_result.final_reaction_probability:.4f}")

        # 传播收敛检查 (基线 vs 加密网格/时间步)
        if cfg.wp_check_convergence:
            def make_prop(n_R, n_r, dt):
                return WavePacket2DPropagator(
                    pes, np.linspace(*R_range, n_R), np.linspace(*r_range, n_r),
                    mass_R, mass_r, dt, cap_edges=cap_edges,
                    cap_width_frac=cap_frac, cap_height=0.15)

            conv = check_convergence(
                make_prop, packet, cfg.wp_n_R, cfg.wp_n_r, cfg.wp_dt,
                cfg.wp_n_steps, product_mask, product_edges,
                reactant_mask=reactant_mask)
            self._results["wp_convergence"] = conv
            logger.info(f"    Convergence check: P_react "
                        f"{conv['baseline_p_react']:.4f} -> "
                        f"{conv['refined_p_react']:.4f} "
                        f"(Δ={conv['abs_diff']:.4f})")
            if conv["abs_diff"] > 0.02:
                logger.warning("    加密设置下 P_react 变化 >0.02: 传播未收敛, "
                               "定量使用前请增大 wp_n_steps/网格")

        # 能量反卷积 (多宽度扫描 → 曲率正则恢复 T(E))
        if cfg.wp_deconvolve:
            logger.info("    Energy deconvolution (multi-width scan) ...")
            sigmas = (0.4, 0.7, 1.1)
            mw = multi_width_scan(
                prop, packet, product_mask, product_edges,
                e_min, e_max, max(5, cfg.wp_scan_points),
                sigmas=sigmas, reactant_mask=reactant_mask,
                n_steps=cfg.wp_n_steps)
            E_dec, T_dec, cond, residual = deconvolve_reaction(mw, mass_R)
            self._results["wp_deconv"] = (E_dec, T_dec)
            self._results["wp_deconv_meta"] = {"cond": cond,
                                              "residual": residual,
                                              "sigmas": sigmas}
            logger.info(f"    Deconvolved T(E): cond(A)={cond:.2e}, "
                        f"relative residual={residual:.4f}")
            if residual > 0.1:
                logger.warning("    反卷积残差 >0.1 (数据与包络模型失配或病态): "
                               "T(E) 曲线仅作诊断, 建议增大 wp_n_steps/扩展能量窗口")

    def _step_dynamics_1d_wavepacket(self):
        """一维含时波包 (Split-Operator + CAP)。"""
        cfg = self.config
        builder = self._results["pes_builder"]
        model = self._results.get("nn_model")
        pes_impl = model if model is not None else builder

        grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.n_grid_points)

        def pes(x):
            return np.asarray(pes_impl(x), dtype=float)

        E0 = 0.5 * (cfg.energy_min + cfg.energy_max)
        x0 = cfg.grid_max - 0.25 * (cfg.grid_max - cfg.grid_min)
        V0 = float(pes(np.array([x0]))[0])
        if E0 <= V0:
            E0 = V0 + 0.05
        p0 = -np.sqrt(2.0 * cfg.mass * (E0 - V0))

        packet = WavePacket1D(x0=x0, p0=p0, sigma=0.3)
        prop = SplitOperatorPropagator(
            mass=cfg.mass, pes=pes, grid=grid, dt=0.05,
            cap_width_frac=0.15, cap_height=2.0,
        )
        psi0 = packet.initialize(grid)

        n_steps = 2000
        save_every = n_steps // 12
        save_steps = list(range(0, n_steps + 1, save_every))
        if save_steps[-1] != n_steps:
            save_steps.append(n_steps)
        save_index = {s: k for k, s in enumerate(save_steps)}

        psi = psi0.copy()
        cum = {"left": 0.0, "right": 0.0}
        absorbed_hist = {e: np.zeros(len(save_steps)) for e in cum}
        psi_all = np.zeros((len(save_steps), grid.size), dtype=complex)

        def record(k, psi):
            psi_all[k] = psi
            for e in cum:
                absorbed_hist[e][k] = cum[e]

        record(0, psi)
        for i in range(1, n_steps + 1):
            psi = prop.step(psi)
            for e, d in prop.last_absorbed.items():
                cum[e] += d
            if i in save_index:
                record(save_index[i], psi)

        times = np.array(save_steps, dtype=float) * prop.dt
        # 波包自右侧出发向左传播 (p0 < 0): 透射 = 越过垒心到达左侧,
        # 反射 = 返回右侧。区域布居 + 对应侧吸收量。
        barrier_center = float(cfg.pes_params.get("r0", 0.74))
        trans = np.array([TransmissionProbability(p, grid, barrier_center)
                          for p in psi_all]) + absorbed_hist["left"]
        reflect = np.array([ReflectionProbability(p, grid, barrier_center)
                            for p in psi_all]) + absorbed_hist["right"]

        from autoquantum.core.base import DynamicsResult
        result = DynamicsResult(
            energy=np.array([E0]),
            transmission=trans,
            reflection=reflect,
            grid=grid,
            wavefunction=psi_all[-1],
        )
        self._results["dynamics_result"] = result
        self._results["dynamics_dim"] = "1d"
        self._results["wavepacket_result_1d"] = (times, psi_all)
        logger.info(f"    1D wavepacket E={E0:.4f} au: "
                    f"P_trans={trans[-1]:.4f}, P_refl={reflect[-1]:.4f}")

    # ------------------------------------------------------------------
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
            f"Method: {self.config.method}",
            f"NN fit: {self.config.use_nn_fit}",
        ]
        if "dynamics_result" in self._results:
            r = self._results["dynamics_result"]
            dim = self._results.get("dynamics_dim", "1d")
            lines.append(f"Dynamics: {dim.upper()}")
            lines.append(f"Energy range: {r.energy[0]:.4f} - {r.energy[-1]:.4f} au")
            lines.append(f"Max reaction probability: {r.transmission.max():.4f}")
        if "nn_fit_rmse" in self._results:
            lines.append(f"NN PES fit RMSE: {self._results['nn_fit_rmse']:.3e} au")
        if "nn_fit_grad_rmse" in self._results:
            lines.append(f"NN PES grad RMSE: {self._results['nn_fit_grad_rmse']:.3e} au/Bohr")
        if "wavepacket_result" in self._results:
            lines.append(f"Wavepacket final P_react: "
                         f"{self._results['wp_final_reaction']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
