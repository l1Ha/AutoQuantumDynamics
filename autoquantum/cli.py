import argparse
import sys
import json

from autoquantum.core.engine import AutoPipeline, PipelineConfig
from autoquantum import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="autoquantum",
        description="AutoQuantum: 分子反应动力学全维量子动力学自动计算平台",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"AutoQuantum v{__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    run_parser = sub.add_parser("run", help="运行完整量子动力学流程")
    run_parser.add_argument(
        "-s", "--system", default="H2_1D",
        choices=["H2_1D", "H3_2D"],
        help="体系名称 (默认: H2_1D)",
    )
    run_parser.add_argument(
        "-p", "--pes", default="morse",
        choices=["morse", "harmonic", "lj", "leps", "eckart"],
        help="势能面类型 (默认: morse)",
    )
    run_parser.add_argument(
        "-m", "--method", default="auto",
        choices=["auto", "stationary", "wavepacket"],
        help="动力学方法: auto=按体系默认(2D波包/1D定态), "
             "stationary=时间无关扫描(2D版为实验性), wavepacket=含时波包",
    )
    run_parser.add_argument(
        "-o", "--output", default=None,
        help="输出目录 (默认: output_{system})",
    )
    run_parser.add_argument(
        "--no-nn", action="store_true",
        help="跳过神经网络拟合步骤",
    )
    run_parser.add_argument(
        "--nn-epochs", type=int, default=300,
        help="神经网络训练轮数 (默认: 300)",
    )
    run_parser.add_argument(
        "--nn-layers", type=int, nargs="+", default=[64, 64, 32],
        help="隐藏层结构 (默认: 64 64 32)",
    )
    run_parser.add_argument(
        "--nn-force-weight", type=float, default=None,
        help="力训练权重 (以 ∂V/∂x 为监督目标; 默认: 2D=1.0, 1D=0)",
    )
    run_parser.add_argument(
        "--e-min", type=float, default=0.001,
        help="最小散射能量 (默认: 0.001)",
    )
    run_parser.add_argument(
        "--e-max", type=float, default=0.25,
        help="最大散射能量 (默认: 0.25)",
    )
    run_parser.add_argument(
        "--e-points", type=int, default=100,
        help="能量扫描点数 (默认: 100)",
    )
    run_parser.add_argument(
        "--wp-points", type=int, default=6,
        help="波包方法能量扫描点数 (默认: 6)",
    )
    run_parser.add_argument(
        "--wp-steps", type=int, default=3000,
        help="波包传播步数 (默认: 3000)",
    )
    run_parser.add_argument(
        "--no-gif", action="store_true",
        help="波包方法不生成 GIF 动画",
    )
    run_parser.add_argument(
        "--wp-deconvolve", action="store_true",
        help="多宽度能量反卷积 (恢复 T(E); 病态反问题, 运行时报告残差)",
    )
    run_parser.add_argument(
        "--wp-convergence", action="store_true",
        help="传播收敛检查 (基线 vs 加密网格/时间步)",
    )

    info_parser = sub.add_parser("info", help="显示系统信息")
    info_parser.add_argument(
        "system", nargs="?", default="H2_1D",
        help="体系名称",
    )

    sample_parser = sub.add_parser(
        "sample", help="用电子结构后端生成 NN 训练数据 (npz)")
    sample_parser.add_argument(
        "--backend", default="demo",
        choices=["demo", "analytic", "xtb", "pyscf", "ase"],
        help="电子结构后端 (demo=内置 LJ 演示, 非量子化学)",
    )
    sample_parser.add_argument(
        "--input", required=True, help="参考几何 XYZ 文件 (Bohr)")
    sample_parser.add_argument(
        "-o", "--output", default="data.npz", help="输出数据集 (.npz)")
    sample_parser.add_argument(
        "--n-per-dim", type=int, default=5, help="每维采样点数")
    sample_parser.add_argument(
        "--range", type=float, nargs=2, default=[-0.4, 0.4],
        help="笛卡尔位移范围 (Bohr)")
    sample_parser.add_argument(
        "--active-atoms", type=int, nargs="+", default=None,
        help="参与位移的原子序号 (默认全部)")
    sample_parser.add_argument(
        "--axes", type=int, nargs="+", default=[0, 1, 2],
        help="参与位移的坐标轴 (0/1/2)")
    sample_parser.add_argument(
        "--min-distance", type=float, default=1.2,
        help="最小原子间距过滤 (Bohr)")
    sample_parser.add_argument(
        "--max-points", type=int, default=5000, help="采样点上限")
    sample_parser.add_argument("--charge", type=int, default=0,
                               help="总电荷 (xtb/pyscf)")
    sample_parser.add_argument("--uhf", type=int, default=0,
                               help="未成对电子数 (xtb)")

    fit_parser = sub.add_parser(
        "fit", help="在数据集 (npz) 上训练 NN 势能代理面")
    fit_parser.add_argument("--data", required=True, help="数据集 (.npz)")
    fit_parser.add_argument("-o", "--output", default="model.pkl",
                            help="输出模型 (.pkl)")
    fit_parser.add_argument("--epochs", type=int, default=800)
    fit_parser.add_argument("--lr", type=float, default=0.005)
    fit_parser.add_argument("--layers", type=int, nargs="+",
                            default=[64, 64, 32])
    fit_parser.add_argument("--force-weight", type=float, default=1.0,
                            help="力训练权重 (0=纯能量拟合)")
    fit_parser.add_argument("--max-train-points", type=int, default=6000)
    fit_parser.add_argument("--no-gif", action="store_true",
                            help=argparse.SUPPRESS)

    sub.add_parser("backends", help="列出电子结构后端可用性")

    args = parser.parse_args()

    if args.command == "run":
        return _run_pipeline(args)
    elif args.command == "info":
        return _show_info(args)
    elif args.command == "sample":
        return _sample_data(args)
    elif args.command == "fit":
        return _fit_nn(args)
    elif args.command == "backends":
        return _show_backends()
    else:
        parser.print_help()


def _run_pipeline(args):
    output = args.output or f"output_{args.system.lower()}"

    is_2d = args.system == "H3_2D" or args.pes in ("leps", "eckart")

    config = PipelineConfig(
        system_name=args.system,
        pes_type=args.pes,
        method=args.method,
        use_nn_fit=not args.no_nn,
        nn_hidden_layers=args.nn_layers,
        nn_epochs=args.nn_epochs,
        nn_force_weight=(args.nn_force_weight if args.nn_force_weight is not None
                         else (1.0 if is_2d else 0.0)),
        energy_min=args.e_min,
        energy_max=args.e_max,
        n_energy_points=args.e_points,
        wp_scan_points=args.wp_points,
        wp_n_steps=args.wp_steps,
        wp_save_gif=not args.no_gif,
        wp_deconvolve=args.wp_deconvolve,
        wp_check_convergence=args.wp_convergence,
        output_dir=output,
    )

    if args.pes == "eckart":
        config.pes_params = {
            "V0": 0.015, "beta": 1.5, "k_r": 0.5,
            "r0": 1.401, "coupling": 0.08,
        }
        if args.system == "H3_2D":
            config.energy_max = 0.05
    elif args.pes == "leps":
        config.pes_params = {
            "D": 0.1744, "alpha": 1.028, "r0": 1.401, "sato": 0.05,
        }
    elif args.system == "H3_2D":
        # H3_2D 未指定 PES 时默认 Eckart 垒
        config.pes_type = "eckart"
        config.pes_params = {
            "V0": 0.015, "beta": 1.5, "k_r": 0.5,
            "r0": 1.401, "coupling": 0.08,
        }
        config.energy_max = 0.05

    pipeline = AutoPipeline(config)
    pipeline.run()
    print()
    print(pipeline.summary())
    return pipeline


def _read_xyz(path: str):
    """解析简单 XYZ 文件 (坐标按文件原样视为 Bohr)。"""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0].split()[0])
    symbols, coords = [], []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, coords


def _sample_data(args):
    import numpy as np
    from autoquantum.pes.calculators import make_calculator
    from autoquantum.pes.abinitio import AbInitioData

    symbols, coords = _read_xyz(args.input)
    print(f"参考几何: {len(symbols)} 原子 ({' '.join(symbols)})")

    kwargs = {}
    if args.backend in ("xtb", "pyscf"):
        kwargs = {"charge": args.charge}
        if args.backend == "xtb":
            kwargs["uhf"] = args.uhf
    calc = make_calculator(args.backend, symbols=symbols, **kwargs)
    print(f"后端: {calc.name} {calc.provenance}")

    data = AbInitioData.sample_geometries(
        calc, np.asarray(coords), ranges=(tuple(args.range),),
        n_per_dim=args.n_per_dim, active_atoms=args.active_atoms,
        axes=args.axes, min_distance=args.min_distance,
        max_points=args.max_points, verbose=True)
    data.save_npz(args.output)
    print(f"已保存 {data.n_points} 个构型 → {args.output}")
    print(f"能量范围: [{data.energies.min():.6f}, {data.energies.max():.6f}] Hartree")
    return 0


def _fit_nn(args):
    import numpy as np
    from autoquantum.pes.abinitio import AbInitioData
    from autoquantum.nn import NNTrainer, TrainingConfig
    from autoquantum.nn.model import PESNN

    data = AbInitioData.load_npz(args.data)
    prov = getattr(data, "provenance", None)
    if prov:
        print(f"数据来源: {prov}")
    print(f"训练点: {data.n_points}, 特征维度: {data.points.shape[1]}")

    dY = data.gradients if (args.force_weight > 0
                            and data.gradients is not None) else None
    if dY is not None:
        # 几何梯度 (n, N_atoms, 3) → 与展平笛卡尔特征 (n, 3N) 对齐
        dY = np.asarray(dY).reshape(dY.shape[0], -1)
    if args.force_weight > 0 and dY is None:
        print("  [warn] 数据集无梯度, 退化为纯能量拟合")

    config = TrainingConfig(hidden_layers=args.layers, epochs=args.epochs,
                            lr=args.lr, force_weight=args.force_weight,
                            max_train_points=args.max_train_points)
    model, history = NNTrainer(config).train(data.points, data.energies, dY=dY)

    pred = model.predict(data.points)
    rmse = float(np.sqrt(np.mean((pred - data.energies) ** 2)))
    span = float(data.energies.max() - data.energies.min())
    print(f"能量 RMSE: {rmse:.3e} Hartree ({rmse / span * 100:.2f}% of span)")
    if data.gradients is not None:
        g_ref = np.asarray(data.gradients).reshape(data.gradients.shape[0], -1)
        g_rmse = float(np.sqrt(np.mean(
            (model.gradient(data.points) - g_ref) ** 2)))
        print(f"梯度 RMSE: {g_rmse:.3e} Hartree/Bohr")

    model.save(args.output)
    print(f"模型已保存 → {args.output}")
    print("用法: from autoquantum.nn.model import PESNN; "
          "model = PESNN.load(path); model.predict(points)")
    return 0


def _show_backends():
    import json
    from autoquantum.pes.calculators import available_calculators
    print(json.dumps(available_calculators(), indent=2, ensure_ascii=False))
    return 0


def _show_info(args):
    info = {
        "version": __version__,
        "system": args.system,
        "available_pes": ["morse", "harmonic", "lj", "leps", "eckart"],
        "available_dynamics": ["1d_scattering", "2d_wavepacket",
                               "1d_wavepacket", "2d_scattering_experimental"],
        "description": {
            "H2_1D": "H₂ 分子一维散射 (Morse 势)",
            "H3_2D": "H + H₂ 二维反应散射 (Eckart 垒 / LEPS + 含时波包)",
        },
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
