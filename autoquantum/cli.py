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

    info_parser = sub.add_parser("info", help="显示系统信息")
    info_parser.add_argument(
        "system", nargs="?", default="H2_1D",
        help="体系名称",
    )

    args = parser.parse_args()

    if args.command == "run":
        return _run_pipeline(args)
    elif args.command == "info":
        return _show_info(args)
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
