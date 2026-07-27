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

    config = PipelineConfig(
        system_name=args.system,
        pes_type=args.pes,
        use_nn_fit=not args.no_nn,
        nn_hidden_layers=args.nn_layers,
        nn_epochs=args.nn_epochs,
        energy_min=args.e_min,
        energy_max=args.e_max,
        n_energy_points=args.e_points,
        output_dir=output,
    )

    if args.system == "H3_2D" or args.pes == "eckart":
        config.pes_type = "eckart"
        config.pes_params = {
            "V0": 0.015, "beta": 1.5, "k_r": 0.5,
            "r0": 1.401, "coupling": 0.08,
        }
        config.energy_max = 0.05
    elif args.system == "H3_2D" or args.pes == "leps":
        config.pes_type = "leps"
        config.pes_params = {
            "D": 0.1744, "alpha": 1.028, "r0": 1.401, "sato": 0.05,
        }

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
        "available_dynamics": ["1d_scattering", "2d_scattering", "wavepacket"],
        "description": {
            "H2_1D": "H₂ 分子一维散射 (Morse 势)",
            "H3_2D": "H + H₂ 二维反应散射 (Eckart 垒)",
        },
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
