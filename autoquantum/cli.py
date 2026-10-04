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
    sample_parser.add_argument("--spin", type=int, default=None,
                               help="自旋参数 2S = Na - Nb (pyscf, 缺省与 --uhf 保持一致)")
    sample_parser.add_argument("--method", default="rhf",
                               help="电子结构方法: rhf, uhf, rohf, dft, rks, roks, uks (pyscf)")
    sample_parser.add_argument("--basis", default="sto-3g",
                               help="基组 (pyscf, 默认 sto-3g)")
    sample_parser.add_argument("--xc", default=None,
                               help="DFT 泛函名称 (如 b3lyp, pbe; pyscf)")
    sample_parser.add_argument("--spin-lock", action="store_true",
                               help="启用严格自旋态检查与自旋污染截断 (pyscf)")
    sample_parser.add_argument("--spin-tol", type=float, default=0.1,
                               help="自旋锁定允许的最大偏差 |<S^2> - S(S+1)| (默认 0.1)")
    sample_parser.add_argument("--mom", action="store_true",
                               help="启用最大重叠法 (MOM) 沿采样序列跟踪特定激发/占据态 (pyscf)")
    _add_solvent_args(sample_parser)

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
    fit_parser.add_argument(
        "--symmetry", action="store_true",
        help="对称函数+共享原子能量模式: 数据集需含 (n,3N) points 与 symbols, "
             "输出置换严格不变的委员会势能面")
    fit_parser.add_argument("--committee", type=int, default=4,
                            help="委员会成员数 (--symmetry, 默认 4)")
    fit_parser.add_argument("--no-gif", action="store_true",
                            help=argparse.SUPPRESS)

    sub.add_parser("backends", help="列出电子结构后端可用性")

    # ---- opt: 几何优化 (极小点) ----
    opt_parser = sub.add_parser("opt", help="几何优化 (BFGS, 解析梯度)")
    _add_calc_args(opt_parser)
    opt_parser.add_argument("-o", "--output", default="optimized.npz",
                            help="输出 npz (含优化几何/能量/梯度)")
    opt_parser.add_argument("--gtol", type=float, default=1e-5,
                            help="梯度收敛阈值 (Hartree/Bohr)")
    opt_parser.add_argument("--max-iter", type=int, default=200)

    # ---- scan: 内坐标 PES 扫描 ----
    sc_parser = sub.add_parser("scan", help="内坐标扫描 -> PES 训练集 (npz)")
    _add_calc_args(sc_parser)
    sc_parser.add_argument("--mode", default="bond",
                           choices=["bond", "angle", "path", "relax-bond"],
                           help="扫描类型")
    sc_parser.add_argument("--atoms", type=int, nargs="+", default=None,
                           help="原子索引: bond i j | angle i j k (顶点 j)")
    sc_parser.add_argument("--range", type=float, nargs=2, default=None,
                           help="扫描范围: 键长 (Bohr) 或键角 (度)")
    sc_parser.add_argument("--n", type=int, default=11, help="扫描点数")
    sc_parser.add_argument("--second", default=None,
                           help="path 模式的第二几何 (XYZ, Bohr)")
    sc_parser.add_argument("-o", "--output", default="scan.npz")

    # ---- freq: 谐振频率 ----
    fr_parser = sub.add_parser("freq", help="谐振频率 (数值 Hessian)")
    _add_calc_args(fr_parser)
    fr_parser.add_argument("--opt-first", action="store_true",
                           help="先做几何优化再算频率")

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
    elif args.command == "opt":
        return _cmd_opt(args)
    elif args.command == "scan":
        return _cmd_scan(args)
    elif args.command == "freq":
        return _cmd_freq(args)
    else:
        parser.print_help()


def _add_solvent_args(p):
    """隐式溶剂参数 (sample/opt/scan/freq 共用; pyscf 后端)。"""
    p.add_argument("--solvent", default=None,
                   help="隐式溶剂名 (如 water/methanol; smd 必须用溶剂名)")
    p.add_argument("--solvent-eps", type=float, default=None,
                   help="溶剂介电常数 (直接指定; smd 不支持)")
    p.add_argument("--solvent-model", default="ddcosmo",
                   choices=["ddcosmo", "pcm", "ddpcm", "smd"],
                   help="隐式溶剂模型 (默认 ddcosmo; smd 仅 SCF 层)")
    p.add_argument("--pcm-variant", default="IEF-PCM",
                   choices=["C-PCM", "IEF-PCM", "COSMO", "SS(V)PE"],
                   help="PCM 变体 (仅 --solvent-model pcm)")


def _solvent_kwargs(args):
    """从 CLI 参数提取溶剂 kwargs (未指定溶剂时不传, 保持默认行为)。"""
    if getattr(args, "solvent", None) is None \
            and getattr(args, "solvent_eps", None) is None:
        return {}
    out = {"solvent_model": args.solvent_model,
           "pcm_variant": args.pcm_variant}
    if args.solvent is not None:
        out["solvent"] = args.solvent
    if args.solvent_eps is not None:
        out["solvent_eps"] = args.solvent_eps
    return out


def _add_calc_args(p):
    """opt/scan/freq 共用的后端参数。"""
    p.add_argument("--backend", default="pyscf", choices=["pyscf", "xtb", "demo"])
    p.add_argument("--input", required=True, help="几何 XYZ 文件 (Bohr)")
    p.add_argument("--method", default="rhf",
                   help="rhf/rohf/uhf/dft/rks/roks/uks/mp2/ccsd/ccsd(t)")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0, help="2S = Na - Nb")
    p.add_argument("--xc", default=None, help="DFT 泛函")
    p.add_argument("--frozen-core", action="store_true", help="冻结核 (MP2/CCSD)")
    _add_solvent_args(p)


def _make_calc(args):
    from autoquantum.pes.calculators import make_calculator
    symbols, coords = _read_xyz(args.input)
    if args.backend == "pyscf":
        return symbols, coords, make_calculator(
            "pyscf", symbols=symbols, basis=args.basis, charge=args.charge,
            spin=args.spin, method=args.method, xc=args.xc,
            frozen_core=args.frozen_core, **_solvent_kwargs(args))
    return symbols, coords, make_calculator(args.backend, symbols=symbols)


def _cmd_opt(args):
    import numpy as np
    from autoquantum.pes.optimize import optimize_geometry
    symbols, coords, calc = _make_calc(args)
    print(f"体系: {len(symbols)} 原子 ({' '.join(symbols)}); "
          f"{args.method}/{args.basis if args.backend == 'pyscf' else args.backend}")
    opt, info = optimize_geometry(calc, np.asarray(coords), gtol=args.gtol,
                                  max_iter=args.max_iter, verbose=True)
    e, g = calc.energy_and_gradient(opt)
    print(f"收敛: {info['converged']} | 迭代 {info['n_iter']} | "
          f"E = {e:.8f} Ha | |g|max = {np.abs(g).max():.2e}")
    print(f"优化几何 (Bohr):\n{np.array2string(opt, precision=6)}")
    np.savez(args.output, coords=opt, energy=e, gradient=g, symbols=symbols,
             converged=info["converged"], method=args.method, basis=args.basis)
    print(f"已保存 → {args.output}")
    return 0


def _cmd_scan(args):
    import numpy as np
    from autoquantum.pes import scan as Scan
    symbols, coords, calc = _make_calc(args)
    c = np.asarray(coords)
    if args.mode in ("bond", "relax-bond"):
        i, j = (args.atoms or [0, 1])[:2]
        lo, hi = args.range or (0.8, 2.5)
        grid = np.linspace(lo, hi, args.n)
        fn = (Scan.relaxed_scan_bond if args.mode == "relax-bond"
              else Scan.scan_bond)
        data = fn(calc, symbols, c, i, j, grid)
    elif args.mode == "angle":
        i, j, k = (args.atoms or [0, 1, 2])[:3]
        lo, hi = args.range or (60.0, 180.0)
        data = Scan.scan_angle(calc, symbols, c, i, j, k,
                               np.linspace(lo, hi, args.n))
    else:                                    # path
        if not args.second:
            raise SystemExit("path 模式需要 --second <XYZ>")
        _, c2 = _read_xyz(args.second)
        data = Scan.scan_path(calc, symbols, c, np.asarray(c2), args.n)
    data.save_npz(args.output, provenance=calc.provenance)
    print(f"扫描 {data.points.shape[0]} 点 → {args.output}; "
          f"能量范围 [{data.energies.min():.6f}, {data.energies.max():.6f}] Ha")
    return 0


def _cmd_freq(args):
    import numpy as np
    from autoquantum.pes.optimize import (optimize_geometry,
                                          harmonic_frequencies)
    symbols, coords, calc = _make_calc(args)
    c = np.asarray(coords)
    if args.opt_first:
        c, info = optimize_geometry(calc, c)
        print(f"先优化: E = {info['energy']:.8f} Ha, |g|max = {info['grad_max']:.2e}")
    freqs, finf = harmonic_frequencies(calc, symbols, c)
    print(f"谐振频率 (cm^-1, {len(freqs)} 个模式):")
    for k, f in enumerate(freqs, 1):
        print(f"  mode {k:2d}: {f:10.2f}{'   (虚频)' if f < 0 else ''}")
    print(f"虚频数 = {finf['n_imag']}")
    return 0


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
        elif args.backend == "pyscf":
            spin_val = args.spin if args.spin is not None else args.uhf
            kwargs.update({
                "spin": spin_val,
                "method": args.method,
                "basis": args.basis,
                "xc": args.xc,
                "spin_lock": args.spin_lock,
                "spin_tol": args.spin_tol,
                "use_mom": args.mom,
            })
            kwargs.update(_solvent_kwargs(args))
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
    if args.symmetry:
        return _fit_symmetry_committee(args, data)

    prov = getattr(data, "provenance", None)
    if prov:
        print(f"数据来源: {prov}")
    print(f"训练点: {data.n_points}, 特征维度: {data.points.shape[1]}")

    dY = data.gradients if (args.force_weight > 0
                            and data.gradients is not None) else None
    if dY is not None and dY.shape != data.points.shape:
        dY = np.asarray(dY).reshape(dY.shape[0], -1)  # (n,N,3) → (n,3N)
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


def _fit_symmetry_committee(args, data):
    """对称函数 + 共享原子能量委员会 (置换/平移/旋转严格不变)。"""
    import numpy as np
    from autoquantum.nn.ensemble import (train_atomic_committee,
                                         AtomicTrainingConfig)
    from autoquantum.nn.symmetry import SymmetryFunctionParams

    if data.points.ndim != 2 or data.points.shape[1] % 3 != 0:
        raise SystemExit("--symmetry 需要 (n, 3N) 笛卡尔坐标数据集")
    symbols = list(data.symbols) if getattr(data, "symbols", None) else None
    if not symbols or len(symbols) * 3 != data.points.shape[1]:
        raise SystemExit("--symmetry 需要数据集内含 symbols (元素序列, 长度 N)")
    coords = data.points.reshape(-1, len(symbols), 3)
    print(f"对称函数模式: {len(symbols)} 原子 {symbols}, "
          f"{coords.shape[0]} 构型, 委员会 {args.committee} 成员")
    committee, info = train_atomic_committee(
        symbols, coords, data.energies, n_models=args.committee,
        config=AtomicTrainingConfig(epochs=args.epochs),
        seed=0)
    print(f"总能量 RMSE: {info['rmse']:.3e} Hartree "
          f"(特征维度 {info['n_features']}/中心)")
    committee.save(args.output)
    print(f"委员会已保存 → {args.output} (置换不变, OOD 不确定性: committee.std(coords))")
    return 0
    prov = getattr(data, "provenance", None)
    if prov:
        print(f"数据来源: {prov}")
    print(f"训练点: {data.n_points}, 特征维度: {data.points.shape[1]}")

    dY = data.gradients if (args.force_weight > 0
                            and data.gradients is not None) else None
    if dY is not None and dY.shape != data.points.shape:
        dY = np.asarray(dY).reshape(dY.shape[0], -1)  # (n,N,3) → (n,3N)
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
