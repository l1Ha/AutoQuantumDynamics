#!/usr/bin/env python3
"""端到端集群生产管线 — 一条命令完成 配置→提交→监控→取回→合并→出图。

    python scripts/production.py --system H3_2D --pes leps \
        --e-min 0.10 --e-max 0.30 --n-points 12 --chunks 4 \
        --grid 192 144 --steps 2500 --partition liquid_high

流程:
    1. 生成 Slurm array 作业脚本 (每分片一个能量窗口)
    2. 同步代码到集群 + sbatch 提交
    3. 轮询 squeue 直到完成 (或 --max-wait 超时)
    4. 取回分片 npz → 本地合并 → 出图
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HOST = os.environ.get("AQD_HOST", "target-server")
REMOTE_DIR = "$HOME/AutoQuantum"
RESULTS_REMOTE = "$HOME/aqd_results"
RESULTS_LOCAL = os.path.join(ROOT, "results")


def _ssh(cmd: str, timeout: int = 120) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"  [warn] ssh 返回 {r.returncode}: {r.stderr[:200]}")
    return r.stdout


def _scp_remote(remote_path: str, local_path: str) -> None:
    subprocess.run(["scp", "-q", "-r", f"{HOST}:{remote_path}", local_path],
                   check=True, timeout=120)


def generate_sbatch(partition: str, args) -> str:
    chunks = args.chunks
    pts_per = max(1, args.n_points // chunks)
    return f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH --array=0-{chunks - 1}
#SBATCH -N 1
#SBATCH --time={args.time_limit}
#SBATCH -o aqd_scan_%a.out
#SBATCH -e aqd_scan_%a.err
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum

SPAN=$(python -c "print(({args.e_max} - {args.e_min}) / {chunks})")
EMIN=$(python -c "print({args.e_min} + $SPAN * $SLURM_ARRAY_TASK_ID)")
EMAX=$(python -c "print({args.e_min} + $SPAN * ($SLURM_ARRAY_TASK_ID + 1))")

FLAGS=""
if [ "{args.torch}" = "1" ]; then FLAGS="--torch --dtype {args.dtype}"; fi

python scripts/cluster_scan.py \\
    --e-min $EMIN --e-max $EMAX --points {pts_per} \\
    --grid-r {args.grid_r} --grid-rv {args.grid_rv} \\
    --steps {args.steps} --part $SLURM_ARRAY_TASK_ID \\
    $FLAGS \\
    --out $HOME/aqd_results/scan_part$SLURM_ARRAY_TASK_ID.npz
"""


def main():
    parser = argparse.ArgumentParser(
        description="端到端集群能量扫描管线")
    parser.add_argument("--system", default="H3_2D",
                        choices=["H3_2D"], help="体系")
    parser.add_argument("--pes", default="leps",
                        choices=["leps", "eckart"], help="势能面")
    parser.add_argument("--e-min", type=float, default=0.10)
    parser.add_argument("--e-max", type=float, default=0.30)
    parser.add_argument("--n-points", type=int, default=12)
    parser.add_argument("--chunks", type=int, default=4,
                        help="Slurm array 分片数 (并行度)")
    parser.add_argument("--grid", type=int, nargs=2, default=[192, 144],
                        metavar=("NR", "NRV"), help="R/r 网格点数")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--partition", default="liquid_high",
                        choices=["liquid_high", "air", "all"],
                        help="Slurm 分区")
    parser.add_argument("--time-limit", default="01:00:00")
    parser.add_argument("--torch", action="store_true",
                        help="使用 GPU (torch) 后端")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"])
    parser.add_argument("--max-wait", type=int, default=3600,
                        help="最大等待秒数 (0=只提交不等待)")
    parser.add_argument("--output-prefix", default="scan")
    args = parser.parse_args()

    print("=" * 60)
    print(f"AutoQuantum 集群生产管线")
    print(f"  体系: {args.system} / {args.pes}")
    print(f"  能量: {args.e_min}–{args.e_max} au ({args.n_points} 点, "
          f"{args.chunks} 分片)")
    print(f"  网格: {args.grid[0]}×{args.grid[1]}, {args.steps} 步")
    print(f"  分区: {args.partition}" +
          (" (GPU)" if args.torch else " (CPU)"))
    print("=" * 60)

    # --- Step 1: 同步代码 ---
    print("\n[1/5] 同步代码 ...")
    subprocess.run(["bash", os.path.join(ROOT, "scripts", "remote.sh"), "sync"],
                   check=True, cwd=ROOT, timeout=300)

    # --- Step 2: 清理远程旧结果并提交 ---
    print("\n[2/5] 提交 Slurm array 作业 ...")
    sbatch_content = generate_sbatch(args.partition, args)
    sbatch_local = os.path.join(ROOT, "results", "aqd_production.sbatch")
    os.makedirs(os.path.dirname(sbatch_local), exist_ok=True)
    with open(sbatch_local, "w") as f:
        f.write(sbatch_content)

    _ssh(f"mkdir -p {RESULTS_REMOTE} && rm -f {RESULTS_REMOTE}/scan_part*.npz")
    subprocess.run(["scp", "-q", sbatch_local, f"{HOST}:~/aqd_production.sbatch"],
                   check=True, timeout=60)
    out = _ssh("sbatch ~/aqd_production.sbatch")
    job_id = out.strip().split()[-1] if out.strip() else ""
    print(f"  已提交 job {job_id}")

    # --- Step 3: 监控 ---
    if args.max_wait > 0:
        print(f"\n[3/5] 等待作业完成 (最长 {args.max_wait}s) ...")
        t0 = time.time()
        while time.time() - t0 < args.max_wait:
            out = _ssh(f"squeue -j {job_id} --states=ALL -h -o %T 2>/dev/null")
            states = [s.strip() for s in out.strip().split("\n") if s.strip()]
            if not states or all(s in ("COMPLETED", "FAILED", "CANCELLED")
                                 for s in states):
                break
            print(f"  ... {len(states)} 个任务运行中 "
                  f"({int(time.time() - t0)}s)", flush=True)
            time.sleep(15)

        final = _ssh(f"sacct -j {job_id} --format=State -n | sort -u | head -3")
        print(f"  最终状态: {final.strip()}")
        if "FAILED" in final:
            print("  [error] 有任务失败, 请检查 ~/aqd_scan_*.err")
            sys.exit(1)

    # --- Step 4: 取回结果 ---
    print("\n[4/5] 取回结果 ...")
    os.makedirs(RESULTS_LOCAL, exist_ok=True)
    remote_dir = RESULTS_REMOTE.replace("$HOME", "/storage/home/lih")
    try:
        _scp_remote(f"{remote_dir}/scan_part*.npz", RESULTS_LOCAL + "/")
    except subprocess.CalledProcessError:
        print("  [warn] 部分分片缺失; 取回已有文件")
    parts = sorted(f for f in os.listdir(RESULTS_LOCAL)
                   if f.startswith("scan_part") and f.endswith(".npz"))
    print(f"  取回 {len(parts)} 个分片")

    # --- Step 5: 合并出图 ---
    print(f"\n[5/5] 合并出图 ...")
    prefix = os.path.join(RESULTS_LOCAL, f"{args.output_prefix}_merged")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "merge_scan.py"),
                    "--parts", os.path.join(RESULTS_LOCAL, "scan_part*.npz"),
                    "--out", prefix, "--plot"], check=True, timeout=300)

    print(f"\n{'=' * 60}")
    print(f"完成! 结果: {prefix}.npz / {prefix}.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
