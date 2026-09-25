#!/usr/bin/env python3
"""合并集群分片扫描结果 → 单一 npz + 对比图。

    python scripts/merge_scan.py --parts "results/scan_part*.npz" \
        --out results/scan_merged --plot
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True, help="分片 npz 的 glob")
    parser.add_argument("--out", required=True, help="合并输出前缀")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(args.parts))
    if not files:
        sys.exit(f"未找到分片: {args.parts}")
    parts = [np.load(f) for f in files]
    energy = np.concatenate([p["energy"] for p in parts])
    reaction = np.concatenate([p["reaction"] for p in parts])
    order = np.argsort(energy)
    energy, reaction = energy[order], reaction[order]
    # 分片闭区间端点重叠: 去重 (保留首个)
    _, keep = np.unique(energy, return_index=True)
    keep = np.sort(keep)
    energy, reaction = energy[keep], reaction[keep]
    r_grid = parts[0]["R_grid"]
    r_grid2 = parts[0]["r_grid"]

    np.savez(args.out + ".npz", energy=energy, reaction=reaction,
             R_grid=r_grid, r_grid2=r_grid2)
    print(f"合并 {len(files)} 分片 → {args.out}.npz ({energy.size} 点)")
    for e, p in zip(energy, reaction):
        print(f"  E={e:.4f}: P_react={p:.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(energy, reaction, "o-", color="#2c3e50", lw=1.8, ms=5)
        ax.set_xlabel("collision energy (au)")
        ax.set_ylabel("P_react")
        ax.set_title("LEPS H+H$_2$: cluster batch scan "
                     f"({energy.size} points)")
        ax.grid(alpha=0.3)
        fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图 → {args.out}.png")


if __name__ == "__main__":
    main()
