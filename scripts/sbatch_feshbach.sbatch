#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_feshbach
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH -o aqd_feshbach_%j.out
#SBATCH -e aqd_feshbach_%j.err

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum

echo "=== node: $(hostname) | job: $SLURM_JOB_ID | $(date) ==="
python -c "import pyscf; print('pyscf', pyscf.__version__)"

# He*(2³S)+Li ²Σ⁺ 自电离宽度 Γ(R): Feshbach 投影 + SA-CASSCF + L² 赝态成像
# 方法: Movre-Thiel-Meyer JCP 113, 1484 (2000); Langhoff 1974; Hazi 1978
python scripts/feshbach_width.py --scan --n-virt 40 \
    --out $HOME/aqd_results/feshbach_width.npz

echo "=== exit code: $? | $(date) ==="
ls -la $HOME/aqd_results/feshbach_width.npz 2>/dev/null || echo "无输出"
