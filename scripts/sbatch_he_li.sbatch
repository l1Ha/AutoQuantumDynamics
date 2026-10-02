#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_heli
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=01:00:00
#SBATCH -o aqd_heli_%j.out
#SBATCH -e aqd_heli_%j.err

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum

echo "=== node: $(hostname) | job: $SLURM_JOB_ID | $(date) ==="
echo "pyscf: $(python -c 'import pyscf; print(pyscf.__version__)')"

# He*(2³S) + Li 亚稳态势能面: 四重态 ROHF+spin_lock+MOM, 双重态基态, 原子 FCI 校验
python scripts/calc_metastable_heli.py \
    --basis aug-cc-pVTZ \
    --n-points 30 \
    --mom \
    --out-dir $HOME/aqd_results/he_li_metastable

echo "=== exit code: $? | $(date) ==="
ls -la $HOME/aqd_results/he_li_metastable/ 2>/dev/null || echo "无输出目录"
