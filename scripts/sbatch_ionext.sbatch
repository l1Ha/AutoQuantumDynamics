#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_ionext
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH -o aqd_ionext_%j.out
#SBATCH -e aqd_ionext_%j.err

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum

echo "=== node: $(hostname) | job: $SLURM_JOB_ID | $(date) ==="
python -c "import pyscf; print('pyscf', pyscf.__version__)"

# He+Li⁺ 离子曲线扩展范围 CCSD(T)+CP: R ∈ [2.3, 20] bohr, 短程加密
# (0.20.2 审计网格 [3.2,12] 缺短程排斥壁 — 参考数据自 2.55 bohr 起)
python scripts/audit_ion_extended.py --basis aug-cc-pVQZ \
    --out $HOME/aqd_results/audit_ion_extended.npz

echo "=== exit code: $? | $(date) ==="
ls -la $HOME/aqd_results/audit_ion_extended.npz 2>/dev/null || echo "无输出"
