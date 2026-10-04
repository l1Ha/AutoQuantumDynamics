#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_corr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH -o aqd_corr_%j.out
#SBATCH -e aqd_corr_%j.err

export OMP_NUM_THREADS=8
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum

echo "=== node: $(hostname) | job: $SLURM_JOB_ID | $(date) ==="
python -c "import pyscf; print('pyscf', pyscf.__version__)"

python scripts/validate_correlated_methods.py
echo "=== exit code: $? | $(date) ==="
