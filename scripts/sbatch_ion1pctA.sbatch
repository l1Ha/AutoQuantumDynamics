#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_ion1pct
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=04:00:00
#SBATCH -o aqd_ion1pct_%j.out
#SBATCH -e aqd_ion1pct_%j.err
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum
echo "=== node: $(hostname) | job: $SLURM_JOB_ID | $(date) ==="
python scripts/iterate_ion_1pct.py --configs ccsdt_qz,ccsd_qz --stride 1
echo "=== exit code: $? | $(date) ==="
