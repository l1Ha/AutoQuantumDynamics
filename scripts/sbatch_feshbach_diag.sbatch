#!/bin/bash
#SBATCH -p liquid_high
#SBATCH -J aqd_fdiag
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH -o aqd_fdiag_%j.out
#SBATCH -e aqd_fdiag_%j.err
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum
echo "=== node: $(hostname) | job: $SLURM_JOB_ID | $(date) ==="
python scripts/feshbach_width.py --R 5.556 --n-virt 22
echo "=== exit code: $? | $(date) ==="
