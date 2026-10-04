#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_diagts
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=01:00:00
#SBATCH -o aqd_diagts_%j.out
#SBATCH -e aqd_diagts_%j.err
export OMP_NUM_THREADS=8
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum
python -u scripts/diag_ts_curvature.py
