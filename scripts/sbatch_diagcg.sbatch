#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_diagcg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=00:40:00
#SBATCH -o aqd_diagcg_%j.out
#SBATCH -e aqd_diagcg_%j.err
export OMP_NUM_THREADS=8
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum
python -u scripts/diag_casscf_grad.py
