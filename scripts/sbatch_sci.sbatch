#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_sci
#SBATCH --exclude=xc002
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=03:00:00
#SBATCH -o aqd_sci_%j.out
#SBATCH -e aqd_sci_%j.err
export OMP_NUM_THREADS=8
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum
python -u scripts/validate_selected_ci.py
