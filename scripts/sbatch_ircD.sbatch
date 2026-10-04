#!/bin/bash
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH -J aqd_ircD
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=00:30:00
#SBATCH -o aqd_ircD_%j.out
#SBATCH -e aqd_ircD_%j.err
export OMP_NUM_THREADS=8
export PYSCF_TMPDIR=$HOME/pyscf_tmp
export PATH=$HOME/aqd-env/bin:$PATH
cd $HOME/AutoQuantum
python -u scripts/validate_irc_x2c.py A C D
