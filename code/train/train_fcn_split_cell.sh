#!/bin/bash
#SBATCH --job-name=train_fcn_split_cell.sh
#SBATCH --partition=super
#SBATCH --nodes=1
#SBATCH --time=30-00:00:00
#SBATCH --output=./train_fcn_split_cell.sh.log
#SBATCH --error=./train_fcn_split_cell.sh.error

module add python/3.7.x-anaconda
source activate s418336
python train_fcn.py cell