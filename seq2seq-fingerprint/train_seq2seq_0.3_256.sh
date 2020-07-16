#!/bin/bash
#SBATCH --job-name=train_seq2seq_0.3_256.sh
#SBATCH --partition=GPUv100s
#SBATCH --nodes=1
#SBATCH --time=30-00:00:00
#SBATCH --output=./train_seq2seq_0.3_256.sh.log
#SBATCH --error=./train_seq2seq_0.3_256.sh.error

module add python/2.7.x-anaconda
source activate py2
python train.py train zinc_gru_0.3_256/ ../data/zinc_train.tokens ../data/zinc_val.tokens --batch_size 64