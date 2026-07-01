#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=p.ada
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=36
#SBATCH --mail-type=none
#SBATCH --time=23:59:00
#SBATCH -o job.out
#SBATCH -e job.err

module purge
conda activate mace_nn
module load cuda/11.6
module load cudnn/8.8.1 gcc/11 impi/2021.3

echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

export PYTHONPATH="/u/lazerpo/conda-envs/mace_nn/lib/python3.9/site-packages:${PYTHONPATH}"

python3 -u /u/lazerpo/gensec/gensec.py $1
