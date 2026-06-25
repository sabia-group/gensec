#!/bin/bash -l

# Example scheduler template for the clean trained-model search run.
# Edit modules, environment name, resources, and GenSec path for your cluster.

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH -o search.out
#SBATCH -e search.err

module purge
# conda activate your_gensec_environment

# Optional: use a model outside this folder instead of ./latest.model
# export GENSEC_MACE_MODEL=/absolute/path/to/model.model
export GENSEC_MACE_DEVICE=${GENSEC_MACE_DEVICE:-cuda}
export GENSEC_MACE_DTYPE=${GENSEC_MACE_DTYPE:-float64}

python /path/to/gensec.py parameters.json
