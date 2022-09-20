#!/bin/bash
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ../../venv_gpu/bin/activate
ml torchvision/0.10.0-fosscuda-2020b-PyTorch-1.9.0
ml timm/0.4.12-fosscuda-2020b-PyTorch-1.9.0
ml scikit-learn/0.24.1-fosscuda-2020b
ml faiss/1.7.1-fosscuda-2020b

#python -m venv venv
#source venv/bin/activate


python3 ../train.py "$@"