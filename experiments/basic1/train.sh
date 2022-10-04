#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

source ../../venv_gpu/bin/activate
ml PyTorch/1.9.0-fosscuda-2020b
ml torchvision/0.10.0-fosscuda-2020b-PyTorch-1.9.0
ml timm/0.4.12-fosscuda-2020b-PyTorch-1.9.0
ml scikit-learn/0.24.1-fosscuda-2020b
ml faiss/1.7.1-fosscuda-2020b

python3 ../train.py "$@"