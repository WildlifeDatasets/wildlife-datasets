#!/usr/bin/sh
ml PyTorch/1.9.0-fosscuda-2020b
ml torchvision/0.10.0-fosscuda-2020b-PyTorch-1.9.0
ml timm/0.4.12-fosscuda-2020b-PyTorch-1.9.0
ml scikit-learn/0.24.1-fosscuda-2020b

ml Anaconda3/2021.05
jupyter-notebook --no-browser --port 8890

# Interactive GPU
#srun -p amdgpufast --gres=gpu:1 --cpus-per-task=12  --pty bash -i
#srun -p gpufast --gres=gpu:1 --cpus-per-task=12  --pty bash -i

# Vevn
#python3 -m venv venv
#source venv/bin/activate 
#deactivate

# Tensorboard
#ml tensorboard/2.8.0-foss-2021a
#tensorboard --logdir='runs'
