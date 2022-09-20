#!/bin/bash
#SBATCH --partition gpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@login.rci.cvut.cz

Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: login.rci.cvut.cz
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
ml PyTorch/1.9.0-fosscuda-2020b
ml torchvision/0.10.0-fosscuda-2020b-PyTorch-1.9.0
ml timm/0.4.12-fosscuda-2020b-PyTorch-1.9.0
ml scikit-learn/0.24.1-fosscuda-2020b

ml Anaconda3/2021.05


source venv_gpu/bin/activate
jupyter-notebook --no-browser --port=${port} --ip=${node}
