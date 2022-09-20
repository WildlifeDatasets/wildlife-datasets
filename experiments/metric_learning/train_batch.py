import os
configs = [
    #"--config 'configs/time_ce.py' ",
    #"--config 'configs/time_arcface.py' ",

    #"--config 'configs/baseline_arcface_sc.py' ",
    #"--config 'configs/time_arcface_sc.py' ",

   "--config 'configs/baseline_arcface_sc_5.py' ",
    #"--config 'configs/time_arcface_sc_5.py' ",

   ]

for c in configs:
    os.system(f"sbatch train.sh {c}")
