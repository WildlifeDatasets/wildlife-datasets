import os
configs = [
    "--config 'configs/baseline.py' ",

    "--config 'configs/img_seg.py' ", 
    "--config 'configs/img_full.py' ",

    "--config 'configs/augment_none.py' ",
    "--config 'configs/augment_randaug.py' ",

    "--config 'configs/size_224.py' ",
    "--config 'configs/size_448.py' ",
    "--config 'configs/size_full.py' ",

    "--config 'configs/features_model_top.py' ",
    "--config 'configs/features_model_left.py' ",
    "--config 'configs/features_model_topleft.py' ",
    "--config 'configs/features_model_right.py' ",
    "--config 'configs/features_model_topright.py' ",
    "--config 'configs/features_onehot.py' ",

    "--config 'configs/train_full_lr03.py' ",
    "--config 'configs/train_full_lr04.py' ",
    "--config 'configs/train_only_last-lr04.py' ",
    "--config 'configs/train_only_last_lr03.py' ", 
    "--config 'configs/train_pretrain_lr03.py' ",
    "--config 'configs/train_pretrain_lr04.py' ",  

   ]

for c in configs:
    os.system(f"sbatch train.sh {c}")
