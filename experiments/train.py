import os
import shutil
from runpy import run_path
from datetime import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .py file")
    parser.add_argument("--epoch_eval", type=int, default=1, help="Evaluation frequency")
    args = parser.parse_args()


    # Load payload
    print(args.config)
    payload = run_path(args.config)
    config = payload['config']
    splits = payload['splits']
    create_trainer = payload['create_trainer']

    # Setup experiment folder
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S-%f")[:-2]
    folder = os.path.join(config['folder'], timestamp + '_' + config['name'])
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copyfile(args.config, os.path.join(folder, 'config.py'))


    for split in splits:
        folder_log = os.path.join(folder, split.get('name', ''))
        writer = SummaryWriter(log_dir=folder_log)

        # Training
        trainer = create_trainer(split['train'] )
        loader = DataLoader(
            dataset = split['train'],
            batch_size = config['batch_size'],
            num_workers = config['workers'],
            shuffle=True,
            )

        for e in range(trainer.current_epoch+1, config['epochs']):
            print(f'======= Train Epoch {e} =======')
            trainer.train_epoch(loader, e)

            # Evaluate metrics
            if e % args.epoch_eval == 0:
                print(f'======= Eval Epoch {e} =======')
                metrics = trainer.evaluate(split)
                print(metrics)
                for name, value in metrics.items():
                    writer.add_scalar(name, value, e)

            trainer.save_checkpoint(os.path.join(folder_log, "model_latest.pth"))        
