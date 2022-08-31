import os
import shutil
import importlib.util
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
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    payload = config.payload

    # Setup loggin in Tensorboard
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S-%f")[:-2]
    folder = os.path.join(payload['config']['folder'], timestamp + '_' + payload['config']['name'])
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create experiment folder 
    if not os.path.exists(os.path.join(folder)):
        os.makedirs(os.path.join(folder))
    shutil.copyfile(args.config, os.path.join(folder, 'config.py'))


    print('Training')
    for dataset_split in payload['datasets']:
        folder_log = os.path.join(folder, dataset_split.get('name', ''))
        writer = SummaryWriter(log_dir=folder_log)

        # Training
        epochs = payload['config']['epochs']
        trainer = payload['create_trainer'](dataset_split['train'])
        loader = DataLoader(
            dataset = dataset_split['train'],
            batch_size = payload['config']['batch_size'],
            num_workers = payload['config']['workers'],
            shuffle=True,
            )

        if "load_from" in payload['config']:
            trainer.load_checkpoint(payload['config']['load_from'])

        for e in range(trainer.current_epoch + 1, epochs):
            print(f'======= Epoch {e} =======')
            trainer.train_epoch(loader, e)

            # Evaluate metrics
            if e % args.epoch_eval == 0:
                metrics = trainer.evaluate(dataset_split)
                print(metrics)
                for name, value in metrics.items():
                    writer.add_scalar(name, value, e)

            trainer.save_checkpoint(os.path.join(folder_log, "model_latest.pth"))        
