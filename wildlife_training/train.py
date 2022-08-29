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
    parser.add_argument("--repeat", type=int, default=1, help="Repeat training")

    args = parser.parse_args()
    config_path = args.config
    for _ in range(args.repeat):
        # Load payload
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        payload = config.payload

        # Setup loggin in Tensorboard
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S-%f")[:-2]
        folder = os.path.join(payload['config']['folder'], timestamp + '_' + payload['config']['name'])
        writer = SummaryWriter(log_dir=folder)

        # Create experiment folder 
        if not os.path.exists(os.path.join(folder)):
            os.makedirs(os.path.join(folder))
        shutil.copyfile(config_path, os.path.join(folder, 'config.py'))

        # Training
        epochs = payload['config']['epochs']
        trainer = payload['trainer']
        loader = DataLoader(
            dataset = payload['datasets']['train'],
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
                metrics = trainer.evaluate(payload['datasets'])
                print(metrics)
                for name, value in metrics.items():
                    writer.add_scalar(name, value, e)

            trainer.save_checkpoint(os.path.join(folder, 'model_latest.pth'))
            if e in payload['config'].get('save_at', []):
                trainer.save_checkpoint(os.path.join(folder, f'model_epoch_{e}.pth'))
        
