import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .utils import prepare_batch

class BasicTrainer():
    def __init__(self, model, optimizer, evaluation=None, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.evaluation = evaluation
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0

    def train_epoch(self, loader, epoch):
        self.current_epoch = epoch
        self.model = self.model.train()
        for batch in tqdm(loader, desc=f'Epoch {epoch}: ', disable=False):
            x, y = prepare_batch(batch, device=self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

    def evaluate(self, datasets):
        dataset_train = datasets['train']
        dataset_valid = datasets['valid']
        return self.evaluation(self.model, dataset_train, dataset_valid)

    def save_checkpoint(self, path):
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model': self.model.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])


class EmbeddingTrainer():
    def __init__(self, embedder, loss_func, optimizers, mining_func=None, evaluation=None, device='cuda'):
        self.embedder = embedder.to(device)
        self.loss_func = loss_func
        self.mining_func = mining_func
        self.optimizers = optimizers
        self.evaluation = evaluation
        self.device = device
        self.current_epoch = 0


    def train_epoch(self, loader, epoch):
        self.current_epoch = epoch
        self.embedder = self.embedder.train()
        for batch in tqdm(loader, desc=f'Epoch {epoch}: ', disable=False):
            x, y = prepare_batch(batch, device=self.device)

            # Optimizers reset 
            for _, optimizer in self.optimizers.items():
                optimizer.zero_grad()

            # Calculate loss
            embeddings = self.embedder(x)
            if self.mining_func is not None:
                indices_tuple = self.mining_func(embeddings, y)
                loss = self.loss_func(embeddings, y, indices_tuple)
            else:
                loss = self.loss_func(embeddings, y)
            loss.backward()

            # Optimizers step
            for _, optimizer in self.optimizers.items():
                optimizer.step()

    def evaluate(self, datasets):
        dataset_train = datasets['reference']
        dataset_valid = datasets['valid']
        return self.evaluation(self.embedder, dataset_train, dataset_valid) 

    def save_checkpoint(self, path):
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'embedder': self.embedder.state_dict(),
            'loss_func': self.loss_func.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.current_epoch = checkpoint['epoch']
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.loss_func.load_state_dict(checkpoint['loss_func'])
