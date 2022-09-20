import torch
from timm import create_model
from timm.data.transforms_factory import create_transform
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from pytorch_metric_learning import losses

import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root)

# Wildlife training
from wildlife_training.trainers import EmbeddingTrainer
from wildlife_training.data import ImageDataset, split_standard, split_expanding_years
from wildlife_training.inference import Evaluation
from wildlife_datasets import datasets

config = {
    'batch_size': 128,
    'device': 'cuda',
    'epochs': 50,
    'workers': 6,
    'folder': 'runs/',
    'embedding_size': 512,
    'name': os.path.splitext(os.path.basename(__file__))[0],
}

# Prepare datasets
dataset_factory = datasets.SeaTurtleID('/mnt/data/turtles/datasets/datasets/SeaTurtleID')
dataset_factory.df = dataset_factory.df[~dataset_factory.df['bbox'].isnull()]
dataset_factory.df = dataset_factory.df[dataset_factory.df.groupby('identity')['id'].transform('count') > 1]
dataset_factory.df = dataset_factory.df.reset_index()
dataset_factory.root = '/home/cermavo3/projects/datasets/data/256x256_bbox' # cropped to 256x256 with bbox

transform_valid = create_transform(
        input_size = 256,
        is_training = False,
    )

transform_train = create_transform(
        input_size = 256,
        is_training = True,
    )

splitter = StratifiedShuffleSplit(
    n_splits=1,
    random_state=1,
    test_size=0.5
    )

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='full', # Already with bbox
    )

evaluation = Evaluation(
    method = 'knn',
    k = 1,
    metrics = {'acc': accuracy_score, 'acc_bal': balanced_accuracy_score},
    device = config['device'],
    batch_size = config['batch_size'],
    num_workers = config['workers'],
    )

# Prepare trainer
def create_trainer(dataset):
    embedder = create_model(
        model_name = 'efficientnet_b0',
        pretrained = True,
        num_classes = config['embedding_size'],
        )

    loss_func = losses.ArcFaceLoss(
        num_classes = dataset.num_classes,
        embedding_size = config['embedding_size'],
        margin = 28.6,
        scale = 64,
        )

    optimizers = {
        'embedder': torch.optim.Adam(
            params=embedder.parameters(),
            lr=1e-3
            ),
        'loss': torch.optim.Adam(
            params=loss_func.parameters(),
            lr=1e-3
            ),
        }

    trainer = EmbeddingTrainer(
        embedder = embedder,
        loss_func = loss_func,
        optimizers = optimizers,
        evaluation = evaluation,
        device = config['device'],
    )
    return trainer