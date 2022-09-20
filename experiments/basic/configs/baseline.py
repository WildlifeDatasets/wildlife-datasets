import torch
from timm import create_model
from timm.data.transforms_factory import create_transform
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import os
import sys
root = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.append(root)

# Wildlife training
from wildlife_training.models import create_model_with_categories
from wildlife_training.trainers import BasicTrainer
from wildlife_training.data import ImageDataset, CategoryImageDataset, split_standard
from wildlife_training.inference import Evaluation
from wildlife_datasets import datasets

config = {
    'batch_size': 128,
    'device': 'cuda',
    'epochs': 50,
    'workers': 12,
    'folder': 'runs/',
    'name': os.path.splitext(os.path.basename(__file__))[0],
    }

# Prepare datasets
dataset_factory = datasets.SeaTurtleID('/mnt/data/turtles/datasets/datasets/SeaTurtleID')
dataset_factory.df = dataset_factory.df[~dataset_factory.df['bbox'].isnull()]
dataset_factory.df = dataset_factory.df[dataset_factory.df.groupby('identity')['id'].transform('count') > 1]
dataset_factory.df = dataset_factory.df.reset_index()

transform_valid = create_transform(
    input_size = 224,
    is_training = False,
    )

transform_train = create_transform(
    input_size = 224,
    is_training = True,
    )

splitter = StratifiedShuffleSplit(
    n_splits=1,
    random_state=1,
    test_size=0.3
    )

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='bbox',
    )

evaluation = Evaluation(
    method = 'classifier',
    metrics = {'acc': accuracy_score, 'acc_bal': balanced_accuracy_score},
    device = config['device'],
    batch_size = config['batch_size'],
    num_workers = config['workers'],
)

# Prepare trainer
def create_trainer(dataset):
    model = create_model(
        model_name = 'efficientnet_b0',
        pretrained = True,
        num_classes = dataset.num_classes,
        )

    optimizer = torch.optim.Adam(
        params = model.parameters(),
        lr = 1e-3
        )

    trainer = BasicTrainer(
        model = model,
        evaluation = evaluation,
        optimizer = optimizer,
        device = config['device'],
    )
    return trainer