import torch
from timm import create_model
from timm.data.transforms_factory import create_transform
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root)

from torchvision import transforms as T

# Wildlife training
from wildlife_training.models import create_model_with_categories
from wildlife_training.trainers import BasicTrainer
from wildlife_training.data import ImageDataset, CategoryImageDataset, split_standard, split_expanding_years
from wildlife_training.inference import Evaluation
from wildlife_datasets import datasets
from wildlife_datasets.utils.splits import ReplicableRandomSplit


def apply_corrections(df):
    corrections = {
            201: 't025',
            1826: 't322',
            2323: 't063',
            3878: 't221',
            3879: 't221',
            3881: 't221',
            3900: 't221',
            3905: 't221',
            4368: 't063',
            4388: 't172',
            4723: 't090',        
            4729: 't090',
            5125: 't388',
            5231: 't243',
            5690: 't200',
            5770: 't236',
            5883: 't160',
            6450: 't018',
            6534: 't063',
            6607: 't441',
            6609: 't441',
            6613: 't441',
            6616: 't441',
            7410: 't063',
        }
    for key in corrections.keys():
        df.loc[df['id'] == key, 'identity'] = corrections[key]
    return df


config = {
    'batch_size': 128,
    'device': 'cuda',
    'epochs': 100,
    'workers': 6,
    'folder': 'runs/',
    'name': os.path.splitext(os.path.basename(__file__))[0],
    }

# Prepare datasets
dataset_factory = datasets.SeaTurtleID('/mnt/data/turtles/datasets/datasets/SeaTurtleID')
dataset_factory.df = dataset_factory.df[~dataset_factory.df['bbox'].isnull()]
dataset_factory.df = dataset_factory.df[dataset_factory.df.groupby('identity')['id'].transform('count') > 1]
dataset_factory.df = dataset_factory.df.reset_index()
dataset_factory.df = apply_corrections(dataset_factory.df)

dataset_factory.root = '/home/cermavo3/projects/datasets/data/256x256_bbox' # cropped to 256x256 with bbox


transform_train = create_transform(
    input_size = 224,
    is_training = True,
    auto_augment = 'rand-m10-n2-mstd1',
    )

transform_valid = create_transform(
    input_size = 224,
    is_training = False,
)


splitter = ReplicableRandomSplit(
    n_splits=5,
    random_state=1,
    train_size=0.7
    )

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='full', # Add bbox/segmentation using different data root.
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