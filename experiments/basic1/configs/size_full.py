import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]
import torchvision.transforms as T
import torch

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=torch.tensor([0.4850, 0.4560, 0.4060]),
        std=torch.tensor([0.2290, 0.2240, 0.2250])
        ),
])

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform,
    transform_valid = transform,
    img_load='bbox',
)