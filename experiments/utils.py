import os
import sys
root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root)

import torchvision.transforms as T
from tqdm import tqdm

# Wildlife training
from wildlife_training.data import ImageDataset
from wildlife_datasets import datasets


def resize_turtles(new_root, size=256, img_load='bbox'):
    '''
    Example usage:

    from utils import resize_turtles
    resize_turtles('../../data/256x256_bbox', size=256, img_load='bbox')
    '''

    dataset_factory = datasets.SeaTurtleID('/mnt/data/turtles/datasets/datasets/SeaTurtleID')
    dataset_factory.df = dataset_factory.df[~dataset_factory.df['bbox'].isnull()]
    dataset_factory.df = dataset_factory.df[dataset_factory.df.groupby('identity')['id'].transform('count') > 1]
    dataset_factory.df = dataset_factory.df.reset_index()

    dataset = ImageDataset(
        dataset_factory.df,
        dataset_factory.root,
        transform=T.Resize(size=size),
        img_load=img_load)

    for i in tqdm(range(len(dataset))):
        path = os.path.join(new_root, dataset.df.iloc[i]['path'])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        image, _ = dataset[i]
        image.save(path)