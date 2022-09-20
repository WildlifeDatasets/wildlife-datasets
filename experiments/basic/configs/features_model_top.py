import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]

dataset_factory = datasets.SeaTurtleID('/mnt/data/turtles/datasets/datasets/SeaTurtleID')
dataset_factory.df = dataset_factory.df.query("position == 'top'")
dataset_factory.df = dataset_factory.df[~dataset_factory.df['bbox'].isnull()]
dataset_factory.df = dataset_factory.df[dataset_factory.df.groupby('identity')['id'].transform('count') > 1]
dataset_factory.df = dataset_factory.df.reset_index()

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='bbox',
    )