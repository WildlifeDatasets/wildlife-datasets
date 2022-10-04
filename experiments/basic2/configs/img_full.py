import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]

dataset_factory.root = '/home/cermavo3/projects/datasets/data/256x256_full' # cropped to 256x256

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='full',
)