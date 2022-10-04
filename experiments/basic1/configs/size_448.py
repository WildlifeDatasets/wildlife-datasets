import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]
config['batch_size'] = 64

transform = create_transform(
    input_size = 448,
    is_training = False,
    )

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform,
    transform_valid = transform,
    img_load='bbox',
)