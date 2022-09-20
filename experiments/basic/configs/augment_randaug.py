import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]

transform_train = create_transform(
    input_size = 224,
    is_training = True,
    auto_augment = 'rand-m10-n2-mstd1',
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