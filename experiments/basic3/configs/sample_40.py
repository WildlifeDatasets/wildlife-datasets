import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]

splits = split_standard(
    df = dataset_factory.df,
    root = dataset_factory.root,
    splitter = splitter,
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='full',
    )

for split in splits:
    for name, dataset in split.items():
        if name in ['train', 'reference']:
            split[name] = ImageDataset(
                df=dataset.df.sample(frac=0.4, replace=True, random_state=1),
                root=dataset.root,
                transform=dataset.transform,
                img_load=dataset.img_load,
            )