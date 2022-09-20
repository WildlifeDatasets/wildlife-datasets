import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline_arcface import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]

dataset_factory.df['year'] = dataset_factory.df.date.str[:4]

splits = split_expanding_years(
    dataset_factory.df,
    dataset_factory.root,
    col_year = 'year',
    create_dataset = ImageDataset,
    transform_train = transform_train,
    transform_valid = transform_valid,
    img_load='full',
)