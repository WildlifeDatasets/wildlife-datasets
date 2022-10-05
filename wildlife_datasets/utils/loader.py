import os
import numpy as np
import pandas as pd

def get_dataset_folder(root_dataset, dataset_class):
    # TODO: hacky solution
    if dataset_class.__name__.endswith('Segmented'):
        return os.path.join(root_dataset, dataset_class.__name__[:-9])
    else:
        return os.path.join(root_dataset, dataset_class.__name__)

def get_dataframe_path(root_dataframe, dataset_class):
    return os.path.join(root_dataframe, dataset_class.__name__ + '.pkl')

def download_datasets(class_datasets, root_dataset, **kwargs):
    for class_dataset in class_datasets:
        download_dataset(class_dataset, root_dataset, **kwargs)

def download_dataset(class_dataset, root_dataset, overwrite=False):    
    root = get_dataset_folder(root_dataset, class_dataset)
    if overwrite or not os.path.exists(root):
        class_dataset.download.get_data(root)
        
def load_datasets(class_datasets, root_dataset, root_dataframe, **kwargs):
    return [load_dataset(class_dataset, root_dataset, root_dataframe, **kwargs) for class_dataset in class_datasets]

def load_dataset(class_dataset, root_dataset, root_dataframe, overwrite=False):
    if not os.path.exists(root_dataset):
        raise(Exception('Data not found. Download them first.'))
    if not os.path.exists(root_dataframe):
        os.makedirs(root_dataframe)

    root = get_dataset_folder(root_dataset, class_dataset)
    df_path = get_dataframe_path(root_dataframe, class_dataset)
    if overwrite or not os.path.exists(df_path):
        dataset = class_dataset(root, None, download=False)
        dataset.df.to_pickle(df_path)
    else:
        df = pd.read_pickle(df_path)
        dataset = class_dataset(root, df, download=False)
    return dataset
