import os
import numpy as np
import pandas as pd

def get_dataset_folder(root_dataset, dataset_class):
    return os.path.join(root_dataset, dataset_class.__name__)

def get_dataframe_path(root_dataframe, dataset_class):
    return os.path.join(root_dataframe, dataset_class.__name__ + '.pkl')

def download_datasets(info_datasets, root_dataset, **kwargs):
    for info_dataset in info_datasets:
        download_dataset(info_dataset, root_dataset, **kwargs)

def download_dataset(info_dataset, root_dataset, overwrite=False):    
    dataset_class = info_dataset[0]
    root = get_dataset_folder(root_dataset, dataset_class)
    if overwrite or not os.path.exists(root):
        dataset_class.download.get_data(root)
        
def load_datasets(info_datasets, root_dataset, root_dataframe, **kwargs):
    return [load_dataset(info_dataset, root_dataset, root_dataframe, **kwargs) for info_dataset in info_datasets]

def load_dataset(info_dataset, root_dataset, root_dataframe, overwrite=False):
    if not os.path.exists(root_dataset):
        raise(Exception('Data not found. Download them first.'))
    if not os.path.exists(root_dataframe):
        os.makedirs(root_dataframe)
    dataset_class = info_dataset[0]

    root = get_dataset_folder(root_dataset, dataset_class)
    df_path = get_dataframe_path(root_dataframe, dataset_class)
    if overwrite or not os.path.exists(df_path):
        dataset = dataset_class(root, None, download=False)
        dataset.df.to_pickle(df_path)
    else:
        df = pd.read_pickle(df_path)
        dataset = dataset_class(root, df, download=False)
    return dataset
