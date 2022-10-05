import os
import numpy as np
import pandas as pd
from .. import datasets

info_datasets_full = [
    datasets.AAUZebraFishID,
    datasets.AerialCattle2017,
    datasets.ATRW,
    datasets.BelugaID,
    datasets.BirdIndividualID,
    datasets.BirdIndividualIDSegmented,
    datasets.CTai,
    datasets.CZoo,
    datasets.Cows2021,
    datasets.Drosophila,
    datasets.FriesianCattle2015,
    datasets.FriesianCattle2017,
    datasets.GiraffeZebraID,
    datasets.Giraffes,
    datasets.HappyWhale,
    datasets.HumpbackWhaleID,
    datasets.HyenaID2022,
    datasets.IPanda50,
    datasets.LeopardID2022,
    datasets.LionData,
    datasets.MacaqueFaces,
    datasets.NDD20,
    datasets.NOAARightWhale,
    datasets.NyalaData,
    datasets.OpenCows2020,
    datasets.SealID,
    datasets.SealIDSegmented,
    datasets.SeaTurtleID,
    datasets.SMALST,
    datasets.StripeSpotter,
    datasets.WhaleSharkID,
    datasets.WNIGiraffes,
    datasets.ZindiTurtleRecall
]

def unique_datasets_list(datasets_list):
    _, idx = np.unique([dataset[0].__name__ for dataset in datasets_list], return_index=True)
    idx = np.sort(idx)

    datasets_list_red = []
    for i in idx:
        datasets_list_red.append(datasets_list[i])

    return datasets_list_red

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
