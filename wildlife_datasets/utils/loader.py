import os
import pandas as pd
from typing import List

def get_dataset_folder(root_dataset: str, dataset_class) -> pd.DataFrame:
    '''
    Gets path to the saved dataset
    '''
    # TODO: hacky solution
    if dataset_class.__name__.endswith('Segmented'):
        return os.path.join(root_dataset, dataset_class.__name__[:-9])
    else:
        return os.path.join(root_dataset, dataset_class.__name__)

def get_dataframe_path(root_dataframe: str, dataset_class) -> str:
    '''
    Gets path to the pickled dataframe
    '''
    return os.path.join(root_dataframe, dataset_class.__name__ + '.pkl')

def download_datasets(class_datasets, root_dataset: str, **kwargs) -> None:
    '''
    Downloads multiple datasets
    '''    
    for class_dataset in class_datasets:
        download_dataset(class_dataset, root_dataset, **kwargs)

def download_dataset(class_dataset, root_dataset: str, overwrite:bool=False) -> None:
    '''
    Downloads one dataset
    '''
    root = get_dataset_folder(root_dataset, class_dataset)
    if overwrite or not os.path.exists(root):
        class_dataset.download.get_data(root)
    
def load_datasets(
        class_datasets: List[object],
        root_dataset: str,
        root_dataframe: str,
        **kwargs
        ) -> List[pd.DataFrame]:
    '''
    Runs load_dataset for multiple datasets.
    '''
    return [load_dataset(class_dataset, root_dataset, root_dataframe, **kwargs) for class_dataset in class_datasets]

def load_dataset(
        class_dataset: object,
        root_dataset: str,
        root_dataframe: str,
        overwrite: bool = False
        ) -> pd.DataFrame:
    '''
    Loads the dataframe corresponding to the dataset.
    If the dataframe was already saved in a pkl file, it loads it.
    Otherwise, it creates the dataframe and saves it in a pkl file.
    '''
    # Check if the dataset is downloaded.
    if not os.path.exists(root_dataset):
        raise(Exception('Data not found. Download them first.'))
    
    # Get paths of the dataset and the pickled dataframe/
    root = get_dataset_folder(root_dataset, class_dataset)
    df_path = get_dataframe_path(root_dataframe, class_dataset)
    if overwrite or not os.path.exists(df_path):
        # Create the dataframe, save it and create the dataset
        dataset = class_dataset(root, None, download=False)
        if not os.path.exists(root_dataframe):
            os.makedirs(root_dataframe)
        dataset.df.to_pickle(df_path)
    else:
        # Load the dataframe and create the dataset
        df = pd.read_pickle(df_path)
        dataset = class_dataset(root, df, download=False)
    return dataset
