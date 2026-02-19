import os

import pandas as pd

from ..datasets import WildlifeDataset


def get_dataset_folder(root_dataset: str, class_dataset: type) -> str:
    """Creates path to the dataset data.

    Args:
        root_dataset (str): Path where all datasets are stored.
        class_dataset (type): Type of WildlifeDataset.

    Returns:
        Path to the stored data.
    """

    return os.path.join(root_dataset, class_dataset.display_name())


def get_dataframe_path(root_dataframe: str, class_dataset: type) -> str:
    """Creates path to the pickled dataframe.

    Args:
        root_dataframe (str): Path where all dataframes are stored.
        class_dataset (type): Type of WildlifeDataset.

    Returns:
        Path to the dataframe.
    """

    return os.path.join(root_dataframe, class_dataset.__name__ + ".pkl")


def load_datasets(
    class_datasets: list[type], root_dataset: str, root_dataframe: str, **kwargs
) -> list[WildlifeDataset]:
    """Loads multiple datasets as described in `load_dataset`.

    Args:
        class_datasets (List[type]): List of types of WildlifeDataset to download.
        root_dataset (str): Path where all datasets are stored.
        root_dataframe (str): Path where all dataframes are stored.

    Returns:
        The list of loaded datasets.
    """

    return [load_dataset(class_dataset, root_dataset, root_dataframe, **kwargs) for class_dataset in class_datasets]


def load_dataset(
    class_dataset: type, root_dataset: str, root_dataframe: str, overwrite: bool = False, **kwargs
) -> WildlifeDataset:
    """Loads dataset from a pickled dataframe or creates it.

    If the dataframe is already saved in a pkl file, it loads it.
    Otherwise, it creates the dataframe and saves it in a pkl file.

    Args:
        class_dataset (type): Type of WildlifeDataset to load.
        root_dataset (str): Path where all datasets are stored.
        root_dataframe (str): Path where all dataframes are stored.
        overwrite (bool, optional): Whether the pickled dataframe should be overwritten.

    Returns:
        The loaded dataset.
    """

    # Check if the dataset is downloaded.
    if not os.path.exists(root_dataset):
        raise (Exception("Data not found. Download them first."))

    # Get paths of the dataset and the pickled dataframe
    root = get_dataset_folder(root_dataset, class_dataset)
    df_path = get_dataframe_path(root_dataframe, class_dataset)
    if not class_dataset.determined_by_df:
        # Create the dataframe, no point in saving as it is not determined by it
        dataset = class_dataset(root, None, **kwargs)
    elif overwrite or not os.path.exists(df_path):
        # Create the dataframe, save it and create the dataset
        dataset = class_dataset(root, None, **kwargs)
        if not os.path.exists(root_dataframe):
            os.makedirs(root_dataframe)
        dataset.df.to_pickle(df_path)
    else:
        # Load the dataframe and create the dataset
        df = pd.read_pickle(df_path)
        dataset = class_dataset(root, df, **kwargs)
    return dataset
