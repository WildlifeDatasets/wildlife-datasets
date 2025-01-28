import os
import numpy as np
import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset

def create_dataset(df, check_files=False, **kwargs):
    dataset = WildlifeDataset(df=df, check_files=check_files, **kwargs)
    dataset.df = dataset.finalize_catalogue(dataset.df)
    return dataset

def load_datasets(dataset_names):
    datasets = []
    for dataset_name in dataset_names:
        csv_name = dataset_name.__name__ + '.csv'
        csv_path = os.path.join('wildlife_datasets', 'tests', 'csv', csv_name)
        df = pd.read_csv(csv_path, index_col=False)
        datasets.append(create_dataset(df))
    return datasets

def add_datasets(datasets, skip_rows=100, ratio_unknown=0.2, ratio_years=0.2):
    for i in range(len(datasets)):
        # Change indexing
        df = datasets[i].df.copy()
        if len(df) >= 2*skip_rows:
            df = df.iloc[skip_rows:]
            datasets.append(create_dataset(df))
        # Add unknown individuals
        df = datasets[i].df.copy()
        n_unknown = np.round(len(df)*ratio_unknown).astype(int)
        idx = np.random.permutation(range(len(df)))[:n_unknown]
        df.loc[df.index[idx], 'identity'] = 'unknown'
        datasets.append(create_dataset(df))
        # Add new years
        if 'date' in df.columns:
            df = datasets[i].df.copy()
            n_years = np.round(len(df)*ratio_years).astype(int)
            df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.date())
            df.loc[df.index[:n_years], 'date'] = df['date'].iloc[:n_years] + pd.offsets.DateOffset(years=10)
            df['date'] = pd.to_datetime(df['date'])
            datasets.append(create_dataset(df))
    for i in range(len(datasets)):
        df = datasets[i].df.copy()
        datasets.append(create_dataset(df, col_label='id', col_path='cesticka'))
    return datasets
