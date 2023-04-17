import os
import numpy as np
import pandas as pd

def load_datasets(dataset_names):
    dfs = []
    for dataset_name in dataset_names:
        csv_name = dataset_name.__name__ + '.csv'
        csv_path = os.path.join('wildlife_datasets', 'tests', 'csv', csv_name)

        df = pd.read_csv(csv_path)
        df = df.drop('Unnamed: 0', axis=1)
        dfs.append(df)
    return dfs

def add_datasets(dfs, skip_rows=100, ratio_unknown=0.2):
    for i in range(len(dfs)):
        df = dfs[i].copy()
        # Change indexing
        if len(df) >= 2*skip_rows:
            df = df.iloc[skip_rows:]
            dfs.append(df)
        # Add unknown individuals
        n_unknown = np.round(len(df)*ratio_unknown).astype(int)
        idx = np.random.permutation(range(len(df)))[:n_unknown]
        df['identity'].iloc[idx] = 'unknown'
        dfs.append(df)
    return dfs
