import os
import pandas as pd

def load_datasets(dataset_names, skip_rows=100):
    dfs = []
    for dataset_name in dataset_names:
        csv_name = dataset_name.__name__ + '.csv'
        csv_path = os.path.join('wildlife_datasets', 'tests', 'csv', csv_name)

        df = pd.read_csv(csv_path)
        df = df.drop('Unnamed: 0', axis=1)
        if len(df) >= 2*skip_rows:
            dfs.append(dataset_name('', df).df.iloc[skip_rows:])
    return dfs