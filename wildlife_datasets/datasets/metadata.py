import os
import pandas as pd

# TODO: add documentation
class Metadata():
    def __init__(self, path):
        df = pd.read_csv(path, index_col='name')
        if 'animals' in df.columns:
            df.loc[df['animals'].isnull(), 'animals'] = '{}'
            df['animals'] = df['animals'].apply(lambda x: eval(x))
        self.df = df

    def __getitem__(self, item):
        return self.df.loc[item].dropna().to_dict()
    
metadata = Metadata(os.path.join(os.path.dirname(__file__), 'metadata.csv'))
