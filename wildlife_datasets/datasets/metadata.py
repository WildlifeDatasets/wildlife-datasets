import os
import pandas as pd

class Metadata():
    def __init__(self, path):
        dir = os.path.dirname(__file__)
        df = pd.read_csv(os.path.join(dir, 'metadata.csv'), index_col='name')
        if 'animals' in df.columns:
            df.loc[df['animals'].isnull(), 'animals'] = '{}'
            df['animals'] = df['animals'].apply(lambda x: eval(x))
        self.df = df

    def __getitem__(self, item):
        return self.df.loc[item].dropna().to_dict()
    
metadata = Metadata('metadata.csv')