import os
import pandas as pd

class Metadata():
    """Class for storing metadata.

    Attributes:
      df (pd.DataFrame): A dataframe of the metadata.
    """

    def __init__(self, path: str):
        """Loads the metadata from a csv file into a dataframe.

        The `animals` column is converted to a list.

        Args:
            path (str): Path of the csv file.
        """

        df = pd.read_csv(path, index_col='name')
        if 'animals' in df.columns:
            df.loc[df['animals'].isnull(), 'animals'] = '{}'
            df['animals'] = df['animals'].apply(lambda x: eval(x))
        self.df = df

    def __getitem__(self, item):
        return self.df.loc[item].dropna().to_dict()

# Load the included metadata    
metadata = Metadata(os.path.join(os.path.dirname(__file__), 'metadata.csv'))
