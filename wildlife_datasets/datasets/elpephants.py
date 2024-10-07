import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .metadata import metadata

class ELPephants(DatasetFactory):
    metadata = metadata['ELPephants']
    archive = 'ELPephant (elephant ID system).zip'

    @classmethod
    def _download(cls):
        exception_text = '''Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#elpephants'''
        raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': data['file'].apply(self.extract_identity),
            'date': data['file'].apply(self.extract_date),
            'orientation': data['file'].apply(self.extract_orientation),
        })
        return self.finalize_catalogue(df)

    def extract_identity(self, x):
        return int(x.strip().split('_')[0])

    def extract_date(self, x):
        for y in x.lower().split('_')[::-1]:
            date = self.extract_date_part(y)
            if isinstance(date, str):
                return date
        return np.nan
        
    def extract_date_part(self, x):
        i_end = -np.inf
        conversion = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
        # Extract month
        for month_try in conversion:
            if month_try in x:
                i = x.index(month_try)
                if i > i_end:
                    month = month_try
                    i_end = i
        # Month not found
        if i_end == -np.inf:
            return np.nan
        # Extract string after month and remove extension
        year = x[i_end+len(month):]
        year = os.path.splitext(year)[0]
        year = year.strip()
        # Extract year from the beginning or end
        if len(year) >= 4:
            if represents_int(year[:4]):
                return f'{conversion[month]}-1-{int(year[:4])}'
            elif represents_int(year[-4:]):
                return f'{conversion[month]}-1-{int(year[-4:])}'
        return np.nan
        
    def extract_orientation(self, x):
        x = x.lower()
        for orientation in ['left', 'right', 'front']:
            if orientation in x:
                return orientation
        return np.nan         

def represents_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True
