import os

import numpy as np
import pandas as pd

from . import utils
from .datasets import WildlifeDataset

summary = {
    'licenses': None,
    'licenses_url': None,
    'url': 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021',
    'publication_url': 'https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13577',
    'cite': 'miele2021revisiting',
    'animals': {'giraffe'},
    'animals_simple': 'giraffes',
    'real_animals': True,
    'year': 2021,
    'reported_n_total': None,
    'reported_n_individuals': None,
    'wild': True,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': 'unknown',
    'size': 1719,
}

class Giraffes(WildlifeDataset):
    summary = summary

    @classmethod
    def _download(cls):
        url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/'
        command = f"wget -rpk -l 10 -np -c --random-wait -U Mozilla {url} -P '.' "
        exception_text = '''Download works only on Linux. Please download it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#giraffes'''
        if os.name == 'posix':
            os.system(command)
        else:
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        pass

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure
        clusters = folders.iloc[:, -2] == 'clusters'
        data, folders = data[clusters], folders[clusters]

        # Finalize the dataframe
        df = pd.DataFrame({    
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders.iloc[:, -1],
            'date': data['file'].apply(lambda x: self.extract_date(x))
        })
        return self.finalize_catalogue(df)
    
    def extract_date(self, x):
        date = x.split('_')[1]
        if date == 'None':
            return np.nan
        else:
            return f'{date[:4]}-{date[4:6]}-{date[6:]}'
