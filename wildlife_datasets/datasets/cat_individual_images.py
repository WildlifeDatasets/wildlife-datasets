import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadKaggle

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://www.kaggle.com/datasets/timost1234/cat-individuals',
    'publication_url': None,
    'cite': 'catindividuals',
    'animals': {'cat'},
    'animals_simple': 'cats',
    'real_animals': True,
    'year': 2020,
    'reported_n_total': 13536,
    'reported_n_individuals': 518,
    'wild': False,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 11000,
}

class CatIndividualImages(DownloadKaggle, DatasetFactory):
    summary = summary
    kaggle_url = 'timost1234/cat-individuals'
    kaggle_type = 'datasets'
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        # Remove 85 duplicate images
        idx = folders[2].isnull()
        data = data[idx]
        folders = folders[idx]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)
