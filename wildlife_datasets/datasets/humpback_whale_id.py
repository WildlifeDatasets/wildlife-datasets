import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadKaggle

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://www.kaggle.com/competitions/humpback-whale-identification/rules#7.-competition-data.',
    'url': 'https://www.kaggle.com/competitions/humpback-whale-identification',
    'publication_url': None,
    'cite': 'humpbackwhale',
    'animals': {'whale'},
    'animals_simple': 'whales',
    'real_animals': True,
    'year': 2019,
    'reported_n_total': None,
    'reported_n_individuals': None,
    'wild': True,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'very long',
    'size': 5911,
}

class HumpbackWhaleID(DownloadKaggle, DatasetFactory):
    summary = summary
    kaggle_url = 'humpback-whale-identification'
    kaggle_type = 'competitions'

    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data.loc[data['Id'] == 'new_whale', 'Id'] = self.unknown_name
        df1 = pd.DataFrame({
            'image_id': data['Image'].str.split('.', expand=True)[0],
            'path': 'train' + os.path.sep + data['Image'],
            'identity': data['Id'],
            'original_split': 'train'
            })
        
        # Find all testing images
        test_files = utils.find_images(os.path.join(self.root, 'test'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        # Create the testing dataframe
        df2 = pd.DataFrame({
            'image_id': test_files.str.split('.', expand=True)[0],
            'path': 'test' + os.path.sep + test_files,
            'identity': self.unknown_name,
            'original_split': 'test'
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])
        return self.finalize_catalogue(df)
