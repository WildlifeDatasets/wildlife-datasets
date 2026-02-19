import os

import numpy as np
import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
    'url': 'https://zindi.africa/competitions/turtle-recall-conservation-challenge',
    'publication_url': None,
    'cite': 'zinditurtles',
    'animals': {'sea turtle'},
    'animals_simple': 'sea turtles',
    'real_animals': True,
    'year': 2022,
    'reported_n_total': None,
    'reported_n_individuals': None,
    'wild': False,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'unknown',
    'size': 6482,
}

class ZindiTurtleRecall(DownloadURL, WildlifeDataset):
    summary = summary
    downloads = [
        ('https://storage.googleapis.com/dm-turtle-recall/train.csv', 'train.csv'),
        ('https://storage.googleapis.com/dm-turtle-recall/extra_images.csv', 'extra_images.csv'),
        ('https://storage.googleapis.com/dm-turtle-recall/test.csv', 'test.csv'),
        ('https://storage.googleapis.com/dm-turtle-recall/images.tar', 'images.tar'),
    ]
    
    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the training images
        assert self.root is not None
        data_train = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data_train['split'] = 'train'

        # Load information about the testing images
        data_test = pd.read_csv(os.path.join(self.root, 'test.csv'))
        data_test['split'] = 'test'

        # Load information about the additional images
        data_extra = pd.read_csv(os.path.join(self.root, 'extra_images.csv'))
        data_extra['split'] = np.nan        

        # Finalize the dataframe
        data = pd.concat([data_train, data_test, data_extra])
        data = data.reset_index(drop=True)        
        data.loc[data['turtle_id'].isnull(), 'turtle_id'] = self.unknown_name
        df = pd.DataFrame({
            'image_id': data['image_id'],
            'path': 'images' + os.path.sep + data['image_id'] + '.JPG',
            'identity': data['turtle_id'],
            'orientation': data['image_location'].str.lower(),
            'original_split': data['split'],
        })
        return self.finalize_catalogue(df)