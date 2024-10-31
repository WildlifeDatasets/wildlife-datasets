import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class ZindiTurtleRecall(DatasetFactory):
    summary = summary['ZindiTurtleRecall']

    @classmethod
    def _download(cls):
        downloads = [
            ('https://storage.googleapis.com/dm-turtle-recall/train.csv', 'train.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/extra_images.csv', 'extra_images.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/test.csv', 'test.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/images.tar', 'images.tar'),
        ]
        for url, file in downloads:
            utils.download_url(url, file)

    @classmethod
    def _extract(cls):
        utils.extract_archive('images.tar', 'images', delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the training images
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