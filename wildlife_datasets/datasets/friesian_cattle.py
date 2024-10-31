import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class FriesianCattle2015(DatasetFactory):
    outdated_dataset = True
    summary = summary['FriesianCattle2015']
    url = 'https://data.bris.ac.uk/datasets/wurzq71kfm561ljahbwjhx9n3/wurzq71kfm561ljahbwjhx9n3.zip'
    archive = 'wurzq71kfm561ljahbwjhx9n3.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        # Extract information from the folder structure
        split = folders[1].replace({'Cows-testing': 'test', 'Cows-training': 'train'})
        identity = folders[2].str.strip('Cow').astype(int)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)

class FriesianCattle2015v2(FriesianCattle2015):
    outdated_dataset = False

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove all identities in training as they are duplicates
        idx_remove = ['Cows-training' in path for path in df.path]
        df = df[~np.array(idx_remove)]

        # Remove specified individuals as they are duplicates
        identities_to_remove = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 37]
        return self.fix_labels_remove_identity(df, identities_to_remove)


class FriesianCattle2017(DatasetFactory):
    summary = summary['FriesianCattle2017']
    url = 'https://data.bris.ac.uk/datasets/2yizcfbkuv4352pzc32n54371r/2yizcfbkuv4352pzc32n54371r.zip'
    archive = '2yizcfbkuv4352pzc32n54371r.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)
