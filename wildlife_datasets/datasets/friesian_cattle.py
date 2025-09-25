import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory

summary_2015 = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/wurzq71kfm561ljahbwjhx9n3',
    'publication_url': 'https://ieeexplore.ieee.org/abstract/document/7532404',
    'cite': 'andrew2016automatic',
    'animals': {'Friesian cattle'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2016,
    'reported_n_total': 377,
    'reported_n_individuals': 40,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': '1 day',
    'size': 76,
}

summary_2017 = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r',
    'publication_url': 'https://openaccess.thecvf.com/content_ICCV_2017_workshops/w41/html/Andrew_Visual_Localisation_and_ICCV_2017_paper.html',
    'cite': 'andrew2017visual',
    'animals': {'Friesian cattle'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2017,
    'reported_n_total': 940,
    'reported_n_individuals': 89,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': '1 day',
    'size': 343,
}

class FriesianCattle2015(DatasetFactory):
    outdated_dataset = True
    summary = summary_2015
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
    summary = summary_2017
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
