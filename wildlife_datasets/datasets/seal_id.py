import os
import shutil

import pandas as pd

from . import utils
from .datasets import WildlifeDataset

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53',
    'publication_url': 'https://www.mdpi.com/1424-8220/22/19/7602',
    'cite': 'nepovinnykh2022sealid',
    'animals': {'ringed seal'},
    'animals_simple': 'seals',
    'real_animals': True,
    'year': 2022,
    'reported_n_total': 2080,
    'reported_n_individuals': 57,
    'wild': True,
    'clear_photos': False,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': '10 years',
    'size': 1816,
}

class SealID(WildlifeDataset):
    summary = summary
    prefix = 'source_'
    archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
    
    @classmethod
    def _download(cls, url=None):
        if url is None:
            raise(Exception('URL must be provided for SealID.\nCheck https://wildlifedatasets.github.io/wildlife-datasets/preprocessing/#sealid'))
        utils.download_url(url, cls.archive)

    @classmethod
    def _extract(cls, **kwargs):
        utils.extract_archive(cls.archive, delete=True)
        utils.extract_archive(os.path.join('SealID', 'full images.zip'), delete=True)
        utils.extract_archive(os.path.join('SealID', 'patches.zip'), delete=True)
        
        # Create new folder for segmented images
        folder_new = os.getcwd() + 'Segmented'
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)
        
        # Move segmented images to new folder
        folder_move = os.path.join('patches', 'segmented')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join('full images', 'segmented_database')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join('full images', 'segmented_query')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        file_copy = os.path.join('patches', 'annotation.csv')
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))
        file_copy = os.path.join('full images', 'annotation.csv')
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))            

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        assert self.root is not None
        data = pd.read_csv(os.path.join(self.root, 'full images', 'annotation.csv'))

        # Finalize the dataframe
        df = pd.DataFrame({    
            'image_id': data['file'].str.split('.', expand=True)[0],
            'path': 'full images' + os.path.sep + self.prefix + data['reid_split'] + os.path.sep + data['file'],
            'identity': data['class_id'].astype(int),
            'original_split': data['segmentation_split'].replace({'training': 'train', 'testing': 'test', 'validation': 'val'}),
            'original_split_reid': data['reid_split'],
        })
        return self.finalize_catalogue(df)


class SealIDSegmented(SealID):
    prefix = 'segmented_'
    warning = '''You are trying to download or extract a segmented dataset.
        It is already included in its non-segmented version.
        Skipping.'''
    
    @classmethod
    def get_data(cls, *args, **kwargs):
        print(cls.warning)

    @classmethod
    def _download(cls, *args, **kwargs):
        print(cls.warning)

    @classmethod
    def _extract(cls, *args, **kwargs):
        print(cls.warning)
