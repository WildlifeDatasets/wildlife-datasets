import os
import shutil
import numpy as np
import pandas as pd
from . import utils
from .datasets import WildlifeDataset

summary = {
    'licenses': None,
    'licenses_url': None,
    'url': 'https://github.com/AndreCFerreira/Bird_individualID',
    'publication_url': 'https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13436',
    'cite': 'ferreira2020deep',
    'animals': {'sociable weaver', 'zebra finch', 'great tit'},
    'animals_simple': 'birds',
    'real_animals': True,
    'year': 2019,
    'reported_n_total': 50643,
    'reported_n_individuals': 50,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': False,
    'cropped': False,
    'span': '15 days',
    'size': 70656,
}

class BirdIndividualID(WildlifeDataset):
    summary = summary
    prefix1 = 'Original_pictures'
    prefix2 = 'IndividualID'
    url = 'https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA'
    archive = 'ferreira_et_al_2020.zip'

    @classmethod
    def _download(cls) -> None:
        exception_text = '''Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#birdindividualid'''
        raise Exception(exception_text)
        # utils.gdown_download(cls.url, cls.archive, exception_text=exception_text)
    
    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

        # Create new folder for segmented images
        folder_new = os.getcwd() + 'Segmented'
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)

        # Move segmented images to new folder
        folder_move = 'Cropped_pictures'
        shutil.move(folder_move, os.path.join(folder_new, folder_move))

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        path = os.path.join(self.root, self.prefix1, self.prefix2)
        data = utils.find_images(path)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Remove images with multiple labels
        idx = folders[2].str.contains('_')
        data = data.loc[~idx]
        folders = folders.loc[~idx]

        # Remove some problems with the sociable_weavers/Test_dataset
        if folders.shape[1] == 4:
            idx = folders[3].isnull()
            folders.loc[~idx, 2] = folders.loc[~idx, 3]

        # Extract information from the folder structure
        split = folders[1].replace({'Test_datasets': 'test', 'Test': 'test', 'Train': 'train', 'Val': 'val'})
        identity = folders[2]
        species = folders[0]
        date = data['file'].apply(lambda x: self.extract_date(x))

        # Finalize the dataframe
        df1 = pd.DataFrame({    
            'image_id': utils.create_id(split + data['file']),
            'path': self.prefix1 + os.path.sep + self.prefix2 + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'date': date,
            'species': species,
            'original_split': split,
        })

        # Add images without labels
        path = os.path.join(self.root, self.prefix1, 'New_birds')
        data = utils.find_images(path)
        species = data['path']
        date = data['file'].apply(lambda x: self.extract_date(x))

        # Finalize the dataframe
        df2 = pd.DataFrame({    
            'image_id': utils.create_id(data['file']),
            'path': self.prefix1 + os.path.sep + 'New_birds' + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': self.unknown_name,
            'date': date,
            'species': species,
            'original_split': np.nan,
        })
        df = pd.concat([df1, df2])
        return self.finalize_catalogue(df)

    def extract_date(self, x):
        x = x.replace(' ', '_')
        date = x.split('_')[-2]
        if date == '2018':
            date = f'2018-{x.split("_")[-3]}-{x.split("_")[-4]}'
        elif date == '1':
            date = x.split('_')[-3]
            date = f'20{date[4:]}-{date[2:4]}-{date[:2]}'
        elif date == 'very':
            date = x.split('_')[-4]
            date = f'20{date[4:]}-{date[2:4]}-{date[:2]}'
        elif date == 'feeder' or date == 'feederB':
            date = f'2018-{x.split("_")[-5]}-{x.split("_")[-4]}'
        elif len(date) == 6 and date != 'feeder':
            date = f'20{date[4:]}-{date[2:4]}-{date[:2]}'
        elif len(date) == 8:
            date = x.split('_')[-3]            
        elif len(date) != 10:
            date = np.nan
        return date

class BirdIndividualIDSegmented(BirdIndividualID):
    prefix1 = 'Cropped_pictures'
    prefix2 = 'IndividuaID'
    warning = '''You are trying to download or extract a segmented dataset.
        It is already included in its non-segmented version.
        Skipping.'''
    
    @classmethod
    def get_data(cls, root, force=False, **kwargs):
        print(cls.warning)

    @classmethod
    def _download(cls):
        print(cls.warning)

    @classmethod
    def _extract(cls):
        print(cls.warning)
