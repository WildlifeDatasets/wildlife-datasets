import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://www.kaggle.com/competitions/happy-whale-and-dolphin/rules/#7.-competition-data.',
    'url': 'https://www.kaggle.com/competitions/happy-whale-and-dolphin',
    'publication_url': 'https://link.springer.com/article/10.1007/s42991-021-00180-9',
    'cite': 'cheeseman2021advanced',
    'animals': {'dolphin', 'whale'},
    'animals_simple': 'dolphins+whales',
    'real_animals': True,
    'year': 2022,
    'reported_n_total': None,
    'reported_n_individuals': None,
    'wild': True,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': 'very long',
    'size': 61912,
}

class HappyWhale(DatasetFactory):
    summary = summary
    archive = 'happy-whale-and-dolphin.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c happy-whale-and-dolphin --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#happywhale'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        try:
            utils.extract_archive(cls.archive, delete=True)
        except:
            exception_text = '''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#happywhale'''
            raise Exception(exception_text)
    
    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = pd.DataFrame({
            'image_id': data['image'].str.split('.', expand=True)[0],
            'path': 'train_images' + os.path.sep + data['image'],
            'identity': data['individual_id'],
            'species': data['species'],
            'original_split': 'train'
            })

        test_files = utils.find_images(os.path.join(self.root, 'test_images'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        # Load the testing data
        df2 = pd.DataFrame({
            'image_id': test_files.str.split('.', expand=True)[0],
            'path': 'test_images' + os.path.sep + test_files,
            'identity': self.unknown_name,
            'species': np.nan,
            'original_split': 'test'
            })
        
        # Finalize the dataframe        
        df = pd.concat([df1, df2])
        return self.finalize_catalogue(df)

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong species names            
        replace_identity = [
            ('bottlenose_dolpin', 'bottlenose_dolphin'),
            ('kiler_whale', 'killer_whale'),
        ]
        return self.fix_labels_replace_identity(df, replace_identity, col='species')
