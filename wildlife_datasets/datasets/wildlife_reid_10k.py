import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k',
    'url': 'https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k',
    'publication_url': 'https://arxiv.org/abs/2406.09211',
    'cite': 'adam',
    'animals': {'multiple'},
    'animals_simple': 'multiple',
    'real_animals': True,
    'year': 2024,
    'reported_n_total': 214262,
    'reported_n_individuals': 1034478,
    'wild': True,
    'clear_photos': None,
    'pose': 'multiple',
    'unique_pattern': None,
    'from_video': False,
    'cropped': True,
    'span': 'very long',
    'size': 24760,
}

class WildlifeReID10k(DatasetFactory):
    summary = summary
    archive = 'wildlifereid-10k.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/wildlifereid-10k --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#wildlifereid-10k'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.root, 'metadata.csv'), low_memory=False)
        df['image_id'] = range(len(df))
        return self.finalize_catalogue(df)
