import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .metadata import metadata

class WildlifeReID10k(DatasetFactory):
    metadata = metadata['WildlifeReID10k']
    archive = 'wildlifereid-10k.zip'

    @classmethod
    def _download(cls):
        # TODO: check
        command = f"datasets download -d wildlifedatasets/wildlifereid-10k --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#wildlifereid-10k'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        df['path'] = 'images/' + df['path']
        return self.finalize_catalogue(df)
