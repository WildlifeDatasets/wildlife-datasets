import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class CatIndividualImages(DatasetFactory):
    summary = summary['CatIndividualImages']
    archive = 'cat-individuals.zip'
    
    @classmethod
    def _download(cls):
        command = f"datasets download -d timost1234/cat-individuals --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#catindividualimages'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        # Remove 85 duplicate images
        idx = folders[2].isnull()
        data = data[idx]
        folders = folders[idx]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)
