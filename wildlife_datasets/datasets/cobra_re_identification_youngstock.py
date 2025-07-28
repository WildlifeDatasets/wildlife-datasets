import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class CoBRAReIdentificationYoungstock(DatasetFactory):
    summary = summary['CoBRAReIdentificationYoungstock']
    url = 'https://zenodo.org/records/15018518/files/re_identification_youngstock.zip?download=1'
    archive = 're_identification_youngstock.zip'

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
        n_folders = max(folders.columns)

        # Extract information
        identity = folders[n_folders]
        orientation = folders[n_folders-1]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(identity + orientation + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'orientation': orientation,
        })
        return self.finalize_catalogue(df)
