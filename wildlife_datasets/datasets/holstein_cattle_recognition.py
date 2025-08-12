import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .utils import find_images
from .summary import summary

class HolsteinCattleRecognition(DatasetFactory):
    summary = summary['HolsteinCattleRecognition']
    url = 'https://dataverse.nl/api/access/dataset/:persistentId/?persistentId=doi:10.34894/O1ZBSA'
    archive = 'dataset.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        # Extract all archives in the original archive
        zip_files = find_images('.', img_extensions='zip')
        for _, zip_file in zip_files.iterrows():
            file_name = os.path.join(zip_file['path'], zip_file['file'])
            utils.extract_archive(file_name, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        data = utils.find_images(self.root)

        # Extract the full images only
        idx = data['path'].apply(lambda x: '__MACOSX' not in x)
        data = data[idx]
        idx1 = data['file'].apply(lambda x: os.path.splitext(x)[0].endswith('full photo'))
        idx2 = data['file'].apply(lambda x: os.path.splitext(x)[0].endswith('full photo (2)'))
        data = data[idx1 + idx2]

        folders = data['path'].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        # Finalize the dataframe
        data['identity'] = folders[n_folders]
        data['path'] = data['path'] + os.path.sep + data['file']
        data['image_id'] = data['file'].str[4:8].astype(str)
        data = data.drop('file', axis=1)

        return self.finalize_catalogue(data)
