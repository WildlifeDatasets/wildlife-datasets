import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class MPDD(DatasetFactory):
    summary = summary['MPDD']
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/v5j6m8dzhv-1.zip'
    archive = 'MPDD.zip'
    
    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        utils.extract_archive(os.path.join('Multi-pose dog dataset', 'MPDD.zip'), delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        identity = data['file'].apply(lambda x: int(x.split('_')[0]))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'],
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'original_split': folders[2]
        })
        return self.finalize_catalogue(df)
