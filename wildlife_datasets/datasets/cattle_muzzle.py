import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class CattleMuzzle(DatasetFactory):
    # TODO: finish
    #summary = summary['CattleMuzzle']
    url = 'https://cloud.une.edu.au/index.php/s/eMwaHAPK08dCDru/download'
    archive = 'Cattle Identification (supplementary material).zip'

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
        image_id = data['file'].apply(lambda x: x.split('_')[1].split('.')[0])
        identity = folders[n_folders]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': image_id,
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
        })
        return self.finalize_catalogue(df)
