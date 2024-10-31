import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class AerialCattle2017(DatasetFactory):
    summary = summary['AerialCattle2017']
    url = 'https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip'
    archive = '3owflku95bxsx24643cybxu3qh.zip'

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

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'video': folders[2].astype(int),
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
