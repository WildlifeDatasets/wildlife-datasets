import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class PolarBearVidID(DatasetFactory):
    summary = summary['PolarBearVidID']
    url = 'https://zenodo.org/records/7564529/files/PolarBearVidID.zip?download=1'
    archive = 'PolarBearVidID.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        metadata = pd.read_csv(os.path.join(self.root, 'animal_db.csv'))
        data = utils.find_images(self.root)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'video': data['file'].str[7:10].astype(int),
            'id': data['path'].astype(int)
        })
        df = pd.merge(df, metadata, on='id', how='left')
        df.rename({'name': 'identity', 'sex': 'gender'}, axis=1, inplace=True)
        df = df.drop(['id', 'zoo', 'tracklets'], axis=1)
        return self.finalize_catalogue(df)
