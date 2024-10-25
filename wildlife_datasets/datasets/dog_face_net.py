import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class DogFaceNet(DatasetFactory):
    summary = summary['DogFaceNet']
    url = 'https://github.com/GuillaumeMougeot/DogFaceNet/releases/download/dataset/DogFaceNet_Dataset_224_1.zip'
    archive = 'DogFaceNet_Dataset_224_1.zip'

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
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)
