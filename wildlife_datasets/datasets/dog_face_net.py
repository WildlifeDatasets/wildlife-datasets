import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://github.com/GuillaumeMougeot/DogFaceNet',
    'publication_url': 'https://link.springer.com/chapter/10.1007/978-3-030-29894-4_34',
    'cite': 'mougeot2019deep',
    'animals': {'dog'},
    'animals_simple': 'dogs',
    'real_animals': True,
    'year': 2019,
    'reported_n_total': 8363,
    'reported_n_individuals': 1393,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 76,
}

class DogFaceNet(DatasetFactory):
    summary = summary
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
