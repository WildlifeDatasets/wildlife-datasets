import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadURL

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://data.mendeley.com/datasets/v5j6m8dzhv/1',
    'publication_url': None,
    'cite': 'he2023animal',
    'animals': {'dog'},
    'animals_simple': 'dogs',
    'real_animals': True,
    'year': 2023,
    'reported_n_total': 1657,
    'reported_n_individuals': 192,
    'wild': False,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': False,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 29.6,
}

class MPDD(DownloadURL, DatasetFactory):
    summary = summary
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/v5j6m8dzhv-1.zip'
    archive = 'MPDD.zip'
    
    @classmethod
    def _extract(cls):
        super()._extract()
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
