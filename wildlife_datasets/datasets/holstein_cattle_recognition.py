import os
import pandas as pd
from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'CC0 1.0 Universal',
    'licenses_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
    'url': 'https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/O1ZBSA',
    'publication_url': 'https://link.springer.com/chapter/10.1007/978-3-030-29891-3_10',
    'cite': 'bhole2019computer',
    'animals': {'cow'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2021,
    'reported_n_total': 1237,
    'reported_n_individuals': 136,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': 'short',
    'size': 500,
}

class HolsteinCattleRecognition(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://dataverse.nl/api/access/dataset/:persistentId/?persistentId=doi:10.34894/O1ZBSA'
    archive = 'dataset.zip'

    @classmethod
    def _extract(cls):
        super()._extract()
        # Extract all archives in the original archive
        zip_files = utils.find_images('.', img_extensions='zip')
        for _, zip_file in zip_files.iterrows():
            file_name = os.path.join(zip_file['path'], zip_file['file'])
            utils.extract_archive(file_name, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
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
