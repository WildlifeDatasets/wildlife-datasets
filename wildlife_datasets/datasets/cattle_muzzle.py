import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadURL

summary = {
    'licenses': None,
    'licenses_url': None,
    'url': 'https://cloud.une.edu.au/index.php/s/eMwaHAPK08dCDru',
    'publication_url': 'https://www.mdpi.com/2073-4395/11/11/2365',
    'cite': 'shojaeipour2021automated',
    'animals': {'cow'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2021,
    'reported_n_total': 2900,
    'reported_n_individuals': 300,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': 'short',
    'size': 17600,
}

class CattleMuzzle(DownloadURL, DatasetFactory):
    summary = summary
    url = 'https://cloud.une.edu.au/index.php/s/eMwaHAPK08dCDru/download'
    archive = 'Cattle Identification (supplementary material).zip'

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
