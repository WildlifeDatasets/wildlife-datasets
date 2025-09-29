import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadURL

summary = {
    'licenses': None,
    'licenses_url': None,
    'url': 'https://github.com/tvanzyl/wildlife_reidentification',
    'publication_url': 'https://ieeexplore.ieee.org/abstract/document/9311574',
    'cite': 'dlamini2020automated',
    'animals': {'lion'},
    'animals_simple': 'lions',
    'real_animals': True,
    'year': 2020,
    'reported_n_total': 750,
    'reported_n_individuals': 98,
    'wild': True,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'unknown',
    'size': 495,
}

class LionData(DownloadURL, DatasetFactory):
    summary = summary
    url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
    archive = 'main.zip'
    rmtree = 'wildlife_reidentification-main/Nyala_Data_Zero'

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[3],
        })
        return self.finalize_catalogue(df)
