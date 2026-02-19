import os

import numpy as np
import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': None,
    'licenses_url': None,
    'url': 'https://github.com/tvanzyl/wildlife_reidentification',
    'publication_url': 'https://ieeexplore.ieee.org/abstract/document/9311574',
    'cite': 'dlamini2020automated',
    'animals': {'nyala'},
    'animals_simple': 'nyalas',
    'real_animals': True,
    'year': 2020,
    'reported_n_total': 1934,
    'reported_n_individuals': 274,
    'wild': True,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': 'unknown',
    'size': 495,
}

class NyalaData(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
    archive = 'main.zip'
    rmtree = 'wildlife_reidentification-main/Lion_Data_Zero'

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure and about orientation
        identity = folders[3].astype(int)
        orientation = np.full(len(data), np.nan, dtype=object)
        orientation[data['file'].str.contains('left')] = 'left'
        orientation[data['file'].str.contains('right')] = 'right'

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'orientation': orientation,
            'original_split': folders[2]
        })
        return self.finalize_catalogue(df)   
