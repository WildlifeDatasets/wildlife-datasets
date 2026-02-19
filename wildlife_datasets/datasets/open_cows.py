import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17',
    'publication_url': 'https://www.sciencedirect.com/science/article/pii/S0168169921001514',
    'cite': 'andrew2021visual',
    'animals': {'cow'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2020,
    'reported_n_total': 4736,
    'reported_n_individuals': 46,
    'wild': False,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': 'short',
    'size': 2272,
}

class OpenCows2020(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip'
    archive = '10m32xl88x2b61zlkkgz3fml17.zip'

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Select only re-identification dataset
        reid = folders[1] == 'identification'
        folders, data = folders[reid], data[reid]

        # Extract information from the folder structure
        split = folders[3]
        identity = folders[4]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'original_split': split
        })
        return self.finalize_catalogue(df)    
