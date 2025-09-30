import os
import pandas as pd
from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://zenodo.org/records/15018518',
    'publication_url': 'https://www.mdpi.com/1424-8220/25/10/2971',
    'cite': 'perneel2025dynamic',
    'animals': {'cow'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2025,
    'reported_n_total': 11438,
    'reported_n_individuals': 48,
    'wild': False,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': '2 years',
    'size': 732,
}

class CoBRAReIdentificationYoungstock(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://zenodo.org/records/15018518/files/re_identification_youngstock.zip?download=1'
    archive = 're_identification_youngstock.zip'
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        # Extract information
        identity = folders[n_folders]
        orientation = folders[n_folders-1]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(identity + orientation + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'orientation': orientation,
        })
        return self.finalize_catalogue(df)
