import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadURL

summary = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/2inu67jru7a6821kkgehxg3cv2',
    'publication_url': 'https://arxiv.org/abs/2410.12695',
    'cite': 'yu2024multicamcows2024',
    'animals': {'Friesian cattle'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2024,
    'reported_n_total': 101329,
    'reported_n_individuals': 90,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': '7 days',
    'size': 36500,
}

class MultiCamCows2024(DownloadURL, DatasetFactory):
    summary = summary
    url = 'https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip'
    archive = '2inu67jru7a6821kkgehxg3cv2.zip'
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        n_cols = len(folders.columns)
        df = pd.DataFrame({
            'image_id': folders[n_cols-1] + '_' + folders[n_cols-2] + '_' + data['file'],
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[n_cols-1].astype(int),
            'date': folders[n_cols-2].apply(self.extract_date),
        })
        return self.finalize_catalogue(df)
    
    def extract_date(self, x):
        conversion = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }        
        year = int(x[:4])
        month = conversion[x[4:7].lower()]
        day = int(x[7:])
        return f'{year}-{month}-{day}'
