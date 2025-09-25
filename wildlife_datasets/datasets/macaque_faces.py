import os
import datetime
import pandas as pd
from . import utils
from .datasets import DatasetFactory

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://github.com/clwitham/MacaqueFaces/blob/master/license.md',
    'url': 'https://github.com/clwitham/MacaqueFaces',
    'publication_url': 'https://www.sciencedirect.com/science/article/pii/S0165027017302637',
    'cite': 'witham2018automated',
    'animals': {'rhesus macaque'},
    'animals_simple': 'macaques',
    'real_animals': True,
    'year': 2018,
    'reported_n_total': 6460,
    'reported_n_individuals': 34,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': True,
    'cropped': True,
    'span': '1.4 years',
    'size': 12,
}

class MacaqueFaces(DatasetFactory):
    summary = summary
    
    @classmethod
    def _download(cls):
        downloads = [
            ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces.zip', 'MacaqueFaces.zip'),
            ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces_ImageInfo.csv', 'MacaqueFaces_ImageInfo.csv'),
        ]
        for url, file in downloads:
            utils.download_url(url, file)

    @classmethod
    def _extract(cls):
        utils.extract_archive('MacaqueFaces.zip', delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        data = pd.read_csv(os.path.join(self.root, 'MacaqueFaces_ImageInfo.csv'))
        date_taken = [datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%Y-%m-%d') for date in data['DateTaken']]
        
        # Finalize the dataframe
        data['Path'] = data['Path'].str.replace('/', os.path.sep)
        df = pd.DataFrame({
            'image_id': pd.Series(range(len(data))),            
            'path': 'MacaqueFaces' + os.path.sep + data['Path'].str.strip(os.path.sep) + os.path.sep + data['FileName'],
            'identity': data['ID'],
            'date': pd.Series(date_taken),
            'category': data['Category']
        })
        return self.finalize_catalogue(df)
