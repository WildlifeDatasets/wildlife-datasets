import os
import datetime
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class MacaqueFaces(DatasetFactory):
    summary = summary['MacaqueFaces']
    
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
