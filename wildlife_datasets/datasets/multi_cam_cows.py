import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class MultiCamCows2024(DatasetFactory):
    summary = summary['MultiCamCows2024']
    url = 'https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip'
    archive = '2inu67jru7a6821kkgehxg3cv2.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
    
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
