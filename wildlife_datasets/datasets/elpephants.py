import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import WildlifeDataset

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://inf-cv.uni-jena.de/home/research/datasets/elpephants/',
    'url': 'https://inf-cv.uni-jena.de/home/research/datasets/elpephants/',
    'publication_url': 'https://openaccess.thecvf.com/content_ICCVW_2019/html/CVWC/Korschens_ELPephants_A_Fine-Grained_Dataset_for_Elephant_Re-Identification_ICCVW_2019_paper.html',
    'cite': 'korschens2019elpephants',
    'animals': {'elephant'},
    'animals_simple': 'elephants',
    'real_animals': True,
    'year': 2019,
    'reported_n_total': 2078,
    'reported_n_individuals': 276,
    'wild': True,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': False,
    'from_video': False,
    'cropped': False,
    'span': '14 years',
    'size': 625,
}

class ELPephants(WildlifeDataset):
    summary = summary
    archive = 'ELPephant (elephant ID system).zip'

    @classmethod
    def _download(cls):
        exception_text = '''Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#elpephants'''
        raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)

        # Create the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': data['file'].apply(self.extract_identity),
            'date': data['file'].apply(self.extract_date),
            'orientation': data['file'].apply(self.extract_orientation),
        })

        # Add training and testing split
        path_txt = utils.find_images(self.root, img_extensions=('.txt',))
        idx_train = np.where(path_txt['file'] == 'train.txt')[0]
        idx_test = np.where(path_txt['file'] == 'val.txt')[0]
        if len(idx_train) == 1 and len(idx_test) == 1:            
            data_train = pd.read_csv(os.path.join(self.root, path_txt['path'].iloc[idx_train[0]], 'train.txt'), header=None, sep='\t')
            data_train = data_train[1].to_numpy()
            data_test = pd.read_csv(os.path.join(self.root, path_txt['path'].iloc[idx_test[0]], 'val.txt'), header=None, sep='\t')
            data_test = data_test[1].to_numpy()
            df['original_split'] = data['file'].apply(lambda x: utils.get_split(x, data_train, data_test))

        # Finalize the dataframe
        return self.finalize_catalogue(df)

    def extract_identity(self, x):
        return int(x.strip().split('_')[0])

    def extract_date(self, x):
        for y in x.lower().split('_')[::-1]:
            date = self.extract_date_part(y)
            if isinstance(date, str):
                return date
        return np.nan
        
    def extract_date_part(self, x):
        i_end = -np.inf
        conversion = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
        # Extract month
        for month_try in conversion:
            if month_try in x:
                i = x.index(month_try)
                if i > i_end:
                    month = month_try
                    i_end = i
        # Month not found
        if i_end == -np.inf:
            return np.nan
        # Extract string after month and remove extension
        year = x[i_end+len(month):]
        year = os.path.splitext(year)[0]
        year = year.strip()
        # Extract year from the beginning or end
        if len(year) >= 4:
            if represents_int(year[:4]):
                return f'{int(year[:4])}-{conversion[month]}-1'
            elif represents_int(year[-4:]):
                return f'{int(year[-4:])}-{conversion[month]}-1'
        return np.nan
        
    def extract_orientation(self, x):
        x = x.lower()
        for orientation in ['left', 'right', 'front']:
            if orientation in x:
                return orientation
        return np.nan         

def represents_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True
