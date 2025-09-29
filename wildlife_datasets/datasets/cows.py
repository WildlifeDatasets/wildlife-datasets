import os
import datetime
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadURL

summary = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7',
    'publication_url': 'https://arxiv.org/abs/2105.01938',
    'cite': 'gao2021towards',
    'animals': {'cow'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2021,
    'reported_n_total': 13784,
    'reported_n_individuals': 181,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': '1 month',
    'size': 18881,
}

class Cows2021(DownloadURL, DatasetFactory):
    outdated_dataset = True
    summary = summary
    url = 'https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip'
    archive = '4vnrca7qw1642qlwxjadp87h7.zip'

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure
        ii = (folders[2] == 'Identification') & (folders[3] == 'Test')
        folders = folders.loc[ii]
        data = data.loc[ii]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[4].astype(int),
        })
        df['date'] = df['path'].apply(lambda x: self.extract_date(x))
        return self.finalize_catalogue(df)

    def extract_date(self, x):
        x = os.path.split(x)[1]
        if x.startswith('image_'):
            x = x[6:]
        if x[7] == '_':
            x = x[8:]
        i1 = x.find('_')
        i2 = x[i1+1:].find('_')
        x = x[:i1+i2+1]
        return datetime.datetime.strptime(x, '%Y-%m-%d_%H-%M-%S').strftime('%Y-%m-%d %H:%M:%S')

class Cows2021v2(Cows2021):
    outdated_dataset = False

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong identities and images
        replace_identity1 = [
            (164, 148),
            (105, 29)
        ]
        replace_identity2 = [
            ('image_0001226_2020-02-11_12-43-7_roi_001.jpg', 137, 107)
        ]
        df = self.fix_labels_replace_identity(df, replace_identity1)
        return self.fix_labels_replace_images(df, replace_identity2)
