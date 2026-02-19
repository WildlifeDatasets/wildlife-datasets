import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://zenodo.org/records/16731160',
    'publication_url': 'https://www.sciencedirect.com/science/article/pii/S1574954125003887',
    'cite': 'guo2025individual',
    'animals': {'eagle'},
    'animals_simple': 'eagles',
    'real_animals': True,
    'year': 2025,
    'reported_n_total': 14817,
    'reported_n_individuals': 47,
    'wild': True,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': False,
    'from_video': True,
    'cropped': True,
    'span': '1 month',
    'size': 2600,
}

class WildRaptorID(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://zenodo.org/records/16731160/files/wild_raptor_id.zip?download=1'
    archive = 'wild_raptor_id.zip'

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        data['file_size'] = [os.path.getsize(os.path.join(self.root, x['path'], x['file'])) for _, x in data.iterrows()]
        data['identity'] = folders[n_folders].apply(lambda x: x.split('_')[0])

        # Very rough way of checking that the files with the same cols are duplicates
        cols = ['file', 'identity']
        for _, df_red in data.groupby(cols):
            if len(df_red) > 1:
                if df_red['file_size'].nunique() != 1:
                    raise Exception('The expected duplicates are not duplicates')
                
        # Remove the duplicates
        data = data.drop_duplicates(subset=cols)

        # Add date
        file = data['file'].apply(lambda x: os.path.splitext(x)[0][1:-1])
        date = file.apply(lambda x: x.split('_')[0])
        date = date.apply(lambda x: f'{x[:4]}-{x[4:6]}-{x[6:8]} {x[8:10]}:{x[10:12]}')
        data['date'] = date

        # Add video        
        for i, (_, df_group) in enumerate(data.groupby(['identity', 'date'])):
            data.loc[df_group.index, 'video'] = i
        data['video'] = data['video'].astype(int)
        
        # Finalize the dataframe
        data['image_id'] = data['identity'] + '_' + file
        data['path'] = data['path'] + os.path.sep + data['file']
        data = data.drop(['file', 'file_size'], axis=1)
        return self.finalize_catalogue(data)
