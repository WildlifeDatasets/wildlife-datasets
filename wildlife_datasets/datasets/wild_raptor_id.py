import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class WildRaptorID(DatasetFactory):
    summary = summary['WildRaptorID']
    url = 'https://zenodo.org/records/16731160/files/wild_raptor_id.zip?download=1'
    archive = 'wild_raptor_id.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
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
