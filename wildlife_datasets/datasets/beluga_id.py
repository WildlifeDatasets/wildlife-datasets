import os
import pandas as pd
from . import utils
from .datasets_wildme import DatasetFactoryWildMe
from .summary import summary

class BelugaID(DatasetFactoryWildMe):
    outdated_dataset = True
    summary = summary['BelugaID']
    downloads = [
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga.coco.tar.gz', 'beluga.coco.tar.gz'),
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga-id-test.zip', 'beluga-id-test.zip'),
    ]

    @classmethod
    def _download(cls):
        for url, archive in cls.downloads:
            utils.download_url(url, archive)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            archive_name = archive.split('.')[0]
            utils.extract_archive(archive, archive_name, delete=True)
    
    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme(os.path.join('beluga', 'beluga'), 2022)


class BelugaIDv2(BelugaID):
    outdated_dataset = False
    
    def create_catalogue(self) -> pd.DataFrame:
        # Use the original data
        df_train = self.create_catalogue_wildme(os.path.join('beluga', 'beluga'), 2022)
        df_train['original_split'] = 'train'

        # Get the conversion of identities between original and new data
        data_train = pd.read_csv(os.path.join(self.root, 'beluga-id-test', 'private_train_metadata.csv'))
        id_conversion = {}
        for old_id, data_train_red in data_train.groupby('original_whale_id'):
            if data_train_red['whale_id'].nunique() != 1:
                raise Exception('Conversion of old to new whale_id is not unique.')
            id_conversion[old_id] = data_train_red['whale_id'].iloc[0]

        # Add the new data
        data_test = pd.read_csv(os.path.join(self.root, 'beluga-id-test', 'private_test_metadata.csv'))
        df_test = pd.DataFrame({
            'path': data_test['path'].apply(lambda x: os.path.join('beluga-id-test', 'code-execution', 'images', os.path.split(x)[-1])),
            'image_id': range(len(data_train), len(data_train)+len(data_test)),
            'identity': data_test['original_whale_id'].apply(lambda x: id_conversion.get(x, 'unknown')),
            'date': data_test['timestamp'],
            'species': 'beluga_whale',
            'orientation': data_test['viewpoint'].replace({'top': 'up'}),
            'original_split': 'test'
        })
        
        # Finalize the dataframe
        df = pd.concat((df_train, df_test)).reset_index(drop=True)
        return self.finalize_catalogue(df)
