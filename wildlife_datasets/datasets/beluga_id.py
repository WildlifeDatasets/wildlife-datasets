import os
import pandas as pd
from .datasets_wildme import DatasetFactoryWildMe
from .downloads import DownloadURL

summary = {
    'licenses': 'Community Data License Agreement – Permissive',
    'licenses_url': 'https://cdla.dev/permissive-1-0/',
    'url': 'https://lila.science/datasets/beluga-id-2022/',
    'publication_url': None,
    'cite': 'belugaid',
    'animals': {'beluga whale'},
    'animals_simple': 'whales',
    'real_animals': True,
    'year': 2022,
    'reported_n_total': 5902,
    'reported_n_individuals': 788,
    'wild': True,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': False,
    'cropped': True,
    'span': '2.1 years',
    'size': 590,
}

class BelugaID(DownloadURL, DatasetFactoryWildMe):
    outdated_dataset = True
    summary = summary
    downloads = [
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga.coco.tar.gz', 'beluga.coco.tar.gz'),
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga-id-test.zip', 'beluga-id-test.zip'),
    ]
    
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
