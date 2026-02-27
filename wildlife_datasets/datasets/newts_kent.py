import os
import re


import numpy as np
import pandas as pd


from .datasets import utils, WildlifeDataset
from .downloads import DownloadPrivate


def restrict(data: pd.DataFrame, folders: pd.DataFrame, idx: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, folders = data[idx], folders[idx]
    while True:
        max_col = np.max(folders.columns)
        if all(folders[max_col].isnull()):
            folders = folders.drop(max_col, axis=1)
        else:
            break
    return data, folders


def get_name(x):
    x = x.upper().replace('-', '_')
    x_splits = x.split('_')
    if len(x_splits) >= 2 and ('IMG_' not in x or len(x_splits) >= 3):
        y = x_splits[-1]
        for a in ['.', ' ', '(', ')']:
            y = y.split(a)[0]
        return y.strip()
    return None


summary = {
    "licenses": "",
    "licenses_url": "",
    "url": "",
    "publication_url": "",
    "cite": "",
    "animals": {"newts"},
    "animals_simple": "newts",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "5 years",
    "size": None,
}


class NewtsKent(DownloadPrivate, WildlifeDataset):
    summary = summary

    def create_catalogue(self, load_segmentation: bool = False) -> pd.DataFrame:
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        if folders[1].nunique() != 1 and folders[1].iloc[0] != 'Identification':
            raise ValueError('Structure wrong')

        data['identity'] = data['file'].apply(get_name)
        data['path'] = data['path'] + os.path.sep + data['file']
        data['year'] = folders[0].apply(lambda x: int(x[:4]))        
        
        # Remove duplicated images
        mask = ~data['path'].apply(lambda x: 'Duplicated' in x)
        data, folders = restrict(data, folders, mask)

        # Remove images without an identity
        mask = ~data['identity'].isnull()
        data, folders = restrict(data, folders, mask)

        # Keep only JUV*, M*, F* and *, where * are 4 or 5 digits.
        pattern1 = r'^[MF]?\d{4,5}$'
        mask1 = data['identity'].apply(lambda x: bool(re.match(pattern1, x)))
        pattern2 = r'^JUV\d{4,5}$'
        mask2 = data['identity'].apply(lambda x: bool(re.match(pattern2, x)))
        data, folders = restrict(data, folders, mask1 | mask2)

        data['image_id'] = utils.get_persistent_id(data['path'])
        data = data.drop('file', axis=1)

        if load_segmentation:
            file_name = os.path.join(self.root, "segmentation.csv")
            data = utils.load_segmentation(data, file_name)
        
        return self.finalize_catalogue(data)

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        replace = [
            ('F1602', 'F1503'),
            ('F2121', 'F1837'), 
            ('F2417', 'F2402'),
            ('F2428', 'F2119'), 
            ('M2206', 'M2104'),
            ('M2207', 'M2104'),
            ('M2336', 'M2108'),
            ('M2422', 'M2219'), 
        ]
        return self.fix_labels_replace_identity(df, replace)        