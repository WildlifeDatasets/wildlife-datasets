import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import hashlib
import json

from .. import downloads
from .metadata import metadata


'''
General:

TODO: 
I would represent keypoints as they are. There is no unified notation
what they can represent (unlike segmentations and bbox) and lot of
imporant information can get lost.

TODO:
I think standard attributes should always be in separate columns
 or always in 'attributes' column as dict.

TODO:
We should at least provide description on how we did the data
processing and the reasoning behind it. Some datasets are especially
dificult and not obvious

'''

def find_images(
    root: str,
    img_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    ) -> pd.DataFrame:
    '''
    Find all image files in folder recursively based on img_extensions. 
    Save filename and relative path from root.
    '''
    data = [] 
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({'path': os.path.relpath(path, start=root), 'file': file})
    return pd.DataFrame(data)

def create_id(string_col: pd.Series) -> pd.Series:
    '''
    Creates unique id from string based on MD5 hash.
    '''
    entity_id = string_col.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id


class DatasetFactory():
    def __init__(
        self, 
        root: str = 'data',
        df: Optional(pd.DataFrame) = None,
        download: bool = False,
        **kwargs
        ):

        self.root = root
        if download and hasattr(self, 'download'): 
            self.download.get_data(os.path.join(root, self.__class__.__name__))
        if df is None:
            self.df = self.create_catalogue(**kwargs)
        else:
            self.df = df

    @classmethod
    def from_file(cls, root: str, file_path: str, **kwargs):
        # TODO: reconsider pickle due to consistency - maybe json?

        df = pd.read_pickle(file_path)
        instance = cls(root, df, **kwargs)
        return instance

    def to_file(self, file_path: str, overwrite: bool = False):
        # TODO: reconsider pickle due to consistency - maybe json?

        if overwrite or not os.path.exists(file_path):
            # TODO: overwrite should be always True and removed as argument.
            self.df.to_pickle(file_path)

    def create_catalogue(self):
        '''
        Create catalogue data frame.
        This method is dataset specific and each dataset needs to override it.
        '''
        raise NotImplementedError()

    def finalize_catalogue(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Finalize catalogue data frame and runs checks.
        '''
        df = self.reorder_df(df)
        df = self.remove_constant_columns(df)
        self.check_unique_id(df)
        self.check_files_exist(df['path'])
        if segmentation in df.columns:
            self.check_files_exist(df['segmentation'])
        return df

    def reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: fixed column names is never good idea. 
        # TODO: 3 columns are always there - order them first
        # TODO: rest alphabetical: df.sort_index(axis=1) + df.loc[:, cols]

        default_order = ['id', 'path', 'identity', 'bbox', 'segmentation', 'mask', 'position', 'species', 'keypoints', 'date', 'video', 'attributes']
        df_names = list(df.columns)
        col_names = []
        for name in default_order:
            if name in df_names:
                col_names.append(name)
        for name in df_names:
            if name not in default_order:
                col_names.append(name)
        
        df = df.sort_values('id').reset_index(drop=True)
        return df.reindex(columns=col_names)

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Removes columns with single unique value.
        '''
        # TODO: try this
        #df = df.drop(columns=df.columns[df.astype('str').nunique()==1])

        for df_name in list(df.columns):
            if df[df_name].astype('str').nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df

    def check_unique_id(self, df: pd.DataFrame) -> None:
        '''
        Check if values in ID column are unique.
        '''
        if len(df['id'].unique()) != len(df):
            raise(Exception('Image ID not unique.'))

    def check_files_exist(self, col: pd.Series) -> None:
        '''
        Check if paths in given column exist.
        '''
        for path in col:
            if type(path) == str and not os.path.exists(os.path.join(self.root, path)):
                raise(Exception('Path does not exist:' + os.path.join(self.root, path)))


class Test(DatasetFactory):
    download = downloads.test
    metadata = metadata['Test']

    def create_catalogue(self) -> pd.DataFrame:
        return pd.DataFrame([1, 2])


class AerialCattle2017(DatasetFactory):
    download = downloads.aerial_cattle_2017
    metadata = metadata['AerialCattle2017']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'video': folders[2],
        })
        return self.finalize_catalogue(df)


class FriesianCattle2015(DatasetFactory):
    download = downloads.friesian_cattle_2015
    metadata = metadata['FriesianCattle2015']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        split = folders[1].replace({'Cows-testing': 'test', 'Cows-training': 'train'})
        assert len(split.unique()) == 2

        identity = folders[2].str.strip('Cow').astype(int)

        df = pd.DataFrame({
            'id': create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)



class SMALST(DatasetFactory):
    # TODO: move to metadata
    licenses = 'MIT License'
    licenses_url = 'https://github.com/silviazuffi/smalst/blob/master/LICENSE.txt'
    url = 'https://github.com/silviazuffi/smalst'
    cite = 'zuffi2019three'
    animals = ('zebra')
    real_animals = False
    year = 2019
    reported_n_total = 12850
    reported_n_identified = 12850
    reported_n_photos = 12850
    reported_n_individuals = 10
    wild = False
    clear_photos = True
    pose = 'multiple'
    unique_pattern = True 
    from_video = False
    full_frame = True
    span = 'artificial'

    def create_catalogue(self) -> pd.DataFrame:
        # Images
        data = find_images(os.path.join(self.root, 'zebra_training_set', 'images'))
        path = data['file'].str.strip('zebra_')
        data['identity'] = path.str[0]
        data['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        data['path'] = 'zebra_training_set' + os.path.sep + 'images' + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        # Masks
        masks = find_images(os.path.join(self.root, 'zebra_training_set', 'bgsub'))
        path = masks['file'].str.strip('zebra_')
        masks['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        masks['mask'] = 'zebra_training_set' + os.path.sep + 'bgsub' + os.path.sep + masks['file']
        masks = masks.drop(['path', 'file'], axis=1)

        df = pd.merge(data, masks, on='id')
        return self.finalize_catalogue(df)


