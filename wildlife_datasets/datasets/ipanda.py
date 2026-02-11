import os
import json
import string
import numpy as np
import pandas as pd
from . import utils
from .datasets import WildlifeDataset

summary = {
    'licenses': None,
    'licenses_url': None,
    'url': 'https://github.com/iPandaDateset/iPanda-50',
    'publication_url': 'https://ieeexplore.ieee.org/abstract/document/9347819',
    'cite': 'wang2021giant',
    'animals': {'great panda'},
    'animals_simple': 'pandas',
    'real_animals': True,
    'year': 2021,
    'reported_n_total': 6874,
    'reported_n_individuals': 50,
    'wild': False,
    'clear_photos': True,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'unknown',
    'size': 929,
}

class IPanda50(WildlifeDataset):
    summary = summary
    downloads = [
        ('https://drive.google.com/uc?id=1nkh-g6a8JvWy-XsMaZqrN2AXoPlaXuFg', 'iPanda50-images.zip'),
        ('https://drive.google.com/uc?id=1gVREtFWkNec4xwqOyKkpuIQIyWU_Y_Ob', 'iPanda50-split.zip'),
        ('https://drive.google.com/uc?id=1jdACN98uOxedZDT-6X3rpbooLAAUEbNY', 'iPanda50-eyes-labels.zip'),
    ]

    @classmethod
    def _download(cls):
        exception_text = '''Download failed. GDown quota probably reached. Download dataset manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#ipanda50'''
        for url, archive in cls.downloads:
            utils.gdown_download(url, archive, exception_text=exception_text)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            utils.extract_archive(archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract keypoint information about eyes
        keypoints = []
        for path in data['path'] + os.path.sep + data['file']:
            path_split = os.path.normpath(path).split(os.path.sep)
            path_json = os.path.join('iPanda50-eyes-labels', path_split[1], os.path.splitext(path_split[2])[0] + '.json')
            keypoints_part = np.full(8, np.nan)        
            if os.path.exists(os.path.join(self.root, path_json)):
                with open(os.path.join(self.root, path_json)) as file:
                    keypoints_file = json.load(file)['shapes']
                    if keypoints_file[0]['label'] == 'right_eye':
                        keypoints_part[0:4] = np.reshape(keypoints_file[0]['points'], 4)
                    if keypoints_file[0]['label'] == 'left_eye':
                        keypoints_part[4:8] = np.reshape(keypoints_file[0]['points'], 4)
                    if len(keypoints_file) == 2 and keypoints_file[1]['label'] == 'right_eye':
                        keypoints_part[0:4] = np.reshape(keypoints_file[1]['points'], 4)
                    if len(keypoints_file) == 2 and keypoints_file[1]['label'] == 'left_eye':
                        keypoints_part[4:8] = np.reshape(keypoints_file[1]['points'], 4)
            keypoints.append(list(keypoints_part))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'keypoints': keypoints
            })

        # Remove non-ASCII characters while keeping backwards compatibility
        file_name = os.path.join(self.root, 'changes.csv')
        if os.path.exists(file_name):
            # Files were already renamed, change image_id to keep backward compability
            df_changes = pd.read_csv(file_name)
            id_changes = {}
            for _, df_row in df_changes.iterrows():
                id_changes[df_row['id_new']] = df_row['id_old']
            df.replace(id_changes, inplace=True)
        else:
            # Rename files, keep original image_id, create list of changes
            ids_old = []
            ids_new = []
            for _, df_row in df.iterrows():
                path_new = ''.join(c for c in df_row['path'] if c in string.printable)
                # Check if there are non-ASCII characters
                if path_new != df_row['path']:
                    # Rename files and df
                    os.rename(os.path.join(self.root, df_row['path']), os.path.join(self.root, path_new))
                    df_row['path'] = path_new
                    # Save changes in image_id
                    ids_old.append(df_row['image_id'])
                    ids_new.append(utils.create_id(pd.Series(os.path.split(df_row['path'])[-1])).iloc[0])
            if len(df) != df['path'].nunique():
                raise(Exception("Non-unique names. Something went wrong when renaming."))
            pd.DataFrame({'id_old': ids_old, 'id_new': ids_new}).to_csv(file_name)

        # Finalize the dataframe        
        return self.finalize_catalogue(df)
