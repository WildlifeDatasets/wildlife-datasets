import os
import shutil
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class CTai(DatasetFactory):
    summary = summary['CTai']
    url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
    archive = 'master.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('chimpanzee_faces-master/datasets_cropped_chimpanzee_faces/data_CZoo')

    def create_catalogue(self) -> pd.DataFrame:            
        # Load information about the dataset
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CTai',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_ctai.txt'), header=None, sep=' ')
        
        # Extract keypoints from the information
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })
        return self.finalize_catalogue(df)

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong identities
        replace_identity = [
            ('Adult', self.unknown_name),
            ('Akouba', 'Akrouba'),
            ('Freddy', 'Fredy'),
            ('Ibrahiim', 'Ibrahim'),
            ('Liliou', 'Lilou'),
            ('Wapii', 'Wapi'),
            ('Woodstiock', 'Woodstock')
        ]
        return self.fix_labels_replace_identity(df, replace_identity)