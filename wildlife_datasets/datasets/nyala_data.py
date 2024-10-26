import os
import shutil
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class NyalaData(DatasetFactory):
    summary = summary['NyalaData']
    url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
    archive = 'main.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('wildlife_reidentification-main/Lion_Data_Zero')

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure and about orientation
        identity = folders[3].astype(int)
        orientation = np.full(len(data), np.nan, dtype=object)
        orientation[data['file'].str.contains('left')] = 'left'
        orientation[data['file'].str.contains('right')] = 'right'

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'orientation': orientation,
            'original_split': folders[2]
        })
        return self.finalize_catalogue(df)   
