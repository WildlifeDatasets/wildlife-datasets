import os
import shutil
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class LionData(DatasetFactory):
    summary = summary['LionData']
    url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
    archive = 'main.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('wildlife_reidentification-main/Nyala_Data_Zero')

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[3],
        })
        return self.finalize_catalogue(df)
