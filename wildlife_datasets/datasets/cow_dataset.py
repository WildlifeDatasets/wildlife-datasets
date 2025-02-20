import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class CowDataset(DatasetFactory):
    summary = summary['CowDataset']
    url = 'https://figshare.com/ndownloader/files/31210192'
    archive = 'cow-dataset.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        # Rename the folder with non-ASCII characters
        dirs = [x for x in os.listdir() if os.path.isdir(x)]
        if len(dirs) != 1:
            raise Exception('There should be only one directory after extracting the file.')
        os.rename(dirs[0], 'images')
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        path = data['path'] + os.path.sep + data['file']
        date = [utils.get_image_date(os.path.join(self.root, p)) for p in path]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': path,
            'identity': folders[1].str.strip('cow_').astype(int),
            'date': date,
        })
        return self.finalize_catalogue(df)
