import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class OpenCows2020(DatasetFactory):
    summary = summary['OpenCows2020']
    url = 'https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip'
    archive = '10m32xl88x2b61zlkkgz3fml17.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Select only re-identification dataset
        reid = folders[1] == 'identification'
        folders, data = folders[reid], data[reid]

        # Extract information from the folder structure
        split = folders[3]
        identity = folders[4]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'original_split': split
        })
        return self.finalize_catalogue(df)    
