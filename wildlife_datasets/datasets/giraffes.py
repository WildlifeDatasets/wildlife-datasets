import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class Giraffes(DatasetFactory):
    summary = summary['Giraffes']

    @classmethod
    def _download(cls):
        url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/'
        command = f"wget -rpk -l 10 -np -c --random-wait -U Mozilla {url} -P '.' "
        exception_text = '''Download works only on Linux. Please download it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#giraffes'''
        if os.name == 'posix':
            os.system(command)
        else:
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        pass

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        # Extract information from the folder structure
        clusters = folders[n_folders-1] == 'clusters'
        data, folders = data[clusters], folders[clusters]

        # Finalize the dataframe
        df = pd.DataFrame({    
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[n_folders],
        })
        return self.finalize_catalogue(df)
