import os
import shutil
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class SMALST(DatasetFactory):
    summary = summary['SMALST']
    url = 'https://drive.google.com/uc?id=1yVy4--M4CNfE5x9wUr1QBmAXEcWb6PWF'
    archive = 'zebra_training_set.zip'

    @classmethod
    def _download(cls):
        exception_text = '''Download failed. GDown quota probably reached. Download dataset manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#smalst'''
        utils.gdown_download(cls.url, cls.archive, exception_text)

    @classmethod
    def _extract(cls):
        exception_text = '''Extracting works only on Linux. Please extract it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#smalst'''
        if os.name == 'posix':
            os.system('jar xvf ' + cls.archive)
            os.remove(cls.archive)
            shutil.rmtree(os.path.join('zebra_training_set', 'annotations'))
            shutil.rmtree(os.path.join('zebra_training_set', 'texmap'))
            shutil.rmtree(os.path.join('zebra_training_set', 'uvflow'))

        else:
            raise Exception(exception_text)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(os.path.join(self.root, 'zebra_training_set', 'images'))
        
        # Extract information about the images
        path = data['file'].str.strip('zebra_')
        data['identity'] = path.str[0]
        data['image_id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        data['path'] = 'zebra_training_set' + os.path.sep + 'images' + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        # Find all masks in root
        masks = utils.find_images(os.path.join(self.root, 'zebra_training_set', 'bgsub'))
        
        # Extract information about the images
        path = masks['file'].str.strip('zebra_')
        masks['image_id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        masks['segmentation'] = 'zebra_training_set' + os.path.sep + 'bgsub' + os.path.sep + masks['file']
        masks = masks.drop(['path', 'file'], axis=1)

        # Finalize the dataframe
        df = pd.merge(data, masks, on='image_id', how='left')
        return self.finalize_catalogue(df)
