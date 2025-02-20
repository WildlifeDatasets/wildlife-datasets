import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class AmvrakikosTurtles(DatasetFactory):    
    summary = summary['AmvrakikosTurtles']
    archive = 'amvrakikosturtles.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/amvrakikosturtles --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#amvrakikosturtles'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'annotations.csv'))

        # Get the bounding box
        columns_bbox = ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']
        bbox = data[columns_bbox].to_numpy()
        bbox = pd.Series(list(bbox))
        path = 'images' + os.path.sep + data['image_name']
        date = [utils.get_image_date(os.path.join(self.root, p)) for p in path]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': path,
            'identity': data['image_name'].apply(lambda x: x.split('_')[0]).astype(int),
            'date': date,
            'orientation': data['image_name'].apply(lambda x: x.split('_')[2]),
            'bbox': bbox,
        })
        df = df[df['orientation'] != 'top']
        df['image_id'] = range(len(df))
        return self.finalize_catalogue(df)
