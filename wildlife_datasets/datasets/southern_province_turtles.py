import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class GreenSeaTurtles(DatasetFactory):
    archive = 'greenseaturtles.zip'
    summary = summary['GreenSeaTurtles']

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/greenseaturtles --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#greenseaturtles'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        file_name = os.path.join(self.root, 'annotations.csv')
        data = pd.read_csv(file_name)

        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': 'data' + os.path.sep + data['image_name'],
            'identity': data['identity'],
            'orientation': data['orientation'],
            'daytime': data['daytime']
        })
        if 'bbox_x' in data.columns:
            df['bbox'] = data[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values.tolist()
        return self.finalize_catalogue(df)
