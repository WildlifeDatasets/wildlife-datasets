import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class ReunionTurtles(DatasetFactory):    
    summary = summary['ReunionTurtles']
    archive = 'reunionturtles.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/reunionturtles --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#reunionturtles'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'data.csv'))

        date = pd.to_datetime(data['Date'])
        year = date.apply(lambda x: x.year)
        path = data['Species'] + os.path.sep + data['Turtle_ID'] + os.path.sep + year.astype(str) + os.path.sep + data['Photo_name']
        orientation = data['Photo_name'].apply(lambda x: os.path.splitext(x)[0].split('_')[2])
        orientation = orientation.replace({'L': 'left', 'R': 'right'})

        # Extract and convert ID codes
        id_code = list(data['ID_Code'].apply(lambda x: x.split(';')))
        max0 = 0
        max1 = 0
        for x in id_code:
            for y in x:
                max0 = max(max0, int(y[0]))
                max1 = max(max1, int(y[1]))
        code = np.zeros((len(id_code), max0, max1), dtype=int)
        for i, x in enumerate(id_code):
            for y in x:
                code[i, int(y[0])-1, int(y[1])-1] = int(y[2])
        code = code.reshape(len(id_code), -1)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': path,
            'identity': data['Turtle_ID'],
            'date': date,
            'orientation': orientation,
            'species': data['Species'],
            'id_code': list(code)
        })
        return self.finalize_catalogue(df)
