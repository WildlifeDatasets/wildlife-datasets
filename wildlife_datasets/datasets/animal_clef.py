import os
import numpy as np
import pandas as pd
from . import utils
from .datasets import WildlifeDataset
from .summary import summary

class AnimalCLEF2025(WildlifeDataset):    
    # TODO: add summary
    # TODO: add download
    #summary = summary['AnimalCLEF2025']
    #archive = 'amvrakikosturtles.zip'

    @classmethod
    def _download(cls):
        raise NotImplementedError
        command = f"datasets download -d wildlifedatasets/amvrakikosturtles --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#amvrakikosturtles'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        raise NotImplementedError
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        return self.finalize_catalogue(metadata)

class AnimalCLEF2025_LynxID2025(WildlifeDataset):
    @classmethod
    def _download(cls):
        raise Exception('This dataset is currently available only as part of the AnimalCLEF2025 competition.')

    @classmethod
    def _extract(cls):
        pass
    
    def create_catalogue(self) -> pd.DataFrame:
        data1 = pd.read_csv(os.path.join(self.root, 'metadata_database.csv'))
        data2 = pd.read_csv(os.path.join(self.root, 'metadata_query.csv'))
        data = pd.concat((data1, data2))
        data['path'] = data['path'].str[65:]
        data['species'] = 'lynx'
        data['date'] = np.nan
        return self.finalize_catalogue(data)
    
class AnimalCLEF2025_SeaTurtleID2022(WildlifeDataset):
    @classmethod
    def _download(cls):
        raise Exception('This dataset is currently available only as part of the AnimalCLEF2025 competition.')

    @classmethod
    def _extract(cls):
        pass
    
    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'annotations.csv'))
        data['image_name'] = data['path'].apply(lambda x: x.split('/')[-1])
        bbox = pd.read_csv(os.path.join(self.root, 'bbox.csv'))
        data = pd.merge(data, bbox, on='image_name')        

        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': data['path'],
            'identity': data['identity'],
            'date': data['date'],
            'bbox': data[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values.tolist()
        })
        return self.finalize_catalogue(df)