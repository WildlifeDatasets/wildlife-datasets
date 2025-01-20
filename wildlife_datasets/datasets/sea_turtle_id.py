import os
import json
import numpy as np
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class SeaTurtleID2022(DatasetFactory):
    summary = summary['SeaTurtleID2022']
    archive = 'seaturtleid2022.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/seaturtleid2022 --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#seaturtleid2022'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = os.path.join('turtles-data', 'data', 'annotations.json')
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)
        path_csv = os.path.join('turtles-data', 'data', 'metadata_splits.csv')
        with open(os.path.join(self.root, path_csv)) as file:
            df_images = pd.read_csv(file)

        # Extract data from the JSON file
        create_dict = lambda i: {
            'id': i['id'],
            'bbox': i['bbox'],
            'image_id': i['image_id'],
            'segmentation': i['segmentation'],
            'orientation': i['attributes']['orientation'] if 'orientation' in i['attributes'] else np.nan
        }
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations'] if i['category_id'] == 3])
        idx_bbox = ~df_annotation['bbox'].isnull()
        df_annotation.loc[idx_bbox,'bbox'] = df_annotation.loc[idx_bbox,'bbox'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df_images.rename({'id': 'image_id'}, axis=1, inplace=True)

        # Merge the information from the JSON file
        df = pd.merge(df_images, df_annotation, on='image_id', how='outer')
        df['path'] = 'turtles-data' + os.path.sep + 'data' + os.path.sep + df['file_name'].str.replace('/', os.path.sep)
        df = df.drop(['id', 'file_name', 'timestamp', 'width', 'height', 'year', 'split_closed_random', 'split_open'], axis=1)
        df.rename({'split_closed': 'original_split'}, axis=1, inplace=True)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])

        return self.finalize_catalogue(df)

class SeaTurtleIDHeads(DatasetFactory):
    summary = summary['SeaTurtleIDHeads']
    archive = 'seaturtleidheads.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/seaturtleidheads --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#seaturtleid'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = 'annotations.json'
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Extract dtaa from the JSON file
        create_dict = lambda i: {'id': i['id'], 'image_id': i['image_id'], 'identity': i['identity'], 'orientation': i['position']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, on='image_id')
        df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
        df = df.drop(['image_id', 'file_name'], axis=1)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])

        # Finalize the dataframe
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)

class SeaTurtleID2022_AnimalCLEF2025(DatasetFactory):
    summary = summary['SeaTurtleID2022_AnimalCLEF2025']

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