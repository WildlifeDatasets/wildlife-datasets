import os
import json
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


class AnimalCLEF2025_SalamanderID2025(WildlifeDataset):
    @classmethod
    def _download(cls):
        raise Exception('This dataset is currently available only as part of the AnimalCLEF2025 competition.')

    @classmethod
    def _extract(cls):
        pass
    
    def create_catalogue(self) -> pd.DataFrame:
        path_json = os.path.join('annotations', 'instances_default.json')

        # Load annotations JSON file
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Check whether segmentation is different from a box
        for ann in data['annotations']:
            if len(ann['bbox']) != 4:
                raise(Exception('Bounding box missing'))
            if ann['attributes']['rotation'] not in [0, 359.99999999929037]:
                raise(Exception('Rotation is not 0'))

        # Extract the data from the JSON file
        create_dict = lambda i: {'bbox': i['bbox'], 'image_id': i['image_id'], 'orientation': i['attributes']['orientation']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'path': i['file_name'], 'image_id': i['id']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, how='left', on='image_id')

        # Include identities
        df['filename'] = df['path'].apply(lambda x: x.split('/')[-1])
        identity = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        identity = identity[['ts', 'filename', 'identity']]
        df = pd.merge(df, identity, on='filename')
        df = df.drop('filename', axis=1)

        # There are 48 images with two bounding boxes which are actually the same.
        idx = []
        for _, df_file in df.groupby('path'):
            idx.append(df_file.index[0])
        df = df.loc[idx]
        
        # Finalize the dataframe
        df['path'] = df['path'].apply(lambda x: 'images/'+  x.split('/')[-1])
        df = df.rename({'ts': 'date'}, axis=1)
        df['date'] = df['date'].str[:10]
        return self.finalize_catalogue(df)


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