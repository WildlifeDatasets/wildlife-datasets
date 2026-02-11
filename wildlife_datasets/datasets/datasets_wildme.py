import os
import json
import numpy as np
import pandas as pd
from . import utils
from .datasets import WildlifeDataset

class WildlifeDatasetWildMe(WildlifeDataset):
    def create_catalogue_wildme(self, prefix: str, year: int) -> pd.DataFrame:
        assert self.root is not None
        # Get paths for annotation JSON file and for folder with images
        path_json = os.path.join(prefix + '.coco', 'annotations', 'instances_train' + str(year) + '.json')
        path_images = os.path.join(prefix + '.coco', 'images', 'train' + str(year))

        # Load annotations JSON file
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Check whether segmentation is different from a box
        for ann in data['annotations']:
            if len(ann['segmentation']) != 1:
                raise(Exception('Wrong number of segmentations'))
        
        # Extract the data from the JSON file
        create_dict = lambda i: {'identity': i['name'], 'bbox': utils.segmentation_bbox(i['segmentation'][0]), 'image_id': i['image_id'], 'category_id': i['category_id'], 'segmentation': i['segmentation'][0], 'orientation': i['viewpoint']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['file_name'], 'image_id': i['id'], 'date': i['date_captured']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])
        species = pd.DataFrame(data['categories'])
        species = species.rename(columns={'id': 'category_id', 'name': 'species'})

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, species, how='left', on='category_id')
        df = pd.merge(df, df_images, how='left', on='image_id')

        # Modify some columns
        df['path'] = path_images + os.path.sep + df['file_name']
        df['id'] = range(len(df))
        df.loc[df['identity'] == '____', 'identity'] = self.unknown_name

        # Remove segmentations which are the same as bounding boxes
        ii = []
        for i in range(len(df)):
            ii.append(utils.is_annotation_bbox(df.iloc[i]['segmentation'], df.iloc[i]['bbox'], tol=3))
        df.loc[ii, 'segmentation'] = np.nan

        # Rename empty dates
        df.loc[df['date'] == 'NA', 'date'] = np.nan

        # Remove superficial columns
        df = df.drop(['image_id', 'file_name', 'supercategory', 'category_id'], axis=1)
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
