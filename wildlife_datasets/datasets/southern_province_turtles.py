import os
import pandas as pd
from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://www.kaggle.com/datasets/wildlifedatasets/southernprovinceturtles',
    'url': 'https://www.kaggle.com/datasets/wildlifedatasets/southernprovinceturtles',
    'publication_url': 'https://www.biorxiv.org/content/10.1101/2024.09.13.612839',
    'cite': 'adam2024exploiting',
    'animals': {'green turtle'},
    'animals_simple': 'sea turtles',
    'real_animals': True,
    'year': 2024,
    'reported_n_total': 481,
    'reported_n_individuals': 51,
    'wild': True,
    'clear_photos': False,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 55,
}

class SouthernProvinceTurtles(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = 'wildlifedatasets/southernprovinceturtles'
    kaggle_type = 'datasets'

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
