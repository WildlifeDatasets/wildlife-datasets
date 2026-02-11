import os
import pandas as pd
from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid',
    'publication_url': 'https://openaccess.thecvf.com/content_WACVW_2020/html/w2/Haurum_Re-Identification_of_Zebrafish_using_Metric_Learning_WACVW_2020_paper.html',
    'cite': 'bruslund2020re',
    'animals': {'zebrafish'},
    'animals_simple': 'fish',
    'real_animals': True,
    'year': 2020,
    'reported_n_total': 6672,
    'reported_n_individuals': 6,
    'wild': False,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': False,
    'from_video': True,
    'cropped': False,
    'span': '1 day',
    'size': 12093,
}

class AAUZebraFish(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = 'aalborguniversity/aau-zebrafish-reid'
    kaggle_type = 'datasets'

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = pd.read_csv(os.path.join(self.root, 'annotations.csv'), sep=';')

        # Modify the bounding boxes into the required format
        columns_bbox = [
            'Upper left corner X',
            'Upper left corner Y',
            'Lower right corner X',
            'Lower right corner Y',
        ]
        bbox = data[columns_bbox].to_numpy()
        bbox[:,2] = bbox[:,2] - bbox[:,0]
        bbox[:,3] = bbox[:,3] - bbox[:,1]
        bbox = pd.Series(list(bbox))

        # Split additional data into a separate structure
        attributes = data['Right,Turning,Occlusion,Glitch'].str.split(',', expand=True)
        attributes.drop([0], axis=1, inplace=True)
        attributes.columns = ['turning', 'occlusion', 'glitch']
        attributes = attributes.astype(int).astype(bool)

        # Split additional data into a separate structure
        orientation = data['Right,Turning,Occlusion,Glitch'].str.split(',', expand=True)[0]
        orientation.replace('1', 'right', inplace=True)
        orientation.replace('0', 'left', inplace=True)

        # Modify information about video sources
        video = data['Filename'].str.split('_',  expand=True)[0]
        video = video.astype('category').cat.codes

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['Object ID'].astype(str) + data['Filename']),
            'path': 'data' + os.path.sep + data['Filename'],
            'identity': data['Object ID'],
            'video': video,
            'bbox': bbox,
            'orientation': orientation,
        })
        df = df.join(attributes)
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
