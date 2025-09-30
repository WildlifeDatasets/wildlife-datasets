import os
import pandas as pd
from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://figshare.com/articles/dataset/data_set_zip/16879780',
    'publication_url': None,
    'cite': 'cowdataset',
    'animals': {'cow'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2021,
    'reported_n_total': 1485,
    'reported_n_individuals': 13,
    'wild': False,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': 'short',
    'size': 4150,
}

class CowDataset(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://figshare.com/ndownloader/files/31210192'
    archive = 'cow-dataset.zip'

    @classmethod
    def _extract(cls):
        super()._extract()
        # Rename the folder with non-ASCII characters
        dirs = [x for x in os.listdir() if os.path.isdir(x)]
        if len(dirs) != 1:
            raise Exception('There should be only one directory after extracting the file.')
        os.rename(dirs[0], 'images')
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        path = data['path'] + os.path.sep + data['file']
        date = [utils.get_image_date(os.path.join(self.root, p)) for p in path]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': path,
            'identity': folders[1].str.strip('cow_').astype(int),
            'date': date,
        })
        return self.finalize_catalogue(df)
