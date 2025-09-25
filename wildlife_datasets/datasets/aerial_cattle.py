import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory

summary = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh',
    'publication_url': 'https://openaccess.thecvf.com/content_ICCV_2017_workshops/w41/html/Andrew_Visual_Localisation_and_ICCV_2017_paper.html',
    'cite': 'andrew2017visual',
    'animals': {'Friesian cattle'},
    'animals_simple': 'cows',
    'real_animals': True,
    'year': 2017,
    'reported_n_total': 46340,
    'reported_n_individuals': 23,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': '1 day',
    'size': 724,
}

class AerialCattle2017(DatasetFactory):
    summary = summary
    url = 'https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip'
    archive = '3owflku95bxsx24643cybxu3qh.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'video': folders[2].astype(int),
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
