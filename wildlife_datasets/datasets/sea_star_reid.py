import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class SeaStarReID2023(DatasetFactory):
    summary = summary['SeaStarReID2023']
    url = 'https://storage.googleapis.com/public-datasets-lila/sea-star-re-id/sea-star-re-id.zip'
    archive = 'sea-star-re-id.zip'
    
    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        species = folders[1].str[:4].replace({'Anau': 'Anthenea australiae', 'Asru': 'Asteria rubens'})
        path = data['path'] + os.path.sep + data['file']
        date = [utils.get_image_date(os.path.join(self.root, p)) for p in path]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': path,
            'identity': folders[1],
            'species': species,
            'date': date
        })
        return self.finalize_catalogue(df)
