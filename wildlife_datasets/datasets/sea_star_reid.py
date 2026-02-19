import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Community Data License Agreement – Permissive, Version 1.0',
    'licenses_url': 'https://cdla.dev/permissive-1-0/',
    'url': 'https://lila.science/sea-star-re-id-2023/',
    'publication_url': 'https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14278',
    'cite': 'wahltinez2024open',
    'animals': {'sea star'},
    'animals_simple': 'sea stars',
    'real_animals': True,
    'year': 2023,
    'reported_n_total': 2187,
    'reported_n_individuals': 95,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 1689,
}

class SeaStarReID2023(DownloadURL, WildlifeDataset):
    summary = summary
    url = 'https://storage.googleapis.com/public-datasets-lila/sea-star-re-id/sea-star-re-id.zip'
    archive = 'sea-star-re-id.zip'

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
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
