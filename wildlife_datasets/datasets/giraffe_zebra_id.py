import pandas as pd

from .datasets_wildme import WildlifeDatasetWildMe
from .downloads import DownloadURL

summary = {
    'licenses': 'Community Data License Agreement – Permissive',
    'licenses_url': 'https://cdla.dev/permissive-1-0/',
    'url': 'https://lila.science/datasets/great-zebra-giraffe-id',
    'publication_url': 'https://aaai.org/papers/15245-15245-animal-population-censusing-at-scale-with-citizen-science-and-photographic-identification/',
    'cite': 'parham2017animal',
    'animals': {'giraffe masai', 'zebra plains'},
    'animals_simple': 'giraffes+zebras',
    'real_animals': True,
    'year': 2017,
    'reported_n_total': 6925,
    'reported_n_individuals': 2056,
    'wild': True,
    'clear_photos': True,
    'pose': 'double',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': '12 days',
    'size': 10433,
}

class GiraffeZebraID(DownloadURL, WildlifeDatasetWildMe):
    summary = summary
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/gzgc.coco.tar.gz'
    archive = 'gzgc.coco.tar.gz'
    
    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('gzgc', 2020)
