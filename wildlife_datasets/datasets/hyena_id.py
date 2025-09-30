import pandas as pd
from .datasets_wildme import WildlifeDatasetWildMe
from .downloads import DownloadURL

summary = {
    'licenses': 'Community Data License Agreement – Permissive',
    'licenses_url': 'https://cdla.dev/permissive-1-0/',
    'url': 'https://lila.science/datasets/hyena-id-2022/',
    'publication_url': None,
    'cite': 'botswana2022',
    'animals': {'spotted hyena'},
    'animals_simple': 'hyenas',
    'real_animals': True,
    'year': 2022,
    'reported_n_total': 3129,
    'reported_n_individuals': 256,
    'wild': True,
    'clear_photos': False,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': 'unknown',
    'size': 3441,
}

class HyenaID2022(DownloadURL, WildlifeDatasetWildMe):
    summary = summary
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/hyena.coco.tar.gz'
    archive = 'hyena.coco.tar.gz'

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('hyena', 2022)
