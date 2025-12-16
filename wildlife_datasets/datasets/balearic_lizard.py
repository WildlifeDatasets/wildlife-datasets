import os
import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle


summary = {
    'licenses': 'Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
    'url': "https://www.kaggle.com/datasets/roberalcaraz/baleariclizard",
    'publication_url': None,
    'cite': None,
    'animals': {'balearic lizard'},
    'animals_simple': 'lizards',
    'real_animals': True,
    'year': 2025,
    'reported_n_total': 4619,
    'reported_n_individuals': 1009,
    'wild': True,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': '15 years',
}


class BalearicLizard(DownloadKaggle, WildlifeDataset):
    """MegaDescriptor BalearicLizard dataset prepared for WildlifeDatasets."""

    summary = summary
    kaggle_url = 'roberalcaraz/baleariclizard'
    kaggle_type = 'datasets'

    def create_catalogue(self, metadata_filename: str = 'metadata.csv') -> pd.DataFrame:
        """Create the catalogue DataFrame expected by WildlifeDatasets and MegaDescriptor.

        The default `metadata_filename` matches the request in the MegaDescriptor
        documentation. If the CSV is nested under a `data/` directory, the method
        transparently retries with that prefix.

        Args:
            metadata_filename: Relative path to the metadata CSV inside ``self.root``.

        Returns:
            pd.DataFrame: A dataframe including the following columns:

                - image_id (str): Unique identifier for each row.
                - id (str): Identity of the depicted individual animal.
                                  If unknown, standardized to the framework’s unknown label.
                - path (str): Relative path to the image file.
                - date (datetime): Observation date. Converted to `datetime`.
        """

        # Load metadata
        metadata_path = os.path.join(self.root, 'curt_metadata.csv')
        df = pd.read_csv(metadata_path)
        
        return self.finalize_catalogue(df)
