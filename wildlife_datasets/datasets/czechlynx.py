import os
import pandas as pd
from .datasets import WildlifeDataset
from .downloads import DownloadKaggle
from .utils import parse_bbox_mask

summary = {
    'licenses': 'Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
    'url': 'https://www.kaggle.com/datasets/picekl/czechlynx/',
    'publication_url': 'https://arxiv.org/abs/2506.04931',
    'cite': 'picek2025czechlynx',
    'animals': {'lynx'},
    'animals_simple': 'cats',
    'real_animals': True,
    'year': 2025,
    'reported_n_total': 37440,
    'reported_n_individuals': 219,
    'wild': True,
    'clear_photos': False,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': True,
    'span': '15 years',
    'size': 13000,
}

class CzechLynx(DownloadKaggle, WildlifeDataset):
    outdated_dataset = True
    summary = summary
    kaggle_url = 'picekl/czechlynx'
    kaggle_type = 'datasets'

    def create_catalogue(self, split: str = 'split-geo_aware') -> pd.DataFrame:
        """
        Creates a comprehensive catalogue DataFrame for the CzechLynx dataset.

        This method reads the dataset's metadata file, processes essential fields required
        by the framework, and returns a detailed dataframe containing information about
        each animal observation and its associated metadata.

        Args:
            split (str): One of the following options:
                - 'split-geo_aware': Split based on spatial location.
                - 'split-time_open': Time-aware split (open-world).
                - 'split-time_closed': Time-aware split (closed-world).
                Default is 'split-geo_aware'.

        Returns:
            pd.DataFrame: A dataframe including the following columns:

                - image_id (str): Unique identifier for each row.
                - identity (str): Identity of the depicted individual animal.
                                  If unknown, standardized to the framework’s unknown label.
                - path (str): Relative path to the image file.
                - source (str): Dataset partition or region (e.g., 'beskydy').
                - date (datetime): Observation date. Converted to `datetime`.
                - relative_age (float): Age estimate relative to first appearance.
                - encounter (int): Unique encounter identifier.
                - coat_pattern (str): Coat pattern or marking description.
                - location (str): Specific trap or site location.
                - cell_code (str): Grid cell reference (e.g., 10km spatial cell).
                - latitude (float): Latitude of the observation point.
                - longitude (float): Longitude of the observation point.
                - trap_id (str): Unique identifier for the camera trap.
                - split-geo_aware (str): Spatial split category.
                - split-time_open (str): Time-aware open-world split.
                - split-time_closed (str): Time-aware closed-world split.

        Notes:
            - The `unique_name` column is renamed to `identity` and then dropped.
            - No filtering is applied to the split; the selected column is kept for downstream use.
            - All metadata columns are retained except explicitly replaced ones.
        """

        # Load metadata
        metadata_path = os.path.join(self.root, 'metadata.csv')
        df = pd.read_csv(metadata_path)

        # Add required columns
        df['image_id'] = df.index.astype(str)
        df['identity'] = df['unique_name']
        idx = ~df['date'].isnull()
        df.loc[idx, 'date'] = df.loc[idx, 'date'].apply(lambda x: str(x)[6:10] + '-' + str(x)[3:5] + '-' + str(x)[:2])
        df.drop(columns=['unique_name'], inplace=True)

        # Keep only selected split column, rename it
        df['original_split'] = df[split]
        df.drop(columns=['unique_name', 'split-geo_aware', 'split-time_open', 'split-time_closed'],
                errors='ignore', inplace=True)

        return self.finalize_catalogue(df)


summary_v2 = {
    'licenses': 'Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
    'url': 'https://www.kaggle.com/datasets/picekl/czechlynx/',
    'publication_url': 'https://arxiv.org/abs/2506.04931',
    'cite': 'picek2025czechlynx',
    'animals': {'lynx'},
    'animals_simple': 'cats',
    'real_animals': True,
    'year': 2025,
    'reported_n_total': 39760,
    'reported_n_individuals': 319,
    'wild': True,
    'clear_photos': False,
    'pose': 'multiple',
    'unique_pattern': True,
    'from_video': False,
    'cropped': False,
    'span': '15 years',
    'size': 11000,
}

class CzechLynxv2(CzechLynx):
    outdated_dataset = False
    summary = summary_v2

    def create_catalogue(self, split: str = 'split-geo_aware') -> pd.DataFrame:
        """
        Creates a comprehensive catalogue DataFrame for the CzechLynx dataset.

        This method reads the dataset's metadata file, processes essential fields required
        by the framework, and returns a detailed dataframe containing information about
        each animal observation and its associated metadata.

        Args:
            split (str): One of the following options:
                - 'split-geo_aware': Split based on spatial location.
                - 'split-time_open': Time-aware split (open-world).
                - 'split-time_closed': Time-aware split (closed-world).
                - 'split-pose': Split for pose estimation.
                Default is 'split-geo_aware'.

        Returns:
            pd.DataFrame: A dataframe including the following columns:

                - image_id (str): Unique identifier for each row.
                - identity (str): Identity of the depicted individual animal.
                                  If unknown, standardized to the framework’s unknown label.
                - path (str): Relative path to the image file.
                - source (str): Dataset partition or region (e.g., 'beskydy').
                - date (datetime): Observation date. Converted to `datetime`.
                - mask (dict): Segmentation mask for the animal.
                - relative_age (float): Age estimate relative to first appearance.
                - encounter (int): Unique encounter identifier.
                - coat_pattern (str): Coat pattern or marking description.
                - location (str): Specific trap or site location.
                - cell_code (str): Grid cell reference (e.g., 10km spatial cell).
                - latitude (float): Latitude of the observation point.
                - longitude (float): Longitude of the observation point.
                - trap_id (str): Unique identifier for the camera trap.
                - pose (dict): Pose of the animals (head, legs, ...).
                - split-geo_aware (str): Spatial split category.
                - split-time_open (str): Time-aware open-world split.
                - split-time_closed (str): Time-aware closed-world split.
                - split-pose (str): Split for pose estimation.

        Notes:
            - The `unique_name` column is renamed to `identity` and then dropped.
            - The `mask` column is renamed to `segmentation` and then dropped.
            - No filtering is applied to the split; the selected column is kept for downstream use.
            - All metadata columns are retained except explicitly replaced ones.
        """

        # Load real metadata
        metadata_path = os.path.join(self.root, 'CzechLynxDataset-Metadata-Real.csv')
        df1 = pd.read_csv(metadata_path)
        df1['real_animal'] = True

        # Load synthetic metadata
        metadata_path = os.path.join(self.root, 'CzechLynxDataset-Metadata-Synthetic.csv')
        df2 = pd.read_csv(metadata_path)
        df2['real_animal'] = False

        # Merge both metadata
        df = pd.concat((df1, df2)).reset_index(drop=True)

        # Add required columns
        df['image_id'] = df.index
        df['identity'] = df['unique_name']
        idx = ~df['date'].isnull()
        df.loc[idx, 'date'] = df.loc[idx, 'date'].apply(lambda x: str(x)[6:10] + '-' + str(x)[3:5] + '-' + str(x)[:2])

        # Keep only selected split column, rename it
        df['original_split'] = df[split]
        df['segmentation'] = df['mask'].apply(parse_bbox_mask)
        df['pose'] = df['pose'].apply(parse_bbox_mask)
        df.drop(columns=['unique_name', 'split-geo_aware', 'split-time_open', 'split-time_closed', 'split-pose', 'mask'], inplace=True)

        return self.finalize_catalogue(df)
