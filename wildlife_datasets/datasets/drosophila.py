import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .downloads import DownloadURL

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/JP4WDF&version=1.3&selectTab=termsTab',
    'url': 'https://github.com/j-schneider/fly_eye',
    'publication_url': 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0205043',
    'cite': 'schneider2018can',
    'animals': {'drosophila'},
    'animals_simple': 'flies',
    'real_animals': True,
    'year': 2018,
    'reported_n_total': 2592000,
    'reported_n_individuals': 60,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': True,
    'from_video': True,
    'cropped': True,
    'span': '3 days',
    'size': 74856,
}

class Drosophila(DownloadURL, DatasetFactory):
    summary = summary
    downloads = [
        ('https://dataverse.scholarsportal.info/api/access/datafile/71066', 'week1_Day1_train_01to05.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71067', 'week1_Day1_train_06to10.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71068', 'week1_Day1_train_11to15.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71069', 'week1_Day1_train_16to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71065', 'week1_Day1_val.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71071', 'week1_Day2_train_01to05.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71072', 'week1_Day2_train_06to10.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71073', 'week1_Day2_train_11to15.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71075', 'week1_Day2_train_16to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71070', 'week1_Day2_val.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71077', 'week1_Day3_01to04.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71078', 'week1_Day3_05to08.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71079', 'week1_Day3_09to12.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71080', 'week1_Day3_13to16.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71081', 'week1_Day3_17to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71083', 'week2_Day1_train_01to05.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71084', 'week2_Day1_train_06to10.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71085', 'week2_Day1_train_11to15.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71086', 'week2_Day1_train_16to20.zip'),        
        ('https://dataverse.scholarsportal.info/api/access/datafile/71082', 'week2_Day1_val.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71094', 'week2_Day2_train_01to05.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71095', 'week2_Day2_train_06to10.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71109', 'week2_Day2_train_11to15.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71110', 'week2_Day2_train_16to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71093', 'week2_Day2_val.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71111', 'week2_Day3_01to04.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71112', 'week2_Day3_05to08.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71115', 'week2_Day3_09to12.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71117', 'week2_Day3_13to16.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71118', 'week2_Day3_17to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71119', 'week3_Day1_train_01to05.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71120', 'week3_Day1_train_06to10.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71121', 'week3_Day1_train_11to15.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71124', 'week3_Day1_train_16to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71097', 'week3_Day1_val.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71125', 'week3_Day2_train_01to05.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71126', 'week3_Day2_train_06to10.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71127', 'week3_Day2_train_11to15.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71128', 'week3_Day2_train_16to20.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71107', 'week3_Day2_val.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71129', 'week3_Day3_01to04.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71130', 'week3_Day3_05to08.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71131', 'week3_Day3_09to12.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71132', 'week3_Day3_13to16.zip'),
        ('https://dataverse.scholarsportal.info/api/access/datafile/71133', 'week3_Day3_17to20.zip'),
    ]

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure
        data['identity'] = self.unknown_name
        for i_week in range(1, 4):
            idx1 = folders[0].str.startswith('week' + str(i_week))
            idx2 = folders[1] == 'val'
            idx3 = folders[2].isnull()
            data.loc[idx1 & ~idx2, 'identity'] = ((i_week-1)*20 + folders.loc[idx1 & ~idx2, 1].astype(int)).astype(str)
            data.loc[idx1 & idx2 & ~idx3, 'identity'] = ((i_week-1)*20 + folders.loc[idx1 & idx2 & ~idx3, 2].astype(int)).astype(str)
            data.loc[idx1 & ~idx2, 'split'] = 'train'
            data.loc[idx1 & idx2, 'split'] = 'val'
        
        # Create id and path
        data['id'] = utils.create_id(folders[0] + data['file'])
        data['path'] = data['path'] + os.path.sep + data['file']
        
        # Finalize the dataframe
        df = data.drop(['file'], axis=1)
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
