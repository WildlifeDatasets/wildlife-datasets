import os
import pandas as pd
import numpy as np
from typing import Optional, List
import json
import datetime

from .. import downloads
from .metadata import metadata
from . import utils


class DatasetFactory():
    def __init__(
        self, 
        root: str,
        df_full: Optional[pd.DataFrame] = None,
        download: bool = False,
        **kwargs
        ):

        self.root = root
        if download and hasattr(self, 'download'): 
            self.download.get_data(root)
        if df_full is None:
            self.df_full = self.create_catalogue(**kwargs)
        else:
            self.df_full = df_full
        self.df = self.create_catalogue_trainable(self.df_full)

    def create_catalogue(self):
        '''
        Creates a dataframe catalogue summarizing the dataset.
        This method is dataset specific and each dataset needs to override it.
        '''
        raise NotImplementedError()

    def create_catalogue_trainable(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Creates a dataframe catalogue summarizing the dataset.
        It should be more prepared for machine learning techniques than create_catalogue().
        '''
        df = df.groupby('identity').filter(lambda x : len(x) >= 2)
        df = df[df['identity'] != 'unknown']
        df.reset_index(drop=True, inplace=True)
        return df
    
    def finalize_catalogue(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Finalizes catalogue dataframe and runs checks for errors.
        '''
        df = self.reorder_df(df)
        df = self.remove_constant_columns(df)
        self.check_unique_id(df)
        self.check_files_exist(df['path'])
        if 'segmentation' in df.columns:
            self.check_files_exist(df['segmentation'])
        return df

    def reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Reorders columns in the dataframe.
        Columns specified in default_order go first.
        '''
        default_order = ['id', 'identity', 'path', 'bbox', 'segmentation', 'position', 'species', 'keypoints']
        df_names = list(df.columns)
        col_names = []
        for name in default_order:
            if name in df_names:
                col_names.append(name)
        for name in df_names:
            if name not in default_order:
                col_names.append(name)
        
        df = df.sort_values('id').reset_index(drop=True)
        return df.reindex(columns=col_names)

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Removes columns with single unique value.
        '''
        for df_name in list(df.columns):
            if df[df_name].astype('str').nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df

    def check_unique_id(self, df: pd.DataFrame) -> None:
        '''
        Checks if values in the 'id' column are unique.
        '''
        if len(df['id'].unique()) != len(df):
            raise(Exception('Image ID not unique.'))

    def check_files_exist(self, col: pd.Series) -> None:
        '''
        Checks if paths in a given column exist.
        '''
        for path in col:
            if type(path) == str and not os.path.exists(os.path.join(self.root, path)):
                raise(Exception('Path does not exist:' + os.path.join(self.root, path)))

    def split_data(self, splitter):
        '''
        Splits data by a splitter class with scikit-like API.
        '''
        indexes = np.arange(len(self.df))
        labels = self.df['identity']

        splits = []
        for train, valid in splitter.split(indexes, labels):
            splits.append([
                self.df.iloc[indexes[train]],
                self.df.iloc[indexes[valid]],
            ])

        if len(splits) == 1:
            return splits[0]
        else:
            return splits


class DatasetFactoryWildMe(DatasetFactory):
    def create_catalogue_wildme(self, prefix: str, year: int) -> pd.DataFrame:
        # Get paths for annotation JSON file and for folder with images
        path_json = os.path.join(prefix + '.coco', 'annotations', 'instances_train' + str(year) + '.json')
        path_images = os.path.join(prefix + '.coco', 'images', 'train' + str(year))

        # Load annotations JSON file
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Check whether segmentation is different from a box
        for ann in data['annotations']:
            if len(ann['segmentation']) != 1:
                raise(Exception('Wrong number of segmentations'))
        
        # Extract the data from the JSON file
        create_dict = lambda i: {'identity': i['name'], 'bbox': utils.segmentation_bbox(i['segmentation'][0]), 'image_id': i['image_id'], 'category_id': i['category_id'], 'segmentation': i['segmentation'][0], 'position': i['viewpoint']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['file_name'], 'image_id': i['id'], 'date': i['date_captured']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])
        species = pd.DataFrame(data['categories'])
        species = species.rename(columns={'id': 'category_id', 'name': 'species'})

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, species, how='left', on='category_id')
        df = pd.merge(df, df_images, how='left', on='image_id')

        # Modify some columns
        df['path'] = path_images + os.path.sep + df['file_name']
        df['id'] = range(len(df))    
        df.loc[df['identity'] == '____', 'identity'] = 'unknown'

        # Remove segmentations which are the same as bounding boxes
        ii = []
        for i in range(len(df)):
            ii.append(utils.is_annotation_bbox(df.iloc[i]['segmentation'], df.iloc[i]['bbox'], tol=3))
        df.loc[ii, 'segmentation'] = np.nan

        # Rename empty dates
        df.loc[df['date'] == 'NA', 'date'] = np.nan

        # Remove superficial columns
        df = df.drop(['image_id', 'file_name', 'supercategory', 'category_id'], axis=1)
        if len(df['species'].unique()) == 1:
            df = df.drop(['species'], axis=1)
        return self.finalize_catalogue(df)


class AAUZebraFishID(DatasetFactory):
    download = downloads.aau_zebrafish_id
    metadata = metadata['AAUZebraFishID']
    
    def create_catalogue(self) -> pd.DataFrame:
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
        position = data['Right,Turning,Occlusion,Glitch'].str.split(',', expand=True)[0]
        position.replace('1', 'right', inplace=True)
        position.replace('0', 'left', inplace=True)

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
            'position': position,
        })
        df = df.join(attributes)
        return self.finalize_catalogue(df)


class AerialCattle2017(DatasetFactory):
    download = downloads.aerial_cattle_2017
    metadata = metadata['AerialCattle2017']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'video': folders[2],
        })
        return self.finalize_catalogue(df)


class ATRW(DatasetFactory):
    download = downloads.atrw
    metadata = metadata['ATRW']

    def create_catalogue(self) -> pd.DataFrame:
        # Load information for the reid_train part of the dataset
        ids = pd.read_csv(os.path.join(self.root, 'atrw_anno_reid_train', 'reid_list_train.csv'),
                        names=['identity', 'path'],
                        header=None
                        )
        ids['id'] = ids['path'].str.split('.', expand=True)[0].astype(int)
        ids['split'] = 'train'

        # Load keypoints for the reid_train part of the dataset
        with open(os.path.join(self.root, 'atrw_anno_reid_train', 'reid_keypoints_train.json')) as file:
            keypoints = json.load(file)
        df_keypoints = {
            'path': pd.Series(keypoints.keys()),
            'keypoints': pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        # Merge information for the reid_train part of the dataset
        df_train = pd.merge(ids, data, on='path')
        df_train['path'] = 'atrw_reid_train' + os.path.sep + 'train' + os.path.sep + df_train['path']

        # Load information for the test_plain part of the dataset
        with open(os.path.join(self.root, 'eval_script', 'ATRWEvalScript-main', 'annotations', 'gt_test_plain.json')) as file:
            identity = json.load(file)
        identity = pd.DataFrame(identity)
        ids = pd.read_csv(os.path.join(self.root, 'atrw_anno_reid_test', 'reid_list_test.csv'),
                        names=['path'],
                        header=None
                        )
        ids['id'] = ids['path'].str.split('.', expand=True)[0].astype(int)
        ids['split'] = 'test'
        ids = pd.merge(ids, identity, left_on='id', right_on='imgid')
        ids = ids.drop(['query', 'frame', 'imgid'], axis=1)
        ids.rename(columns = {'entityid': 'identity'}, inplace = True)

        # Load keypoints for the test part of the dataset
        with open(os.path.join(self.root, 'atrw_anno_reid_test', 'reid_keypoints_test.json')) as file:
            keypoints = json.load(file)
        df_keypoints = {
            'path': pd.Series(keypoints.keys()),
            'keypoints': pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        # Merge information for the test_plain part of the dataset
        df_test1 = pd.merge(ids, data, on='path')
        df_test1['path'] = 'atrw_reid_test' + os.path.sep + 'test' + os.path.sep + df_test1['path']

        # Load information for the test_wild part of the dataset
        with open(os.path.join(self.root, 'eval_script', 'ATRWEvalScript-main', 'annotations', 'gt_test_wild.json')) as file:
            identity = json.load(file)
        ids = utils.find_images(os.path.join(self.root, 'atrw_detection_test', 'test'))
        ids['imgid'] = ids['file'].str.split('.', expand=True)[0].astype('int')
        entries = []
        for key in identity.keys():
            for entry in identity[key]:
                bbox = [entry['bbox'][0], entry['bbox'][1], entry['bbox'][2]-entry['bbox'][0], entry['bbox'][3]-entry['bbox'][1]]
                entries.append({'imgid': int(key), 'bbox': bbox, 'identity': entry['eid']})
        entries = pd.DataFrame(entries)

        # Merge information for the test_wild part of the dataset
        df_test2 = pd.merge(ids, entries, on='imgid')
        df_test2['path'] = 'atrw_detection_test' + os.path.sep + 'test' + os.path.sep + df_test2['file']
        df_test2['id'] = df_test2['imgid'].astype(str) + '_' + df_test2['identity'].astype(str)
        df_test2['split'] = 'test'
        df_test2 = df_test2.drop(['file', 'imgid'], axis=1)

        # Finalize the dataframe
        df = pd.concat([df_train, df_test1, df_test2])
        df['id'] = utils.create_id(df.id.astype(str))
        return self.finalize_catalogue(df)


class BelugaID(DatasetFactoryWildMe):
    download = downloads.beluga_id
    metadata = metadata['BelugaID']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('beluga', 2022)



class BirdIndividualID(DatasetFactory):
    download = downloads.bird_individual_id
    metadata = metadata['BirdIndividualID']
    prefix1 = 'Original_pictures'
    prefix2 = 'IndividualID'

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        path = os.path.join(self.root, self.prefix1, self.prefix2)
        data = utils.find_images(path)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Remove images with multiple labels
        idx = folders[2].str.contains('_')
        data = data.loc[~idx]
        folders = folders.loc[~idx]

        # Remove some problems with the sociable_weavers/Test_dataset
        if folders.shape[1] == 4:
            idx = folders[3].isnull()
            folders.loc[~idx, 2] = folders.loc[~idx, 3]

        # Extract information from the folder structure
        split = folders[1].replace({'Test_datasets': 'test', 'Test': 'test', 'Train': 'train', 'Val':'val'})
        identity = folders[2]
        species = folders[0]

        # Finalize the dataframe
        df1 = pd.DataFrame({    
            'id': utils.create_id(split + data['file']),
            'path': self.prefix1 + os.path.sep + self.prefix2 + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'species': species,
            'split': split,
        })

        # Add images without labels
        path = os.path.join(self.root, self.prefix1, 'New_birds')
        data = utils.find_images(path)
        species = data['path']

        # Finalize the dataframe
        df2 = pd.DataFrame({    
            'id': utils.create_id(data['file']),
            'path': self.prefix1 + os.path.sep + 'New_birds' + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': 'unknown',
            'species': species,
            'split': 'unassigned',
        })
        return self.finalize_catalogue(pd.concat([df1, df2]))


class BirdIndividualIDSegmented(BirdIndividualID):
    prefix1 = 'Cropped_pictures'
    prefix2 = 'IndividuaID'


class CTai(DatasetFactory):
    download = downloads.c_tai
    metadata = metadata['CTai']

    def create_catalogue(self) -> pd.DataFrame:
        # Define the wrong identity names
        replace_names = [
            ('Adult', 'unknown'),
            ('Akouba', 'Akrouba'),
            ('Freddy', 'Fredy'),
            ('Ibrahiim', 'Ibrahim'),
            ('Liliou', 'Lilou'),
            ('Wapii', 'Wapi'),
            ('Woodstiock', 'Woodstock')
        ]
            
        # Load information about the dataset
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CTai',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_ctai.txt'), header=None, sep=' ')
        
        # Extract keypoints from the information
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })

        # Replace the wrong identities
        for replace_tuple in replace_names:
            df['identity'] = df['identity'].replace({replace_tuple[0]: replace_tuple[1]})
        return self.finalize_catalogue(df)


class CZoo(DatasetFactory):
    download = downloads.c_zoo
    metadata = metadata['CZoo']

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CZoo',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_czoo.txt'), header=None, sep=' ')

        # Extract keypoints from the information
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })
        return self.finalize_catalogue(df)


class Cows2021(DatasetFactory):
    download = downloads.cows_2021
    metadata = metadata['Cows2021']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure
        ii = (folders[2] == 'Identification') & (folders[3] == 'Test')
        folders = folders.loc[ii]
        data = data.loc[ii]

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[4].astype(int),
        })
        return self.finalize_catalogue(df)


class Drosophila(DatasetFactory):
    download = downloads.drosophila
    metadata = metadata['Drosophila']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure
        data['identity'] = 'unknown'
        for i_week in range(1, 4):
            idx1 = folders[0].str.startswith('week' + str(i_week))
            idx2 = folders[1] == 'val'
            idx3 = folders[2].isnull()
            data.loc[idx1 & ~idx2, 'identity'] = (i_week-1)*20 + folders.loc[idx1 & ~idx2, 1].astype(int)
            data.loc[idx1 & idx2 & ~idx3, 'identity'] = (i_week-1)*20 + folders.loc[idx1 & idx2 & ~idx3, 2].astype(int)
            data.loc[idx1 & ~idx2, 'split'] = 'train'
            data.loc[idx1 & idx2, 'split'] = 'val'
        
        # Create id and path
        data['id'] = utils.create_id(folders[0] + data['file'])
        data['path'] = data['path'] + os.path.sep + data['file']
        
        # Finalize the dataframe
        df = data.drop(['file'], axis=1)
        return self.finalize_catalogue(df)


class FriesianCattle2015(DatasetFactory):
    download = downloads.friesian_cattle_2015
    metadata = metadata['FriesianCattle2015']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        # Extract information from the folder structure
        split = folders[1].replace({'Cows-testing': 'test', 'Cows-training': 'train'})
        identity = folders[2].str.strip('Cow').astype(int)

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)


class FriesianCattle2017(DatasetFactory):
    download = downloads.friesian_cattle_2017
    metadata = metadata['FriesianCattle2017']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)


class GiraffeZebraID(DatasetFactoryWildMe):
    download = downloads.giraffe_zebra_id
    metadata = metadata['GiraffeZebraID']
    
    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('gzgc', 2020)


class Giraffes(DatasetFactory):
    download = downloads.giraffes
    metadata = metadata['Giraffes']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        path = os.path.join('pbil.univ-lyon1.fr', 'pub', 'datasets', 'miele2021')
        data = utils.find_images(os.path.join(self.root, path))
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure
        clusters = folders[0] == 'clusters'
        data, folders = data[clusters], folders[clusters]

        # Finalize the dataframe
        df = pd.DataFrame({    
            'id': utils.create_id(data['file']),
            'path': path + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
        })
        return self.finalize_catalogue(df)


class HappyWhale(DatasetFactory):
    download = downloads.happy_whale
    metadata = metadata['HappyWhale']
    
    def create_catalogue(self) -> pd.DataFrame:
        # Define the wrong species names
        replace_names = [
            ('bottlenose_dolpin', 'bottlenose_dolphin'),
            ('kiler_whale', 'killer_whale'),
        ]

        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = pd.DataFrame({
            'id': data['image'].str.split('.', expand=True)[0],
            'path': 'train_images' + os.path.sep + data['image'],
            'identity': data['individual_id'],
            'species': data['species'],
            'split': 'train'
            })

        # Replace the wrong species names            
        for replace_tuple in replace_names:
            df1['species'] = df1['species'].replace({replace_tuple[0]: replace_tuple[1]})

        test_files = utils.find_images(os.path.join(self.root, 'test_images'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        # Load the testing data
        df2 = pd.DataFrame({
            'id': test_files.str.split('.', expand=True)[0],
            'path': 'test_images' + os.path.sep + test_files,
            'identity': 'unknown',
            'species': np.nan,
            'split': 'test'
            })
        
        # Finalize the dataframe        
        df = pd.concat([df1, df2])    
        return self.finalize_catalogue(df)


class HumpbackWhaleID(DatasetFactory):
    download = downloads.humpback_whale
    metadata = metadata['HumpbackWhaleID']

    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data.loc[data['Id'] == 'new_whale', 'Id'] = 'unknown'
        df1 = pd.DataFrame({
            'id': data['Image'].str.split('.', expand=True)[0],
            'path': 'train' + os.path.sep + data['Image'],
            'identity': data['Id'],
            'split': 'train'
            })
        
        # Find all testing images
        test_files = utils.find_images(os.path.join(self.root, 'test'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        # Create the testing dataframe
        df2 = pd.DataFrame({
            'id': test_files.str.split('.', expand=True)[0],
            'path': 'test' + os.path.sep + test_files,
            'identity': 'unknown',
            'split': 'test'
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])    
        return self.finalize_catalogue(df)


class HyenaID2022(DatasetFactoryWildMe):
    download = downloads.hyena_id_2022
    metadata = metadata['HyenaID2022']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('hyena', 2022)


class IPanda50(DatasetFactory):
    download = downloads.ipanda_50
    metadata = metadata['IPanda50']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract keypoint information about eyes
        keypoints = []
        for path in data['path'] + os.path.sep + data['file']:
            path_split = os.path.normpath(path).split(os.path.sep)
            path_json = os.path.join('iPanda50-eyes-labels', path_split[1], os.path.splitext(path_split[2])[0] + '.json')
            keypoints_part = np.full(8, np.nan)        
            if os.path.exists(os.path.join(self.root, path_json)):
                with open(os.path.join(self.root, path_json)) as file:
                    keypoints_file = json.load(file)['shapes']
                    if keypoints_file[0]['label'] == 'right_eye':
                        keypoints_part[0:4] = np.reshape(keypoints_file[0]['points'], 4)
                    if keypoints_file[0]['label'] == 'left_eye':
                        keypoints_part[4:8] = np.reshape(keypoints_file[0]['points'], 4)
                    if len(keypoints_file) == 2 and keypoints_file[1]['label'] == 'right_eye':
                        keypoints_part[0:4] = np.reshape(keypoints_file[1]['points'], 4)
                    if len(keypoints_file) == 2 and keypoints_file[1]['label'] == 'left_eye':
                        keypoints_part[4:8] = np.reshape(keypoints_file[1]['points'], 4)
            keypoints.append(list(keypoints_part))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'keypoints': keypoints
            })
        return self.finalize_catalogue(df)


class LeopardID2022(DatasetFactoryWildMe):
    download = downloads.leopard_id_2022
    metadata = metadata['LeopardID2022']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('leopard', 2022)


class LionData(DatasetFactory):
    download = downloads.lion_data
    metadata = metadata['LionData']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[3],
        })
        return self.finalize_catalogue(df)


class MacaqueFaces(DatasetFactory):
    download = downloads.macaque_faces
    metadata = metadata['MacaqueFaces']
    
    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        data = pd.read_csv(os.path.join(self.root, 'MacaqueFaces_ImageInfo.csv'))
        date_taken = [datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%Y-%m-%d') for date in data['DateTaken']]
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),
            'path': 'MacaqueFaces' + os.path.sep + data['Path'].str.strip(os.path.sep) + os.path.sep + data['FileName'],
            'identity': data['ID'],
            'date': pd.Series(date_taken),
            'category': data['Category']
        })
        return self.finalize_catalogue(df)


class NDD20(DatasetFactory):
    download = downloads.ndd20
    metadata = metadata['NDD20']

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the above-water dataset
        with open(os.path.join(self.root, 'ABOVE_LABELS.json')) as file:
            data = json.load(file)
        
        # Analyze the information about the above-water dataset
        entries = []
        for key in data.keys():
            regions = data[key]['regions']
            for region in regions:
                if 'id' in region['region_attributes']:
                    identity = region['region_attributes']['id']
                else:
                    identity = 'unknown'
                segmentation = np.zeros(2*len(region['shape_attributes']['all_points_x']))
                segmentation[0::2] = region['shape_attributes']['all_points_x']
                segmentation[1::2] = region['shape_attributes']['all_points_y']
                entries.append({
                    'identity': identity,
                    'species': region['region_attributes']['species'],
                    'out_of_focus': np.nan,
                    'file_name': data[key]['filename'],
                    'reg_type': region['shape_attributes']['name'],
                    'segmentation': segmentation,
                    'position': 'above'
                })
        
        # Load information about the below-water dataset
        with open(os.path.join(self.root, 'BELOW_LABELS.json')) as file:
            data = json.load(file)
            
        # Analyze the information about the below-water dataset
        for key in data.keys():
            regions = data[key]['regions']
            for region in regions:
                if 'id' in region['region_attributes']:
                    identity = region['region_attributes']['id']
                else:
                    identity = 'unknown'
                segmentation = np.zeros(2*len(region['shape_attributes']['all_points_x']))
                segmentation[0::2] = region['shape_attributes']['all_points_x']
                segmentation[1::2] = region['shape_attributes']['all_points_y']
                entries.append({
                    'identity': identity,
                    'species': 'WBD',
                    'out_of_focus': region['region_attributes']['out of focus'] == 'true',
                    'file_name': data[key]['filename'],
                    'reg_type': region['shape_attributes']['name'],
                    'segmentation': segmentation,
                    'position': 'below'
                })

        # Create the dataframe from entries 
        df = pd.DataFrame(entries)
        if len(df.reg_type.unique()) != 1:
            raise(Exception('Multiple segmentation types'))

        # Finalize the dataframe
        df['id'] = range(len(df))
        df['path'] = df['position'].str.upper() + os.path.sep + df['file_name']
        df = df.drop(['reg_type', 'file_name'], axis=1)
        return self.finalize_catalogue(df)


class NOAARightWhale(DatasetFactory):
    download = downloads.noaa_right_whale
    metadata = metadata['NOAARightWhale']

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the training dataset
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = pd.DataFrame({
            #.str.strip('Cow').astype(int)
            'id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': data['whaleID'],
            })

        # Load information about the testing dataset
        data = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'))
        df2 = pd.DataFrame({
            'id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': 'unknown',
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])    
        return self.finalize_catalogue(df)


class NyalaData(DatasetFactory):
    download = downloads.nyala_data
    metadata = metadata['NyalaData']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure and about position
        identity = folders[3].astype(int)
        position = np.full(len(data), np.nan, dtype=object)
        position[['left' in filename for filename in data['file']]] = 'left'
        position[['right' in filename for filename in data['file']]] = 'right'

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'position': position,
        })
        return self.finalize_catalogue(df)   


class OpenCows2020(DatasetFactory):
    download = downloads.open_cows_2020
    metadata = metadata['OpenCows2020']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Select only re-identification dataset
        reid = folders[1] == 'identification'
        folders, data = folders[reid], data[reid]

        # Extract information from the folder structure
        split = folders[3]
        identity = folders[4]

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)    


class SealID(DatasetFactory):
    download = downloads.seal_id
    metadata = metadata['SealID']
    prefix = 'source_'

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        data = pd.read_csv(os.path.join(self.root, 'full images', 'annotation.csv'))

        # Finalize the dataframe
        df = pd.DataFrame({    
            'id': data['file'].str.split('.', expand=True)[0],
            'path': 'full images' + os.path.sep + self.prefix + data['reid_split'] + os.path.sep + data['file'],
            'identity': data['class_id'].astype(int),
            'reid_split': data['reid_split'],
            'segmentation_split': data['segmentation_split'],
        })
        return self.finalize_catalogue(df)


class SealIDSegmented(SealID):
    prefix = 'segmented_'


class SeaTurtleID(DatasetFactory):
    download = downloads.sea_turtle_id
    metadata = metadata['SeaTurtleID']

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = 'annotations_new.json'
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Extract dtaa from the JSON file
        create_dict = lambda i: {'id': i['id'], 'bbox': i['bbox'], 'image_id': i['image_id'], 'identity': i['identity'], 'segmentation': i['segmentation'], 'position': i['position']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, on='image_id')
        df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
        df = df.drop(['image_id', 'file_name'], axis=1)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:])

        # Finalize the dataframe
        return self.finalize_catalogue(df)


class SeaTurtleIDHeads(DatasetFactory):
    # TODO: add download and metadata
    #download = downloads.sea_turtle_id_heads
    #metadata = metadata['SeaTurtleIDHeads']

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = 'annotations.json'
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Extract dtaa from the JSON file
        create_dict = lambda i: {'id': i['id'], 'image_id': i['image_id'], 'identity': i['identity'], 'position': i['position']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, on='image_id')
        df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
        df = df.drop(['image_id', 'file_name'], axis=1)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:])

        # Finalize the dataframe
        return self.finalize_catalogue(df)


class SMALST(DatasetFactory):
    download = downloads.smalst
    metadata = metadata['SMALST']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(os.path.join(self.root, 'zebra_training_set', 'images'))
        
        # Extract information about the images
        path = data['file'].str.strip('zebra_')
        data['identity'] = path.str[0]
        data['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        data['path'] = 'zebra_training_set' + os.path.sep + 'images' + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        # Find all masks in root
        masks = utils.find_images(os.path.join(self.root, 'zebra_training_set', 'bgsub'))
        
        # Extract information about the images
        path = masks['file'].str.strip('zebra_')
        masks['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        masks['segmentation'] = 'zebra_training_set' + os.path.sep + 'bgsub' + os.path.sep + masks['file']
        masks = masks.drop(['path', 'file'], axis=1)

        # Finalize the dataframe
        df = pd.merge(data, masks, on='id')
        return self.finalize_catalogue(df)


class StripeSpotter(DatasetFactory):
    download = downloads.stripe_spotter
    metadata = metadata['StripeSpotter']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)

        # Extract information about the images
        data['index'] = data['file'].str[-7:-4].astype(int)
        category = data['file'].str.split('-', expand=True)[0]
        data = data[category == 'img'] # Only full images
        
        # Load additional information
        data_aux = pd.read_csv(os.path.join(self.root, 'data', 'SightingData.csv'))
        data = pd.merge(data, data_aux, how='left', left_on='index', right_on='#imgindex')
        data.loc[data['animal_name'].isnull(), 'animal_name'] = 'unknown'
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path':  data['path'] + os.path.sep + data['file'],
            'identity': data['animal_name'],
            'bbox': pd.Series([[int(a) for a in b.split(' ')] for b in data['roi']]),
            'position': data['flank'],
            'photo_quality': data['photo_quality'],
        })
        return self.finalize_catalogue(df)  


class WhaleSharkID(DatasetFactoryWildMe):
    download = downloads.whale_shark_id
    metadata = metadata['WhaleSharkID']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('whaleshark', 2020)


class WNIGiraffes(DatasetFactory):
    download = downloads.wni_giraffes
    metadata = metadata['WNIGiraffes']

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        # Load information about keypoints
        with open(os.path.join(self.root, 'wni_giraffes_train.json')) as file:
            keypoints = json.load(file)
        
        # Extract information about keypoints
        create_dict = lambda i: {'file': os.path.split(i['filename'])[1], 'keypoints': self.extract_keypoints(i)}
        df_keypoints = pd.DataFrame([create_dict(i) for i in keypoints['annotations']])

        # Merge information about images and keypoints
        data = pd.merge(data, df_keypoints, how='left', on='file')
        data['id'] = utils.create_id(data['file'])
        data['identity'] = folders[1].astype(int)
        data['path'] = data['path'] + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        # Finalize the dataframe
        return self.finalize_catalogue(data)

    def extract_keypoints(self, row: pd.DataFrame) -> List[float]:
        keypoints = [row['keypoints']['too']['median_x'], row['keypoints']['too']['median_y'],
                row['keypoints']['toh']['median_x'], row['keypoints']['toh']['median_y'],
                row['keypoints']['ni']['median_x'], row['keypoints']['ni']['median_y'],
                row['keypoints']['fbh']['median_x'], row['keypoints']['fbh']['median_y'],
            ]
        keypoints = np.array(keypoints)
        keypoints[keypoints == None] = np.nan
        return list(keypoints)


class ZindiTurtleRecall(DatasetFactory):
    download = downloads.zindi_turtle_recall
    metadata = metadata['ZindiTurtleRecall']

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the training images
        data_train = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data_train['split'] = 'train'

        # Load information about the testing images
        data_test = pd.read_csv(os.path.join(self.root, 'test.csv'))
        data_test['split'] = 'test'

        # Load information about the additional images
        data_extra = pd.read_csv(os.path.join(self.root, 'extra_images.csv'))
        data_extra['split'] = 'unassigned'        

        # Finalize the dataframe
        data = pd.concat([data_train, data_test, data_extra])
        data = data.reset_index(drop=True)        
        data.loc[data['turtle_id'].isnull(), 'turtle_id'] = 'unknown'
        df = pd.DataFrame({
            'id': data['image_id'],
            'path': 'images' + os.path.sep + data['image_id'] + '.JPG',
            'identity': data['turtle_id'],
            'position': data['image_location'].str.lower(),
            'split': data['split'],
        })
        return self.finalize_catalogue(df)