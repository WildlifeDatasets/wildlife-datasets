import os
import shutil
import pandas as pd
import numpy as np
from typing import Optional, List
import json
import datetime

from .. import splits
from .metadata import metadata
from . import utils


class DatasetFactory():
    """Base class for creating datasets.

    Attributes:
      df (pd.DataFrame): A full dataframe of the data.
      df_ml (pd.DataFrame): A dataframe of data for machine learning models.
      download (module): Script for downloading the dataset.
      metadata (dict): Metadata of the dataset.
      root (str): Root directory for the data.
      unknown_name (str): Name of the unknown class.
    """

    unknown_name = 'unknown'
    download_warning = '''You are trying to download an already downloaded dataset.
        This message may have happened to due interrupted download or extract.
        To force the download use the `force=True` keyword such as
        get_data(..., force=True) or download(..., force=True).
        '''
    download_mark_name = 'already_downloaded'

    def __init__(
            self, 
            root: str,
            df: Optional[pd.DataFrame] = None,
            **kwargs) -> None:
        """Initializes the class.

        If `df` is specified, it copies it. Otherwise, it creates it
        by the `create_catalogue` method.
        It creates `df_ml` by the `create_catalogue_ml` method.

        Args:
            root (str): Root directory for the data.
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.
        """

        self.root = root
        if df is None:
            self.df = self.create_catalogue(**kwargs)
        else:
            self.df = df.copy()
        self.add_splits()

    @classmethod
    def get_data(cls, root, force=False, **kwargs):
        dataset_name = cls.__name__
        mark_file_name = os.path.join(root, cls.download_mark_name)
        
        already_downloaded = os.path.exists(mark_file_name)
        if already_downloaded and not force:
            print('DATASET %s: DOWNLOADING STARTED.' % dataset_name)
            print(cls.download_warning)
        else:
            print('DATASET %s: DOWNLOADING STARTED.' % dataset_name)
            cls.download(root, force=force, **kwargs)
            print('DATASET %s: EXTRACTING STARTED.' % dataset_name)
            cls.extract(root,  **kwargs)
            print('DATASET %s: FINISHED.\n' % dataset_name)

    @classmethod
    def download(cls, root, force=False, **kwargs):
        dataset_name = cls.__name__
        mark_file_name = os.path.join(root, cls.download_mark_name)
        
        already_downloaded = os.path.exists(mark_file_name)
        if already_downloaded and not force:
            print('DATASET %s: DOWNLOADING STARTED.' % dataset_name)            
            print(cls.download_warning)
        else:
            if os.path.exists(mark_file_name):
                os.remove(mark_file_name)
            with utils.data_directory(root):
                cls._download(**kwargs)
            open(mark_file_name, 'a').close()
        
    @classmethod    
    def extract(cls, root, **kwargs):
        with utils.data_directory(root):
            cls._extract(**kwargs)
        mark_file_name = os.path.join(root, cls.download_mark_name)
        open(mark_file_name, 'a').close()
        
    def create_catalogue(self):
        """Creates the dataframe.

        Raises:
            NotImplementedError: Needs to be implemented by subclasses.
        """

        raise NotImplementedError('Needs to be implemented by subclasses.')
    
    def add_splits(self) -> None:
        """Drops existing splits and adds automatically generated split.

        The split ignores individuals named `self.unknown_name`.
        These rows will not belong to a split.
        The added split is machine-independent.
        It is the closed-set (random) split with 80% in the training set.
        """

        # Drop already existing splits
        cols_to_drop = ['split', 'reid_split', 'segmentation_split']
        self.df = self.df.drop(cols_to_drop, axis=1, errors='ignore')
        
        # Add the default split
        splitter = splits.ClosedSetSplit(0.8, identity_skip=self.unknown_name)
        self.add_split(3, 'split', splitter)

    def add_split(self, position: int, col_name: str, splitter: splits.BalancedSplit) -> None:       
        """Adds a split to the column named col_name.

        Args:
            position (int): Where the split should be placed.
            col_name (str): Name of the column.
            splitter (splits.BalancedSplit): Any class with `split` method
                returning training and testing set indices.                
        """
        idx_train, idx_test = splitter.split(self.df)[0]
        add = {}
        for i in idx_train:
            add[i] = 'train'
        for i in idx_test:
            add[i] = 'test'
        n_col = min(position, len(self.df.columns))
        self.df.insert(n_col, col_name, pd.Series(add))
        
    def finalize_catalogue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders the dataframe and check file paths.

        Reorders the columns and removes constant columns.
        Checks if columns are in correct formats.
        Checks if ids are unique and if all files exist.

        Args:
            df (pd.DataFrame): A full dataframe of the data.

        Returns:
            A full dataframe of the data, slightly modified.
        """

        self.check_required_columns(df)
        self.check_types_columns(df)
        df = self.reorder_df(df)
        df = self.remove_constant_columns(df)
        self.check_unique_id(df)
        self.check_files_exist(df['path'])
        self.check_files_names(df['path'])
        if 'segmentation' in df.columns:
            self.check_files_exist(df['segmentation'])
        return df

    def check_required_columns(self, df: pd.DataFrame) -> None:
        """Check if all required columns are present.

        Args:
            df (pd.DataFrame): A full dataframe of the data.
        """

        for col_name in ['image_id', 'identity', 'path']:
            if col_name not in df.columns:
                raise(Exception('Column %s must be in the dataframe columns.' % col_name))

    def check_types_columns(self, df: pd.DataFrame) -> None:
        """Checks if columns are in correct formats.

        The format are specified in `requirements`, which is list
        of tuples. The first value is the name of the column
        and the second value is a list of formats. The column
        must be at least one of the formats.

        Args:
            df (pd.DataFrame): A full dataframe of the data.
        """

        requirements = [
            ('image_id', ['int', 'str']),
            ('identity', ['int', 'str']),
            ('path', ['str']),
            ('bbox', ['list_numeric']),
            ('date', ['date']),
            ('keypoints', ['list_numeric']),
            ('position', ['str']),
            ('species', ['str', 'list']),
            ('video', ['int']),
        ]
        # Verify if the columns are in correct formats
        for col_name, allowed_types in requirements:
            if col_name in df.columns:
                # Remove empty values to be sure
                col = df[col_name][~df[col_name].isnull()]
                if len(col) > 0:
                    self.check_types_column(col, col_name, allowed_types)
    
    def check_types_column(self, col: pd.Series, col_name: str, allowed_types: List[str]) -> None:
        """Checks if the column `col` is in the format `allowed_types`.

        Args:
            col (pd.Series): Column to be checked.
            col_name (str): Column name used only for raising exceptions.
            allowed_types (List[str]): List of strings with allowed values:
                `int` (all values must be integers),
                `str` (strings),
                `list` (lists),
                `list_numeric` (lists with numeric values),
                `date` (dates as tested by `pd.to_datetime`).
        """

        if 'int' in allowed_types and pd.api.types.is_integer_dtype(col):
            return None
        if 'str' in allowed_types and pd.api.types.is_string_dtype(col):
            return None
        if 'list' in allowed_types and pd.api.types.is_list_like(col):
            check = True
            for val in col:
                if not pd.api.types.is_list_like(val):
                    check = False
                    break
            if check:                
                return None        
        if 'list_numeric' in allowed_types and pd.api.types.is_list_like(col):
            check = True
            for val in col:            
                if not pd.api.types.is_list_like(val) and not pd.api.types.is_numeric_dtype(pd.Series(val)):
                    check = False
                    break
            if check:                
                return None
        if 'date' in allowed_types:
            try:
                pd.to_datetime(col)
                return None
            except:
                pass
        raise(Exception('Column %s has wrong type. Allowed types = %s' % (col_name, str(allowed_types))))

    def reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders rows and columns in the dataframe.

        Rows are sorted based on id.
        Columns are reorder based on the `default_order` list.

        Args:
            df (pd.DataFrame): A full dataframe of the data.

        Returns:
            A full dataframe of the data, slightly modified.
        """

        default_order = ['image_id', 'identity', 'path', 'bbox', 'date', 'keypoints', 'orientation', 'segmentation', 'species']
        df_names = list(df.columns)
        col_names = []
        for name in default_order:
            if name in df_names:
                col_names.append(name)
        for name in df_names:
            if name not in default_order:
                col_names.append(name)
        
        df = df.sort_values('image_id').reset_index(drop=True)
        return df.reindex(columns=col_names)

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes columns with a single unique value.

        Args:
            df (pd.DataFrame): A full dataframe of the data.

        Returns:
            A full dataframe of the data, slightly modified.
        """ 

        for df_name in list(df.columns):
            if df[df_name].astype('str').nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df

    def check_unique_id(self, df: pd.DataFrame) -> None:
        """Checks if values in the id column are unique.

        Args:
            df (pd.DataFrame): A full dataframe of the data.
        """

        if len(df['image_id'].unique()) != len(df):
            raise(Exception('Image ID not unique.'))

    def check_files_exist(self, col: pd.Series) -> None:
        """Checks if paths in a given column exist.

        Args:
            col (pd.Series): A column of a dataframe.
        """

        for path in col:
            if type(path) == str and not os.path.exists(os.path.join(self.root, path)):
                raise(Exception('Path does not exist:' + os.path.join(self.root, path)))

    def check_files_names(self, col: pd.Series) -> None:
        """Checks if paths contain .

        Args:
            col (pd.Series): A column of a dataframe.
        """

        for path in col:
            try:
                path.encode("iso-8859-1")
            except UnicodeEncodeError:
                raise(Exception('Characters in path may cause problems. Please use only ISO-8859-1 characters: ' + os.path.join(path)))

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
        create_dict = lambda i: {'identity': i['name'], 'bbox': utils.segmentation_bbox(i['segmentation'][0]), 'image_id': i['image_id'], 'category_id': i['category_id'], 'segmentation': i['segmentation'][0], 'orientation': i['viewpoint']}
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
        df.loc[df['identity'] == '____', 'identity'] = self.unknown_name

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class AAUZebraFish(DatasetFactory):
    metadata = metadata['AAUZebraFish']
    archive = 'aau-zebrafish-reid.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d aalborguniversity/aau-zebrafish-reid"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#aauzebrafish'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        
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
        orientation = data['Right,Turning,Occlusion,Glitch'].str.split(',', expand=True)[0]
        orientation.replace('1', 'right', inplace=True)
        orientation.replace('0', 'left', inplace=True)

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
            'orientation': orientation,
        })
        df = df.join(attributes)
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class AerialCattle2017(DatasetFactory):
    metadata = metadata['AerialCattle2017']
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


class ATRW(DatasetFactory):
    metadata = metadata['ATRW']
    url = 'https://github.com/cvwc2019/ATRWEvalScript/archive/refs/heads/main.zip'
    archive = 'main.zip'
    downloads = [
        # Wild dataset (Detection)
        ('https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_detection_test.tar.gz', 'atrw_detection_test.tar.gz'),

        # Re-ID dataset
        ('https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_reid_train.tar.gz', 'atrw_reid_train.tar.gz'),
        ('https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_anno_reid_train.tar.gz', 'atrw_anno_reid_train.tar.gz'),
        ('https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_reid_test.tar.gz', 'atrw_reid_test.tar.gz'),
        ('https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_anno_reid_test.tar.gz', 'atrw_anno_reid_test.tar.gz'),
    ]

    @classmethod
    def _download(cls):
        for url, archive in cls.downloads:
            utils.download_url(url, archive)
        # Evaluation scripts
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            archive_name = archive.split('.')[0]
            utils.extract_archive(archive, archive_name, delete=True)
        # Evaluation scripts
        utils.extract_archive(cls.archive, 'eval_script', delete=True)
    
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
        df['id'] = utils.create_id(df['id'].astype(str))
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class BelugaID(DatasetFactoryWildMe):
    metadata = metadata['BelugaID']
    url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/beluga.coco.tar.gz'
    archive = 'beluga.coco.tar.gz'

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('beluga', 2022)

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)


class BirdIndividualID(DatasetFactory):
    metadata = metadata['BirdIndividualID']
    prefix1 = 'Original_pictures'
    prefix2 = 'IndividualID'
    url = 'https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA'
    archive = 'ferreira_et_al_2020.zip'

    @classmethod
    def _download(cls):
        exception_text = '''Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#birdindividualid'''
        raise Exception(exception_text)
        # utils.gdown_download(cls.url, cls.archive, exception_text=exception_text)
    
    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

        # Create new folder for segmented images
        folder_new = os.getcwd() + 'Segmented'
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)

        # Move segmented images to new folder
        folder_move = 'Cropped_pictures'
        shutil.move(folder_move, os.path.join(folder_new, folder_move))

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
            'identity': self.unknown_name,
            'species': species,
            'split': 'unassigned',
        })
        df = pd.concat([df1, df2])
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class BirdIndividualIDSegmented(BirdIndividualID):
    prefix1 = 'Cropped_pictures'
    prefix2 = 'IndividuaID'
    warning = '''You are trying to download or extract a segmented dataset.
        It is already included in its non-segmented version.
        Skipping.'''
    
    @classmethod
    def get_data(cls, root, name=None):
        print(cls.warning)

    @classmethod
    def _download(cls):
        print(cls.warning)

    @classmethod
    def _extract(cls):
        print(cls.warning)

class CTai(DatasetFactory):
    metadata = metadata['CTai']
    url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
    archive = 'master.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('chimpanzee_faces-master/datasets_cropped_chimpanzee_faces/data_CZoo')

    def create_catalogue(self) -> pd.DataFrame:
        # Define the wrong identity names
        replace_names = [
            ('Adult', self.unknown_name),
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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class CZoo(DatasetFactory):
    metadata = metadata['CZoo']
    url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
    archive = 'master.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('chimpanzee_faces-master/datasets_cropped_chimpanzee_faces/data_CTai')

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class Cows2021(DatasetFactory):
    metadata = metadata['Cows2021']
    url = 'https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip'
    archive = '4vnrca7qw1642qlwxjadp87h7.zip'

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
        df['date'] = df['path'].apply(lambda x: self.extract_date(x))
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)

    def extract_date(self, x):
        x = os.path.split(x)[1]
        if x.startswith('image_'):
            x = x[6:]
        if x[7] == '_':
            x = x[8:]
        i1 = x.find('_')
        i2 = x[i1+1:].find('_')
        x = x[:i1+i2+1]
        return datetime.datetime.strptime(x, '%Y-%m-%d_%H-%M-%S').strftime('%Y-%m-%d %H:%M:%S')


class Drosophila(DatasetFactory):
    metadata = metadata['Drosophila']
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

    @classmethod
    def _download(cls):
        for url, archive in cls.downloads:
            utils.download_url(url, archive)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            utils.extract_archive(archive, extract_path=os.path.splitext(archive)[0], delete=True)

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


class FriesianCattle2015(DatasetFactory):
    metadata = metadata['FriesianCattle2015']
    url = 'https://data.bris.ac.uk/datasets/wurzq71kfm561ljahbwjhx9n3/wurzq71kfm561ljahbwjhx9n3.zip'
    archive = 'wurzq71kfm561ljahbwjhx9n3.zip'

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class FriesianCattle2017(DatasetFactory):
    metadata = metadata['FriesianCattle2017']
    url = 'https://data.bris.ac.uk/datasets/2yizcfbkuv4352pzc32n54371r/2yizcfbkuv4352pzc32n54371r.zip'
    archive = '2yizcfbkuv4352pzc32n54371r.zip'

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
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class GiraffeZebraID(DatasetFactoryWildMe):
    metadata = metadata['GiraffeZebraID']
    url = 'https://lilablobssc.blob.core.windows.net/giraffe-zebra-id/gzgc.coco.tar.gz'
    archive = 'gzgc.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
    
    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('gzgc', 2020)


class Giraffes(DatasetFactory):
    metadata = metadata['Giraffes']

    @classmethod
    def _download(cls):
        url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/'
        command = f"wget -rpk -l 10 -np -c --random-wait -U Mozilla {url} -P '.' "
        exception_text = '''Download works only on Linux. Please download it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#giraffes'''
        if os.name == 'posix':
            os.system(command)
        else:
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        pass

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        # Extract information from the folder structure
        clusters = folders[n_folders-1] == 'clusters'
        data, folders = data[clusters], folders[clusters]

        # Finalize the dataframe
        df = pd.DataFrame({    
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[n_folders],
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class HappyWhale(DatasetFactory):
    metadata = metadata['HappyWhale']
    archive = 'happy-whale-and-dolphin.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c happy-whale-and-dolphin --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#happywhale'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        try:
            utils.extract_archive(cls.archive, delete=True)
        except:
            exception_text = '''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#happywhale'''
            raise Exception(exception_text)
    
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
            'identity': self.unknown_name,
            'species': np.nan,
            'split': 'test'
            })
        
        # Finalize the dataframe        
        df = pd.concat([df1, df2])
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class HumpbackWhaleID(DatasetFactory):
    metadata = metadata['HumpbackWhaleID']
    archive = 'humpback-whale-identification.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c humpback-whale-identification --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#humpbackwhale'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        try:
            utils.extract_archive(cls.archive, delete=True)
        except:
            exception_text = '''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#humpbackwhale'''
            raise Exception(exception_text)

    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data.loc[data['Id'] == 'new_whale', 'Id'] = self.unknown_name
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
            'identity': self.unknown_name,
            'split': 'test'
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class HyenaID2022(DatasetFactoryWildMe):
    metadata = metadata['HyenaID2022']
    url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/hyena.coco.tar.gz'
    archive = 'hyena.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('hyena', 2022)


class IPanda50(DatasetFactory):
    metadata = metadata['IPanda50']
    downloads = [
        ('https://drive.google.com/uc?id=1nkh-g6a8JvWy-XsMaZqrN2AXoPlaXuFg', 'iPanda50-images.zip'),
        ('https://drive.google.com/uc?id=1gVREtFWkNec4xwqOyKkpuIQIyWU_Y_Ob', 'iPanda50-split.zip'),
        ('https://drive.google.com/uc?id=1jdACN98uOxedZDT-6X3rpbooLAAUEbNY', 'iPanda50-eyes-labels.zip'),
    ]

    @classmethod
    def _download(cls):
        exception_text = '''Download failed. GDown quota probably reached. Download dataset manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#ipanda50'''
        for url, archive in cls.downloads:
            utils.gdown_download(url, archive, exception_text=exception_text)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            utils.extract_archive(archive, delete=True)

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)

        # Remove non-ASCII characters from image names
        import string
        for _, df_row in df.iterrows():
            path_new = ''.join(c for c in df_row['path'] if c in string.printable)
            if path_new != df_row['path']:
                os.rename(os.path.join(self.root, df_row['path']), os.path.join(self.root, path_new))
                df_row['path'] = path_new
        if len(df) != df['path'].nunique():
            raise(Exception("Non-unique names. Something went wrong when renaming."))

        # Backward compability because of renaming non-ASCII names
        ids_change = [
            ['148e54a1a3a68bde', '00a704e4c2dabee0'],
            ['2d40cf5a2a539fff', '0167f91b4b783389'],
            ['50f109220cba6f14', '02e92626705e1e84'],
            ['2da790802f7c3309', '094a0eef60098f21'],
            ['eba318016427bb9d', '0a030ca1f22cba47'],
            ['b8c9cc36cee72b34', '0d722b90bcdc4855'],
            ['ab29e7e4409cfd97', '0f61b472c54588cc'],
            ['0bebe8c94c83647b', '1028d0c2b4505376'],
            ['37d82e4eaa7a765d', '124b2e39a92766a6'],
            ['3e6a03686b5f2d55', '18b7bcae62315342'],
            ['becf9cb6c9a736ae', '1ed25d59b2b6c354'],
            ['082c11aa0f98a81d', '1feb586750483756'],
            ['212faff57f94936b', '205f4c327235cc4e'],
            ['a40502fa413834d4', '20d8cfcf161d32c8'],
            ['179b1a59c5f9acde', '25086b8930698c3c'],
            ['d8cc24a399b14782', '260bccca8ba09579'],
            ['82f8ea0e97129f5c', '26f68f6e85ee1701'],
            ['ee95cbd6ef25030a', '2881b22a405b0f20'],
            ['fb4e61b9b0daf745', '2cc2825459b60562'],
            ['996d7a70f669ed18', '2f28176d833c2687'],
            ['0e5755065a6623a2', '3051c933b4daa402'],
            ['989886c3ef126ba6', '325ae1820e3f6feb'],
            ['717b9a1bad86b050', '342b22967ee0caea'],
            ['ffe03f63865c867d', '36f5eaf55ce1cc75'],
            ['ae8dcba9506e3838', '38fc50994879f8e2'],
            ['df5f0e3e2644e4b7', '3a6aa0a6cc2106fa'],
            ['747fcac6b444d398', '3a77dcd32cf19784'],
            ['53467ce0e076e9f6', '3abd84677560c571'],
            ['7bc65b71dfa78e4e', '3cd7c19e6228aea1'],
            ['f4a1b7e9a4858f10', '3da27cbdd5c56db8'],
            ['d15f6df4bb7fb3be', '3e3ec0f0d6c70a3f'],
            ['764410ab39686bfd', '3f3e348ca233e6fc'],
            ['68ef26164bae09fe', '40626fcc58f90167'],
            ['136baebcedb2b9e6', '4108b24fbd1ed31f'],
            ['30bb36ca821d89bf', '454504780151a996'],
            ['0b0ccb1a84562c35', '46f1e41e04325436'],
            ['5faf81af57dc10e8', '474732211741dccd'],
            ['38c76d699661761a', '496d1ef5402fdcc2'],
            ['d6625029088acab9', '4bd2f1995732f4e0'],
            ['19f9f39c9288197e', '4c468f1d486bfc23'],
            ['40198a3c13eee3e4', '4eabfb3613025c56'],
            ['9967553ee2ea2a04', '4f6efc6fb7cd4bc3'],
            ['8a6649a278801a4d', '51116e6a4db15546'],
            ['6b7e7f9baab76235', '51843a031a76c43b'],
            ['5f55f610a621ed4e', '5459e35de38ebe61'],
            ['87682645f6ccb6a3', '5c05d69772002b05'],
            ['cc1f7ca695a00399', '5cb2753d3c6c0a37'],
            ['b513ff4959418419', '5d7ba5e54dfee635'],
            ['5530adbc06b39421', '5f50c695235c6579'],
            ['932324d7672541d1', '606903d2ae5d03bc'],
            ['e07c21eed84e32ce', '612d37c49786e485'],
            ['8800920dd42b600c', '615f7a1b6dfc6aa1'],
            ['e7a924e50cb7a091', '6250533422d486ee'],
            ['b2a996e8ac0e604d', '641fa169006b4cdd'],
            ['00589c7b8d6e868e', '6422272c99ff4ada'],
            ['e3326190fd5150bd', '6501aa400e7e1812'],
            ['b7022678458a1cc0', '684c0341172cde02'],
            ['5b3f61ab78af2fb3', '6a844a7a99a85825'],
            ['02174bb0fdd36c92', '6bf3c3f2aa170783'],
            ['b5dd675a22518d56', '6bf5c3b11f2e9f32'],
            ['1f37cdd9dc733c75', '6c5b3803908f96dc'],
            ['a04137b44cbc89be', '6cb61717a4843999'],
            ['9afe1a623392e011', '6ebd36c98a6715c1'],
            ['88487bb6afa52dc1', '75bf38bec0460410'],
            ['9479bab1adbb919d', '79a61a935cf0d45c'],
            ['22a3bf9b7128f969', '7b8d4004da046f46'],
            ['29712c4333085ee6', '7e3b78fb5a5a55d1'],
            ['5b335dadc7a059f0', '828c7a89cdd0bc5c'],
            ['50b0ac629f8c8592', '84f8c9ccada9c7e9'],
            ['f97a5d0285e12b84', '88e6cd70364e54d5'],
            ['f9abb2ee0e40bf9a', '8962c96a27ee5f78'],
            ['34ce980cebf36396', '8c07611797dca302'],
            ['39f2a66dafd37e51', '8ec8ff2987783c4c'],
            ['e4e43360a43f7e98', '91b3ec8c188d115f'],
            ['a1a324eb229ec42f', '9523cb545fcf7f35'],
            ['1b08b95330258f5a', '9963817415c81fc0'],
            ['06c10ac43e46d58e', '9a036d5670fded48'],
            ['39052401c0b2f410', '9c0d1bfb5da26f55'],
            ['a5e83fcca15fe1d0', '9cc2a758c2924c9f'],
            ['034a05a9c31f59ab', '9e36dd73c1c9b3df'],
            ['3f675a3c13583851', '9fa45b2eba285f32'],
            ['d88e763820b371c9', 'a09351d4b3e634c4'],
            ['9161792b0bb4ca75', 'a18c7f86857aea52'],
            ['fbb296586684a614', 'a55025626ff3602a'],
            ['1fb33db508b25f63', 'a6613d2487eeea59'],
            ['435df428a53a8148', 'a981596bd3a169c3'],
            ['d6de1acfc5d8f5d4', 'ac50dbf29524803c'],
            ['e4c46da4c0a5f1b3', 'b296743eecac582a'],
            ['41b213a3ca0cc603', 'bd3a9fda89e35f8d'],
            ['c05c911fabe70667', 'c057a21569bc68de'],
            ['e470316e8eb13680', 'c101fe8dccdfed67'],
            ['ae0ad68cc0e43ade', 'c6b36648ba43e7a2'],
            ['96f97e9c24c8d595', 'ce9c363e1a2a466e'],
            ['85128bf24e8dbe6b', 'd4e29367ef95b8ca'],
            ['ec918c84a3f5adc5', 'd7c8282e00f2efda'],
            ['ae95277f6a4eecb8', 'd8bb58f4ffab3224'],
            ['6e797da86e5e3347', 'dac3a7ba978cc4ce'],
            ['d4a68676c25b5f39', 'daff3497bd006ff6'],
            ['4f0bb78071702349', 'dd1d74a31269f0f6'],
            ['ee186d70e1b2c7c5', 'dd4b98608d12962e'],
            ['f30a2747fd8f75af', 'de22b730dc51c751'],
            ['a5723db64c2583d4', 'e23965a5a4f94b5e'],
            ['81bb9efda8b0a974', 'e585abd35ee270ce'],
            ['fdc1233c2575ec0d', 'e69bc861988489f8'],
            ['c0e44f33589adaad', 'ea0dc21c4d711ca4'],
            ['88677c48ff2cd9b6', 'eb9c0d6ac933a0fb'],
            ['2a4538293242bdb2', 'ed123e244f6cddb3'],
            ['0bf97358e2493682', 'ed957d9afe2d441e'],
            ['0d1639535e024d7e', 'ee5258be8af64408'],
            ['da4d28e176efde20', 'ee86adf7c257ea93'],
            ['d001965e4f1dc51c', 'f3caa9c0051c44ec'],
            ['52979c94473059eb', 'f5cb2f3875e38f72'],
            ['f0f1cb04c08d8442', 'fadd52fa6fbf48b1'],
            ['63f661722a7d5c60', 'fde6956c6d3aa274'],
            ['a5359e0ea14dfd2b', 'ffe2054e86631b8c'],
        ]
        for id_change in ids_change:
            df['image_id'] = df['image_id'].replace(id_change[0], id_change[1])
        
        return self.finalize_catalogue(df)


class LeopardID2022(DatasetFactoryWildMe):
    metadata = metadata['LeopardID2022']
    url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/leopard.coco.tar.gz'
    archive = 'leopard.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('leopard', 2022)


class LionData(DatasetFactory):
    metadata = metadata['LionData']
    url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
    archive = 'main.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('wildlife_reidentification-main/Nyala_Data_Zero')

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class MacaqueFaces(DatasetFactory):
    metadata = metadata['MacaqueFaces']
    
    @classmethod
    def _download(cls):
        downloads = [
            ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces.zip', 'MacaqueFaces.zip'),
            ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces_ImageInfo.csv', 'MacaqueFaces_ImageInfo.csv'),
        ]
        for url, file in downloads:
            utils.download_url(url, file)

    @classmethod
    def _extract(cls):
        utils.extract_archive('MacaqueFaces.zip', delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the dataset
        data = pd.read_csv(os.path.join(self.root, 'MacaqueFaces_ImageInfo.csv'))
        date_taken = [datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%Y-%m-%d') for date in data['DateTaken']]
        
        # Finalize the dataframe
        data['Path'] = data['Path'].str.replace('/', os.path.sep)
        #display(data['Path'].str.strip(os.path.sep))
        #display(os.path.sep)
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),            
            'path': 'MacaqueFaces' + os.path.sep + data['Path'].str.strip(os.path.sep) + os.path.sep + data['FileName'],
            'identity': data['ID'],
            'date': pd.Series(date_taken),
            'category': data['Category']
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class MPDD(DatasetFactory):
    metadata = metadata['MPDD']
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/v5j6m8dzhv-1.zip'
    archive = 'MPDD.zip'
    
    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        utils.extract_archive(os.path.join('Multi-pose dog dataset', 'MPDD.zip'), delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        data = utils.find_images(self.root)
        identity = data['file'].apply(lambda x: int(x.split('_')[0]))

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': data['file'],
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class NDD20(DatasetFactory):
    metadata = metadata['NDD20']
    url = 'https://data.ncl.ac.uk/ndownloader/files/22774175'
    archive = 'NDD20.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)    

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
                    identity = self.unknown_name
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
                    'orientation': 'above'
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
                    identity = self.unknown_name
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
                    'orientation': 'below'
                })

        # Create the dataframe from entries 
        df = pd.DataFrame(entries)
        if len(df.reg_type.unique()) != 1:
            raise(Exception('Multiple segmentation types'))

        # Finalize the dataframe
        df['id'] = range(len(df))
        df['path'] = df['orientation'].str.upper() + os.path.sep + df['file_name']
        df = df.drop(['reg_type', 'file_name'], axis=1)
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class NOAARightWhale(DatasetFactory):
    metadata = metadata['NOAARightWhale']
    archive = 'noaa-right-whale-recognition.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c noaa-right-whale-recognition --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#noaarightwhale'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        try:
            utils.extract_archive(cls.archive, delete=True)
            utils.extract_archive('imgs.zip', delete=True)
            # Move misplaced image
            shutil.move('w_7489.jpg', 'imgs')
            os.remove('w_7489.jpg.zip')
        except:
            exception_text = '''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#noaarightwhale'''
            raise Exception(exception_text)

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
            'identity': self.unknown_name,
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class NyalaData(DatasetFactory):
    metadata = metadata['NyalaData']
    url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
    archive = 'main.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        shutil.rmtree('wildlife_reidentification-main/Lion_Data_Zero')

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Extract information from the folder structure and about orientation
        identity = folders[3].astype(int)
        orientation = np.full(len(data), np.nan, dtype=object)
        orientation[['left' in filename for filename in data['file']]] = 'left'
        orientation[['right' in filename for filename in data['file']]] = 'right'

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'orientation': orientation,
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)   


class OpenCows2020(DatasetFactory):
    metadata = metadata['OpenCows2020']
    url = 'https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip'
    archive = '10m32xl88x2b61zlkkgz3fml17.zip'

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)    


class PolarBearVidID(DatasetFactory):
    metadata = metadata['PolarBearVidID']
    url = 'https://zenodo.org/records/7564529/files/PolarBearVidID.zip?download=1'
    archive = 'PolarBearVidID.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        metadata = pd.read_csv(os.path.join(self.root, 'animal_db.csv'))
        data = utils.find_images(self.root)

        # Convert numbers into animal names
        path_to_names = {}
        for _, metadata_row in metadata.iterrows():
            path_to_names[metadata_row['id']] = metadata_row['name']
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': data['path'].apply(lambda x: path_to_names[int(x)]),
            'video': data['file'].apply(lambda x: int(x[7:10]))
        })
        return self.finalize_catalogue(df)


class SealID(DatasetFactory):
    metadata = metadata['SealID']
    prefix = 'source_'
    archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
    
    @classmethod
    def _download(cls, url=None):
        if url is None:
            raise(Exception('URL must be provided for SealID.\nCheck https://wildlifedatasets.github.io/wildlife-datasets/downloads/#sealid'))
        utils.download_url(url, cls.archive)

    @classmethod
    def _extract(cls, **kwargs):
        utils.extract_archive(cls.archive, delete=True)
        utils.extract_archive(os.path.join('SealID', 'full images.zip'), delete=True)
        utils.extract_archive(os.path.join('SealID', 'patches.zip'), delete=True)
        
        # Create new folder for segmented images
        folder_new = os.getcwd() + 'Segmented'
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)
        
        # Move segmented images to new folder
        folder_move = os.path.join('patches', 'segmented')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join('full images', 'segmented_database')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join('full images', 'segmented_query')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        file_copy = os.path.join('patches', 'annotation.csv')
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))
        file_copy = os.path.join('full images', 'annotation.csv')
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))            

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class SealIDSegmented(SealID):
    prefix = 'segmented_'
    warning = '''You are trying to download or extract a segmented dataset.
        It is already included in its non-segmented version.
        Skipping.'''
    
    @classmethod
    def get_data(cls, *args, **kwargs):
        print(cls.warning)

    @classmethod
    def _download(cls, *args, **kwargs):
        print(cls.warning)

    @classmethod
    def _extract(cls, *args, **kwargs):
        print(cls.warning)


class SeaStarReID2023(DatasetFactory):
    metadata = metadata['SeaStarReID2023']
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
        species = folders[1].apply(lambda x: x[:4])
        species.replace('Anau', 'Anthenea australiae', inplace=True)
        species.replace('Asru', 'Asteria rubens', inplace=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'species': species
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class SeaTurtleID(DatasetFactory):
    metadata = metadata['SeaTurtleID']
    archive = 'seaturtleid.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/seaturtleid --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#seaturtleid'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = 'annotations.json'
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Extract dtaa from the JSON file
        create_dict = lambda i: {'id': i['id'], 'bbox': i['bbox'], 'image_id': i['image_id'], 'identity': i['identity'], 'segmentation': i['segmentation'], 'orientation': i['position']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        idx_bbox = ~df_annotation['bbox'].isnull()
        df_annotation.loc[idx_bbox,'bbox'] = df_annotation.loc[idx_bbox,'bbox'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, on='image_id')
        df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
        df = df.drop(['image_id', 'file_name'], axis=1)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])

        # Finalize the dataframe
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class SeaTurtleIDHeads(DatasetFactory):
    metadata = metadata['SeaTurtleIDHeads']
    archive = 'seaturtleidheads.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/seaturtleidheads --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#seaturtleid'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = 'annotations.json'
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Extract dtaa from the JSON file
        create_dict = lambda i: {'id': i['id'], 'image_id': i['image_id'], 'identity': i['identity'], 'orientation': i['position']}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, on='image_id')
        df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
        df = df.drop(['image_id', 'file_name'], axis=1)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])

        # Finalize the dataframe
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class SMALST(DatasetFactory):
    metadata = metadata['SMALST']
    url = 'https://drive.google.com/uc?id=1yVy4--M4CNfE5x9wUr1QBmAXEcWb6PWF'
    archive = 'zebra_training_set.zip'

    @classmethod
    def _download(cls):
        exception_text = '''Download failed. GDown quota probably reached. Download dataset manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#smalst'''
        utils.gdown_download(cls.url, cls.archive, exception_text)

    @classmethod
    def _extract(cls):
        exception_text = '''Extracting works only on Linux. Please extract it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#smalst'''
        if os.name == 'posix':
            os.system('jar xvf ' + cls.archive)
            os.remove(cls.archive)
            shutil.rmtree(os.path.join('zebra_training_set', 'annotations'))
            shutil.rmtree(os.path.join('zebra_training_set', 'texmap'))
            shutil.rmtree(os.path.join('zebra_training_set', 'uvflow'))

        else:
            raise Exception(exception_text)

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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class StripeSpotter(DatasetFactory):
    metadata = metadata['StripeSpotter']

    @classmethod
    def _download(cls):
        downloads = [
            ('https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.zip', 'data-20110718.zip'),
            ('https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z02', 'data-20110718.z02'),
            ('https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z01', 'data-20110718.z01'),
        ]
        for url, archive in downloads:
            utils.download_url(url, archive)

    @classmethod
    def _extract(cls):
        exception_text = '''Extracting works only on Linux. Please extract it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#stripespotter'''
        if os.name == 'posix':
            os.system(f"zip -s- data-20110718.zip -O data-full.zip")
            os.system(f"unzip data-full.zip")
            os.remove('data-20110718.zip')
            os.remove('data-20110718.z01')
            os.remove('data-20110718.z02')
            os.remove('data-full.zip')
        else:
            raise Exception(exception_text)       

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
        data.loc[data['animal_name'].isnull(), 'animal_name'] = self.unknown_name
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'id': utils.create_id(data['file']),
            'path':  data['path'] + os.path.sep + data['file'],
            'identity': data['animal_name'],
            'bbox': pd.Series([[int(a) for a in b.split(' ')] for b in data['roi']]),
            'orientation': data['flank'],
            'photo_quality': data['photo_quality'],
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)  


class WhaleSharkID(DatasetFactoryWildMe):
    metadata = metadata['WhaleSharkID']
    url = 'https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz'
    archive = 'whaleshark.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('whaleshark', 2020)


class WNIGiraffes(DatasetFactory):
    metadata = metadata['WNIGiraffes']
    url = "https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train_images.zip"
    archive = 'wni_giraffes_train_images.zip'
    url2 = 'https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train.zip'
    archive2 = 'wni_giraffes_train.zip'

    @classmethod
    def _download(cls):
        exception_text = '''Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#wnigiraffes'''
        raise Exception(exception_text)
        #os.system(f'azcopy cp {cls.url} {cls.archive}')
        #utils.download_url(cls.url2, cls.archive2)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        utils.extract_archive(cls.archive2, delete=True)

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
        data.rename({'id': 'image_id'}, axis=1, inplace=True)
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
    metadata = metadata['ZindiTurtleRecall']

    @classmethod
    def _download(cls):
        downloads = [
            ('https://storage.googleapis.com/dm-turtle-recall/train.csv', 'train.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/extra_images.csv', 'extra_images.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/test.csv', 'test.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/images.tar', 'images.tar'),
        ]
        for url, file in downloads:
            utils.download_url(url, file)

    @classmethod
    def _extract(cls):
        utils.extract_archive('images.tar', 'images', delete=True)

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
        data.loc[data['turtle_id'].isnull(), 'turtle_id'] = self.unknown_name
        df = pd.DataFrame({
            'id': data['image_id'],
            'path': 'images' + os.path.sep + data['image_id'] + '.JPG',
            'identity': data['turtle_id'],
            'orientation': data['image_location'].str.lower(),
            'split': data['split'],
        })
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)