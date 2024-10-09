import os
import shutil
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Callable, Tuple
import json
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import string

from .. import splits
from .metadata import metadata
from . import utils


class DatasetFactory():
    """Base class for creating datasets.

    Attributes:    
      df (pd.DataFrame): A full dataframe of the data.
      metadata (dict): Metadata of the dataset.
      root (str): Root directory for the data.
      update_wrong_labels(bool): Whether `fix_labels` should be called.
      unknown_name (str): Name of the unknown class.
      outdated_dataset (bool): Tracks whether dataset was replaced by a new version.
    """

    unknown_name = 'unknown'
    outdated_dataset = False
    download_warning = '''You are trying to download an already downloaded dataset.
        This message may have happened to due interrupted download or extract.
        To force the download use the `force=True` keyword such as
        get_data(..., force=True) or download(..., force=True).
        '''
    download_mark_name = 'already_downloaded'
    license_file_name = 'LICENSE_link'

    def __init__(
            self, 
            root: str,
            df: Optional[pd.DataFrame] = None,
            update_wrong_labels: bool = True,
            **kwargs) -> None:
        """Initializes the class.

        If `df` is specified, it copies it. Otherwise, it creates it
        by the `create_catalogue` method.

        Args:
            root (str): Root directory for the data.
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.
            update_wrong_labels (bool, optional): Whether `fix_labels` should be called.
        """

        if not os.path.exists(root):
            raise Exception('root does not exist. You may have have mispelled it.')
        if self.outdated_dataset:
            print('This dataset is outdated. You may want to call a newer version such as %sv2.' % self.__class__.__name__)
        self.root = root
        self.update_wrong_labels = update_wrong_labels
        if df is None:
            self.df = self.create_catalogue(**kwargs)
        else:
            self.df = df.copy()

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
            if hasattr(cls, 'metadata') and 'licenses_url' in cls.metadata:
                with open(os.path.join(root, cls.license_file_name), 'w') as file:
                    file.write(cls.metadata['licenses_url'])
        
    @classmethod    
    def extract(cls, root, **kwargs):
        with utils.data_directory(root):
            cls._extract(**kwargs)
        mark_file_name = os.path.join(root, cls.download_mark_name)
        open(mark_file_name, 'a').close()
    
    @classmethod
    def display_name(cls) -> str:
        cls_parent = cls.__bases__[0]
        while cls_parent != object and cls_parent.outdated_dataset:
            cls = cls_parent
            cls_parent = cls.__bases__[0]            
        return cls.__name__

    def create_catalogue(self):
        """Creates the dataframe.

        Raises:
            NotImplementedError: Needs to be implemented by subclasses.
        """

        raise NotImplementedError('Needs to be implemented by subclasses.')
    
    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixes labels in dataframe.
        
        Automatically called in `finalize_catalogue`.                
        """

        return df

    def fix_labels_replace_identity(
            self,
            df: pd.DataFrame,
            replace_identity: List[Tuple],
            col: str = 'identity'
            ) -> pd.DataFrame:
        """Replaces all instances of identities.

        Args:
            df (pd.DataFrame): A full dataframe of the data.
            replace_identity (List[Tuple]): List of (old_identity, new_identity)
            col (str, optional): Column to replace in.

        Returns:
            A full dataframe of the data.
        """
        for old_identity, new_identity in replace_identity:
            df[col] = df[col].replace({old_identity: new_identity})
        return df

    def fix_labels_remove_identity(
            self,
            df: pd.DataFrame,
            identities_to_remove: List,
            col: str = 'identity'
            ) -> pd.DataFrame:
        """Removes all instances of identities.

        Args:
            df (pd.DataFrame): A full dataframe of the data.
            identities_to_remove (List): List of identities to remove.
            col (str, optional): Column to remove from.

        Returns:
            A full dataframe of the data.
        """
        idx_remove = [identity in identities_to_remove for identity in df[col]]
        return df[~np.array(idx_remove)]

    def fix_labels_replace_images(
            self,
            df: pd.DataFrame,
            replace_identity: List[Tuple],
            col: str = 'identity'
            ) -> pd.DataFrame:
        """Replaces specified images with specified identities.

        It looks for a subset of image_name in df['path'].
        It may cause problems with `os.path.sep`.

        Args:
            df (pd.DataFrame): A full dataframe of the data.
            replace_identity (List[Tuple]): List of (image_name, old_identity, new_identity).
            col (str, optional): Column to replace in.

        Returns:
            A full dataframe of the data.
        """
        for image_name, old_identity, new_identity in replace_identity:
            n_replaced = 0
            for index, df_row in df.iterrows():
                # Check that there is a image with the required name and identity 
                if image_name in df_row['path'] and old_identity == df_row[col]:
                    df.loc[index, col] = new_identity
                    n_replaced += 1
            if n_replaced == 0:
                print('File name %s with identity %s was not found.' % (image_name, str(old_identity)))
            elif n_replaced > 1:
                print('File name %s with identity %s was found multiple times.' % (image_name, str(old_identity)))
        return df

    def plot_grid(
            self,
            n_rows: int = 5,
            n_cols: int = 8,
            offset: float = 10,
            img_min: float = 100,
            rotate: bool = True,
            header_cols: Optional[List[str]] = None,
            idx: Optional[Union[List[bool],List[int]]] = None,
            loader: Optional[Callable] = None,
            background_color: Tuple[int] = (0, 0, 0),
            **kwargs
            ) -> None:
        """Plots a grid of size (n_rows, n_cols) with images from the dataframe.

        Args:
            df (pd.DataFrame): Dataframe with column `path` (relative path).
            root (str): Root folder where the images are stored. 
            n_rows (int, optional): The number of rows in the grid.
            n_cols (int, optional): The number of columns in the grid.
            offset (float, optional): The offset between images.
            img_min (float, optional): The minimal size of the plotted images.
            rotate (bool, optional): Rotates the images to have the same orientation.
            header_cols (Optional[List[str]], optional): List of headers for each column.
            idx (Optional[Union[List[bool],List[int]]], optional): List of indices to plot. None plots random images. Index -1 plots an empty image.
            loader (Optional[Callable], optional): Loader of images. Useful for including transforms.
            background_color (Tuple[int], optional): Background color of the grid.
        """

        if len(self.df) == 0:
            return None
        
        # Select indices of images to be plotted
        if idx is None:
            n = min(len(self.df), n_rows*n_cols)
            idx = np.random.permutation(len(self.df))[:n]
        else:
            if isinstance(idx, pd.Series):
                idx = idx.values
            if isinstance(idx[0], (bool, np.bool_)):
                idx = np.where(idx)[0]
            n = min(np.array(idx).size, n_rows*n_cols)
            idx = np.matrix.flatten(np.array(idx))[:n]

        # Load images and compute their ratio
        ratios = []
        ims = []
        for k in idx:
            if k >= 0:
                # Load the image with index k
                if loader is None:
                    file_path = os.path.join(self.root, self.df.iloc[k]['path'])
                    im = utils.get_image(file_path)
                else:
                    im = loader(k)
                ims.append(im)
                ratios.append(im.size[0] / im.size[1])
            else:
                # Load a black image
                ims.append(Image.fromarray(np.zeros((2, 2), dtype = "uint8")))

        # Safeguard when all indices are -1
        if len(ratios) == 0:
            return None
        
        # Get the size of the images after being resized
        ratio = np.median(ratios)
        if ratio > 1:    
            img_w, img_h = int(img_min*ratio), int(img_min)
        else:
            img_w, img_h = int(img_min), int(img_min/ratio)

        # Compute height offset if headers are present
        if header_cols is not None:
            offset_h = 30
            if len(header_cols) != n_cols:
                raise(Exception("Length of header_cols must be the same as n_cols."))
        else:
            offset_h = 0

        # Create an empty image grid
        im_grid = Image.new('RGB', (n_cols*img_w + (n_cols-1)*offset, offset_h + n_rows*img_h + (n_rows-1)*offset), background_color)

        # Fill the grid image by image
        pos_y = offset_h
        for i in range(n_rows):
            row_h = 0
            for j in range(n_cols):
                k = (n_cols)*i + j
                if k < n:
                    # Possibly rotate the image
                    im = ims[k]
                    if rotate and ((ratio > 1 and im.size[0] < im.size[1]) or (ratio < 1 and im.size[0] > im.size[1])):
                        im = im.transpose(Image.Transpose.ROTATE_90)

                    # Rescale the image
                    im.thumbnail((img_w,img_h))
                    row_h = max(row_h, im.size[1])

                    # Place the image on the grid
                    pos_x = j*img_w + j*offset
                    im_grid.paste(im, (pos_x,pos_y))
            if row_h > 0:
                pos_y += row_h + offset
        im_grid = im_grid.crop((0, 0, im_grid.size[0], pos_y-offset))
 
        # Plot the image and add column headers if present
        fig = plt.figure()
        fig.patch.set_visible(False)
        ax = fig.add_subplot(111)
        plt.axis('off')
        plt.imshow(im_grid)
        if header_cols is not None:
            color = kwargs.pop('color', 'white')
            ha = kwargs.pop('ha', 'center')
            va = kwargs.pop('va', 'center')
            for i, header in enumerate(header_cols):
                pos_x = (i+0.5)*img_w + i*offset
                pos_y = offset_h/2
                plt.text(pos_x, pos_y, str(header), color=color, ha=ha, va=va, **kwargs)
        return fig

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

        if self.update_wrong_labels:
            df = self.fix_labels(df)
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
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class AAUZebraFish(DatasetFactory):
    metadata = metadata['AAUZebraFish']
    archive = 'aau-zebrafish-reid.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d aalborguniversity/aau-zebrafish-reid"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#aauzebrafish'''
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
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/test/atrw_detection_test.tar.gz', 'atrw_detection_test.tar.gz'),

        # Re-ID dataset
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/train/atrw_reid_train.tar.gz', 'atrw_reid_train.tar.gz'),
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/train/atrw_anno_reid_train.tar.gz', 'atrw_anno_reid_train.tar.gz'),
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/test/atrw_reid_test.tar.gz', 'atrw_reid_test.tar.gz'),
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/test/atrw_anno_reid_test.tar.gz', 'atrw_anno_reid_test.tar.gz'),
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
        ids['original_split'] = 'train'

        # Load keypoints for the reid_train part of the dataset
        with open(os.path.join(self.root, 'atrw_anno_reid_train', 'reid_keypoints_train.json')) as file:
            keypoints = json.load(file)
        df_keypoints = {
            'path': pd.Series(keypoints.keys()),
            'keypoints': pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        # Merge information for the reid_train part of the dataset
        df_train = pd.merge(ids, data, on='path', how='left')
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
        ids['original_split'] = 'test'
        ids = pd.merge(ids, identity, left_on='id', right_on='imgid', how='left')
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
        df_test1 = pd.merge(ids, data, on='path', how='left')
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
        df_test2 = pd.merge(ids, entries, on='imgid', how='left')
        df_test2['path'] = 'atrw_detection_test' + os.path.sep + 'test' + os.path.sep + df_test2['file']
        df_test2['id'] = df_test2['imgid'].astype(str) + '_' + df_test2['identity'].astype(str)
        df_test2['original_split'] = 'test'
        df_test2 = df_test2.drop(['file', 'imgid'], axis=1)

        # Finalize the dataframe
        df = pd.concat([df_train, df_test1, df_test2])
        df['id'] = utils.create_id(df['id'].astype(str))
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        return self.finalize_catalogue(df)


class BelugaID(DatasetFactoryWildMe):
    outdated_dataset = True
    metadata = metadata['BelugaID']
    downloads = [
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga.coco.tar.gz', 'beluga.coco.tar.gz'),
        ('https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga-id-test.zip', 'beluga-id-test.zip'),
    ]

    @classmethod
    def _download(cls):
        for url, archive in cls.downloads:
            utils.download_url(url, archive)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            archive_name = archive.split('.')[0]
            utils.extract_archive(archive, archive_name, delete=True)
    
    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme(os.path.join('beluga', 'beluga'), 2022)


class BelugaIDv2(BelugaID):
    outdated_dataset = False
    
    def create_catalogue(self) -> pd.DataFrame:
        # Use the original data
        df_train = self.create_catalogue_wildme(os.path.join('beluga', 'beluga'), 2022)
        df_train['original_split'] = 'train'

        # Get the conversion of identities between original and new data
        data_train = pd.read_csv(os.path.join(self.root, 'beluga-id-test', 'private_train_metadata.csv'))
        id_conversion = {}
        for old_id, data_train_red in data_train.groupby('original_whale_id'):
            if data_train_red['whale_id'].nunique() != 1:
                raise Exception('Conversion of old to new whale_id is not unique.')
            id_conversion[old_id] = data_train_red['whale_id'].iloc[0]

        # Add the new data
        data_test = pd.read_csv(os.path.join(self.root, 'beluga-id-test', 'private_test_metadata.csv'))
        df_test = pd.DataFrame({
            'path': data_test['path'].apply(lambda x: os.path.join('beluga-id-test', 'code-execution', 'images', os.path.split(x)[-1])),
            'image_id': range(len(data_train), len(data_train)+len(data_test)),
            'identity': data_test['original_whale_id'].apply(lambda x: id_conversion.get(x, 'unknown')),
            'date': data_test['timestamp'],
            'orientation': data_test['viewpoint'].replace({'top': 'up'}),
            'original_split': 'test'
        })
        
        # Finalize the dataframe
        df = pd.concat((df_train, df_test)).reset_index(drop=True)
        return self.finalize_catalogue(df)


class BirdIndividualID(DatasetFactory):
    metadata = metadata['BirdIndividualID']
    prefix1 = 'Original_pictures'
    prefix2 = 'IndividualID'
    url = 'https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA'
    archive = 'ferreira_et_al_2020.zip'

    @classmethod
    def _download(cls):
        exception_text = '''Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#birdindividualid'''
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
        split = folders[1].replace({'Test_datasets': 'test', 'Test': 'test', 'Train': 'train', 'Val': 'val'})
        identity = folders[2]
        species = folders[0]

        # Finalize the dataframe
        df1 = pd.DataFrame({    
            'image_id': utils.create_id(split + data['file']),
            'path': self.prefix1 + os.path.sep + self.prefix2 + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'species': species,
            'original_split': split,
        })

        # Add images without labels
        path = os.path.join(self.root, self.prefix1, 'New_birds')
        data = utils.find_images(path)
        species = data['path']

        # Finalize the dataframe
        df2 = pd.DataFrame({    
            'image_id': utils.create_id(data['file']),
            'path': self.prefix1 + os.path.sep + 'New_birds' + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': self.unknown_name,
            'species': species,
            'original_split': np.nan,
        })
        df = pd.concat([df1, df2])
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


class CatIndividualImages(DatasetFactory):
    metadata = metadata['CatIndividualImages']
    archive = 'cat-individuals.zip'
    
    @classmethod
    def _download(cls):
        command = f"datasets download -d timost1234/cat-individuals --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#catindividualimages'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        # Remove 85 duplicate images
        idx = folders[2].isnull()
        data = data[idx]
        folders = folders[idx]

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)


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
        # Load information about the dataset
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CTai',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_ctai.txt'), header=None, sep=' ')
        
        # Extract keypoints from the information
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })
        return self.finalize_catalogue(df)

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong identities
        replace_identity = [
            ('Adult', self.unknown_name),
            ('Akouba', 'Akrouba'),
            ('Freddy', 'Fredy'),
            ('Ibrahiim', 'Ibrahim'),
            ('Liliou', 'Lilou'),
            ('Wapii', 'Wapi'),
            ('Woodstiock', 'Woodstock')
        ]
        return self.fix_labels_replace_identity(df, replace_identity)


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
            'image_id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })
        return self.finalize_catalogue(df)


class CowDataset(DatasetFactory):
    metadata = metadata['CowDataset']
    url = 'https://figshare.com/ndownloader/files/31210192'
    archive = 'cow-dataset.zip'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
        # Rename the folder with non-ASCII characters
        dirs = [x for x in os.listdir() if os.path.isdir(x)]
        if len(dirs) != 1:
            raise Exception('There should be only one directory after extracting the file.')
        os.rename(dirs[0], 'images')
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].str.strip('cow_').astype(int),
        })
        return self.finalize_catalogue(df)


class Cows2021(DatasetFactory):
    outdated_dataset = True
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
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[4].astype(int),
        })
        df['date'] = df['path'].apply(lambda x: self.extract_date(x))
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


class Cows2021v2(Cows2021):
    outdated_dataset = False

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong identities and images
        replace_identity1 = [
            (164, 148),
            (105, 29)
        ]
        replace_identity2 = [
            ('image_0001226_2020-02-11_12-43-7_roi_001.jpg', 137, 107)
        ]
        df = self.fix_labels_replace_identity(df, replace_identity1)
        return self.fix_labels_replace_images(df, replace_identity2)
        

class DogFaceNet(DatasetFactory):
    metadata = metadata['DogFaceNet']
    url = 'https://github.com/GuillaumeMougeot/DogFaceNet/releases/download/dataset/DogFaceNet_Dataset_224_1.zip'
    url_split = [
        'https://github.com/GuillaumeMougeot/DogFaceNet/releases/download/dataset/classes_train.txt',
        'https://github.com/GuillaumeMougeot/DogFaceNet/releases/download/dataset/classes_test.txt'        
    ]
    archive = 'DogFaceNet_Dataset_224_1.zip'
    archive_split = ['classes_train.txt', 'classes_test.txt']

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)
        for url, archive in zip(cls.url_split, cls.archive_split):
            utils.download_url(url, archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)
    
    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Load classes
        classes_train = pd.read_csv(os.path.join(self.root, 'classes_train.txt'), header=None)
        classes_train = classes_train[0].unique()
        classes_test = pd.read_csv(os.path.join(self.root, 'classes_test.txt'), header=None)
        classes_test = classes_test[0].unique()
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'original_split': folders[1].astype(int).apply(lambda x: utils.get_split(x, classes_train, classes_test))
        })
        return self.finalize_catalogue(df)


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
    outdated_dataset = True
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
            'image_id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)


class FriesianCattle2015v2(FriesianCattle2015):
    outdated_dataset = False

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove all identities in training as they are duplicates
        idx_remove = ['Cows-training' in path for path in df.path]
        df = df[~np.array(idx_remove)]

        # Remove specified individuals as they are duplicates
        identities_to_remove = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 37]
        return self.fix_labels_remove_identity(df, identities_to_remove)


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
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)


class GiraffeZebraID(DatasetFactoryWildMe):
    metadata = metadata['GiraffeZebraID']
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/gzgc.coco.tar.gz'
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
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#giraffes'''
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
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[n_folders],
        })
        return self.finalize_catalogue(df)


class HappyWhale(DatasetFactory):
    metadata = metadata['HappyWhale']
    archive = 'happy-whale-and-dolphin.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c happy-whale-and-dolphin --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#happywhale'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        try:
            utils.extract_archive(cls.archive, delete=True)
        except:
            exception_text = '''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#happywhale'''
            raise Exception(exception_text)
    
    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = pd.DataFrame({
            'image_id': data['image'].str.split('.', expand=True)[0],
            'path': 'train_images' + os.path.sep + data['image'],
            'identity': data['individual_id'],
            'species': data['species'],
            'original_split': 'train'
            })

        test_files = utils.find_images(os.path.join(self.root, 'test_images'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        # Load the testing data
        df2 = pd.DataFrame({
            'image_id': test_files.str.split('.', expand=True)[0],
            'path': 'test_images' + os.path.sep + test_files,
            'identity': self.unknown_name,
            'species': np.nan,
            'original_split': 'test'
            })
        
        # Finalize the dataframe        
        df = pd.concat([df1, df2])
        return self.finalize_catalogue(df)

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace the wrong species names            
        replace_identity = [
            ('bottlenose_dolpin', 'bottlenose_dolphin'),
            ('kiler_whale', 'killer_whale'),
        ]
        return self.fix_labels_replace_identity(df, replace_identity, col='species')


class HumpbackWhaleID(DatasetFactory):
    metadata = metadata['HumpbackWhaleID']
    archive = 'humpback-whale-identification.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c humpback-whale-identification --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#humpbackwhale'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        try:
            utils.extract_archive(cls.archive, delete=True)
        except:
            exception_text = '''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#humpbackwhale'''
            raise Exception(exception_text)

    def create_catalogue(self) -> pd.DataFrame:
        # Load the training data
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data.loc[data['Id'] == 'new_whale', 'Id'] = self.unknown_name
        df1 = pd.DataFrame({
            'image_id': data['Image'].str.split('.', expand=True)[0],
            'path': 'train' + os.path.sep + data['Image'],
            'identity': data['Id'],
            'original_split': 'train'
            })
        
        # Find all testing images
        test_files = utils.find_images(os.path.join(self.root, 'test'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        # Create the testing dataframe
        df2 = pd.DataFrame({
            'image_id': test_files.str.split('.', expand=True)[0],
            'path': 'test' + os.path.sep + test_files,
            'identity': self.unknown_name,
            'original_split': 'test'
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])
        return self.finalize_catalogue(df)


class HyenaID2022(DatasetFactoryWildMe):
    metadata = metadata['HyenaID2022']
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/hyena.coco.tar.gz'
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
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#ipanda50'''
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
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'keypoints': keypoints
            })

        # Remove non-ASCII characters while keeping backwards compatibility
        file_name = os.path.join(self.root, 'changes.csv')
        if os.path.exists(file_name):
            # Files were already renamed, change image_id to keep backward compability
            df_changes = pd.read_csv(file_name)
            id_changes = {}
            for _, df_row in df_changes.iterrows():
                id_changes[df_row['id_new']] = df_row['id_old']
            df.replace(id_changes, inplace=True)
        else:
            # Rename files, keep original image_id, create list of changes
            ids_old = []
            ids_new = []
            for _, df_row in df.iterrows():
                path_new = ''.join(c for c in df_row['path'] if c in string.printable)
                # Check if there are non-ASCII characters
                if path_new != df_row['path']:
                    # Rename files and df
                    os.rename(os.path.join(self.root, df_row['path']), os.path.join(self.root, path_new))
                    df_row['path'] = path_new
                    # Save changes in image_id
                    ids_old.append(df_row['image_id'])
                    ids_new.append(utils.create_id(pd.Series(os.path.split(df_row['path'])[-1])).iloc[0])
            if len(df) != df['path'].nunique():
                raise(Exception("Non-unique names. Something went wrong when renaming."))
            pd.DataFrame({'id_old': ids_old, 'id_new': ids_new}).to_csv(file_name)

        # Finalize the dataframe        
        return self.finalize_catalogue(df)


class LeopardID2022(DatasetFactoryWildMe):
    metadata = metadata['LeopardID2022']
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/leopard.coco.tar.gz'
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
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[3],
        })
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
        df = pd.DataFrame({
            'image_id': pd.Series(range(len(data))),            
            'path': 'MacaqueFaces' + os.path.sep + data['Path'].str.strip(os.path.sep) + os.path.sep + data['FileName'],
            'identity': data['ID'],
            'date': pd.Series(date_taken),
            'category': data['Category']
        })
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
        folders = data['path'].str.split(os.path.sep, expand=True)
        identity = data['file'].apply(lambda x: int(x.split('_')[0]))
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'],
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'original_split': folders[2]
        })
        return self.finalize_catalogue(df)


class NDD20(DatasetFactory):
    outdated_dataset = True
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
        df['image_id'] = range(len(df))
        df['path'] = df['orientation'].str.upper() + os.path.sep + df['file_name']
        df = df.drop(['reg_type', 'file_name'], axis=1)
        return self.finalize_catalogue(df)


class NDD20v2(NDD20):
    outdated_dataset = False

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        for i, df_row in df.iterrows():
            # Rewrite wrong segmentations. There is no dolphin -> should be deleted.
            # But that would break compability and the identity is unknown anyway.
            if len(df_row['segmentation']) == 4:
                img = utils.get_image(os.path.join(self.root, df_row['path']))
                w, h = img.size
                df.at[i, 'segmentation'] = np.array(utils.bbox_segmentation([0, 0, w, h]))
        return df


class NOAARightWhale(DatasetFactory):
    metadata = metadata['NOAARightWhale']
    archive = 'noaa-right-whale-recognition.zip'

    @classmethod
    def _download(cls):
        command = f"competitions download -c noaa-right-whale-recognition --force"
        exception_text = '''Kaggle terms must be agreed with.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#noaarightwhale'''
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
                Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#noaarightwhale'''
            raise Exception(exception_text)

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the training dataset
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = pd.DataFrame({
            'image_id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': data['whaleID'],
            'original_split': 'train'
            })

        # Load information about the testing dataset
        data = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'))
        df2 = pd.DataFrame({
            'image_id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': self.unknown_name,
            'original_split': 'test'
            })
        
        # Finalize the dataframe
        df = pd.concat([df1, df2])
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
        orientation[data['file'].str.contains('left')] = 'left'
        orientation[data['file'].str.contains('right')] = 'right'

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'orientation': orientation,
            'original_split': folders[2]
        })
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
            'image_id': utils.create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'original_split': split
        })
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

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'video': data['file'].str[7:10].astype(int),
            'id': data['path'].astype(int)
        })
        df = pd.merge(df, metadata, on='id', how='left')
        df.rename({'name': 'identity', 'sex': 'gender'}, axis=1, inplace=True)
        df = df.drop(['id', 'zoo', 'tracklets'], axis=1)
        return self.finalize_catalogue(df)


class GreenSeaTurtles(DatasetFactory):
    # TODO: add metadata
    # TODO: change the names everywhere
    # TODO: fix documentation everywhere
    archive = 'sarahzelvy.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/sarahzelvy --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#greenseaturtles'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        file_name = os.path.join(self.root, 'annotations.csv')
        data = pd.read_csv(file_name)

        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': 'data' + os.path.sep + data['image_name'],
            'identity': data['identity'],
            'orientation': data['orientation'],
            'daytime': data['daytime']
        })
        if 'bbox_x' in data.columns:
            df['bbox'] = data[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values.tolist()
        return self.finalize_catalogue(df)


class SealID(DatasetFactory):
    metadata = metadata['SealID']
    prefix = 'source_'
    archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
    
    @classmethod
    def _download(cls, url=None):
        if url is None:
            raise(Exception('URL must be provided for SealID.\nCheck https://wildlifedatasets.github.io/wildlife-datasets/preprocessing/#sealid'))
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
            'image_id': data['file'].str.split('.', expand=True)[0],
            'path': 'full images' + os.path.sep + self.prefix + data['reid_split'] + os.path.sep + data['file'],
            'identity': data['class_id'].astype(int),
            'original_split': data['segmentation_split'].replace({'training': 'train', 'testing': 'test', 'validation': 'val'}),
            'original_split_reid': data['reid_split'],
        })
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
        species = folders[1].str[:4].replace({'Anau': 'Anthenea australiae', 'Asru': 'Asteria rubens'})

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'species': species
        })
        return self.finalize_catalogue(df)


class SeaTurtleID2022(DatasetFactory):
    metadata = metadata['SeaTurtleID2022']
    archive = 'seaturtleid2022.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/seaturtleid2022 --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#seaturtleid'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load annotations JSON file
        path_json = os.path.join('turtles-data', 'data', 'annotations.json')
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)
        path_csv = os.path.join('turtles-data', 'data', 'metadata_splits.csv')
        with open(os.path.join(self.root, path_csv)) as file:
            df_images = pd.read_csv(file)

        # Extract data from the JSON file
        create_dict = lambda i: {
            'id': i['id'],
            'bbox': i['bbox'],
            'image_id': i['image_id'],
            'segmentation': i['segmentation'],
            'orientation': i['attributes']['orientation'] if 'orientation' in i['attributes'] else np.nan
        }
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations'] if i['category_id'] == 3])
        idx_bbox = ~df_annotation['bbox'].isnull()
        df_annotation.loc[idx_bbox,'bbox'] = df_annotation.loc[idx_bbox,'bbox'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df_images.rename({'id': 'image_id'}, axis=1, inplace=True)

        # Merge the information from the JSON file
        df = pd.merge(df_images, df_annotation, on='image_id', how='outer')
        df['path'] = 'turtles-data' + os.path.sep + 'data' + os.path.sep + df['file_name'].str.replace('/', os.path.sep)
        df = df.drop(['id', 'file_name', 'timestamp', 'width', 'height', 'year', 'split_closed_random', 'split_open'], axis=1)
        df.rename({'split_closed': 'original_split'}, axis=1, inplace=True)
        df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])

        return self.finalize_catalogue(df)


class SeaTurtleIDHeads(DatasetFactory):
    metadata = metadata['SeaTurtleIDHeads']
    archive = 'seaturtleidheads.zip'

    @classmethod
    def _download(cls):
        command = f"datasets download -d wildlifedatasets/seaturtleidheads --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#seaturtleid'''
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
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#smalst'''
        utils.gdown_download(cls.url, cls.archive, exception_text)

    @classmethod
    def _extract(cls):
        exception_text = '''Extracting works only on Linux. Please extract it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#smalst'''
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
        data['image_id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        data['path'] = 'zebra_training_set' + os.path.sep + 'images' + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        # Find all masks in root
        masks = utils.find_images(os.path.join(self.root, 'zebra_training_set', 'bgsub'))
        
        # Extract information about the images
        path = masks['file'].str.strip('zebra_')
        masks['image_id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        masks['segmentation'] = 'zebra_training_set' + os.path.sep + 'bgsub' + os.path.sep + masks['file']
        masks = masks.drop(['path', 'file'], axis=1)

        # Finalize the dataframe
        df = pd.merge(data, masks, on='image_id', how='left')
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
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#stripespotter'''
        if os.name == 'posix':
            os.system(f"zip -s- data-20110718.zip -O data-full.zip")
            if not os.path.exists('data-full.zip'):
                raise Exception('Download or extraction failed. Check if zip is installed.')
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
        data = data[data['file'].str.startswith('img')]
        
        # Load additional information
        data_aux = pd.read_csv(os.path.join(self.root, 'data', 'SightingData.csv'))
        data = pd.merge(data, data_aux, how='left', left_on='index', right_on='#imgindex')
        data.loc[data['animal_name'].isnull(), 'animal_name'] = self.unknown_name
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path':  data['path'] + os.path.sep + data['file'],
            'identity': data['animal_name'],
            'bbox': pd.Series([[int(a) for a in b.split(' ')] for b in data['roi']]),
            'orientation': data['flank'],
            'photo_quality': data['photo_quality'],
            'date': data['sighting_date']
        })
        return self.finalize_catalogue(df)  


class WhaleSharkID(DatasetFactoryWildMe):
    metadata = metadata['WhaleSharkID']
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/whaleshark.coco.tar.gz'
    archive = 'whaleshark.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('whaleshark', 2020)


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
        data_extra['split'] = np.nan        

        # Finalize the dataframe
        data = pd.concat([data_train, data_test, data_extra])
        data = data.reset_index(drop=True)        
        data.loc[data['turtle_id'].isnull(), 'turtle_id'] = self.unknown_name
        df = pd.DataFrame({
            'image_id': data['image_id'],
            'path': 'images' + os.path.sep + data['image_id'] + '.JPG',
            'identity': data['turtle_id'],
            'orientation': data['image_location'].str.lower(),
            'original_split': data['split'],
        })
        return self.finalize_catalogue(df)