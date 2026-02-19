from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_coco
from matplotlib.figure import Figure
from PIL import Image

from . import utils


class WildlifeDataset:
    """Base class for creating datasets.

    Attributes:    
      df (pd.DataFrame): A full dataframe of the data.
      summary (dict): Summary of the dataset.
      root (str): Root directory for the data.
      update_wrong_labels (bool): Whether `fix_labels` should be called.
      unknown_name (str): Name of the unknown class.
      outdated_dataset (bool): Tracks whether dataset was replaced by a new version.
      determined_by_df (bool): Specifies whether dataset is completely determined by its dataframe.
      saved_to_system_folder (bool): Specifies whether dataset is saved to system (hidden) folders.
      remove_columns (bool): Specifies whether constant columns are removed in `finalize_catalogue`.
      check_files (bool): Specifies whether files should be checks for existence in `finalize_catalogue`.
      transform (Callable): Applied transform when loading the image.
      img_load (str): Applied transform when loading the image.
      labels_string (List[str]): List of labels in strings.
      load_label (bool): Whether dataset[k] should return only image or also identity.
      factorize_label (bool): Whether labels are returned factorized (intergers) or original (possibly strings).
      col_path (str): Column name containing image paths.
      col_label (str): Column name containing individual animal names (labels).
    """

    summary = {}
    unknown_name = 'unknown'
    outdated_dataset = False
    determined_by_df = True
    saved_to_system_folder = False
    download_warning = '''You are trying to download an already downloaded dataset.
        This message may have happened to due interrupted download or extract.
        To force the download use the `force=True` keyword such as
        get_data(..., force=True) or download(..., force=True).
        '''
    download_mark_name = 'already_downloaded'
    license_file_name = 'LICENSE_link'

    def __init__(
            self, 
            root: str | None = None,
            df: pd.DataFrame | None = None,
            update_wrong_labels: bool = True,
            transform: Callable | None = None,
            img_load: str = "full",
            remove_unknown: bool = False,
            remove_columns: bool = False,
            check_files: bool = True,
            load_label: bool = False,
            factorize_label: bool = False,
            col_path: str = "path",
            col_label: str = "identity",            
            **kwargs) -> None:
        """Initializes the class.

        If `df` is specified, it copies it. Otherwise, it creates it
        by the `create_catalogue` method.

        Args:
            root (Optional[str], optional): Root directory for the data.
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.
            update_wrong_labels (bool, optional): Whether `fix_labels` should be called.
            transform (Optional[Callable], optional): Applied transform when loading the image.
            img_load (str, optional): Applied transform when loading the image.
            remove_unknown (bool, optional): Whether unknown identities should be removed.
            remove_columns (bool, optional): Whether constant columns are removed in `finalize_catalogue`.
            check_files (bool, optional): Whether files should be checks for existence in `finalize_catalogue`.
            load_label (bool, optional): Whether dataset[k] should return only image or also identity.
            factorize_label (bool, optional): Whether labels are returned factorized (intergers) or original (possibly strings).
            col_path (str, optional): Column name containing image paths.
            col_label (str, optional): Column name containing individual animal names (labels).
        """
        
        if not self.saved_to_system_folder and root is not None and not os.path.exists(root):
            raise Exception('root does not exist. You may have have mispelled it.')
        if self.outdated_dataset:
            print('This dataset is outdated. You may want to call a newer version such as %sv2.' % self.__class__.__name__)
        self.update_wrong_labels = update_wrong_labels
        self.root = root
        self.col_path = col_path
        self.col_label = col_label
        self.remove_columns = remove_columns
        self.check_files = check_files
        # If df is not provided, create it
        if df is None:
            df = self.create_catalogue(**kwargs)
        else:
            if not self.determined_by_df:
                print('This dataset is not determined by dataframe. But you construct it so.')
        assert df is not None

        if remove_unknown:
            df = df[df[self.col_label] != self.unknown_name]
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_load = img_load
        if self.img_load == "auto":
            if "segmentation" in self.df:
                self.img_load = "bbox_mask"
            elif "bbox" in self.df:
                self.img_load = "bbox"
            else:
                self.img_load = "full"
        self.load_label = load_label
        self.factorize_label = factorize_label
        self.compute_attributes()

    @property
    def labels_string(self):
        return self.df[self.col_label].astype(str).to_numpy()

    @property
    def num_classes(self):
        return self.df[self.col_label].nunique()

    @property
    def identities(self):
        return self.df[self.col_label].unique()
    
    @property
    def metadata(self):
        return self.df
        
    @metadata.setter
    def metadata(self, value):
        self.df = value	

    @contextmanager
    def temporary_attrs(self, **kwargs):
        old_values = {k: getattr(self, k) for k in kwargs}
        for k, v in kwargs.items():
            setattr(self, k, v)
        try:
            yield
        finally:
            for k, v in old_values.items():
                setattr(self, k, v)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Image.Image | tuple[Image.Image, int | str]:
        """Load an image with iloc `idx` with transforms `self.transform` and `self.img_load` applied.

        Args:
            idx (int): Index of the image.

        Returns:
            Loaded image.
        """

        img = self.get_image(idx)
        img = self.apply_segmentation(img, idx)
        if self.transform:
            img = self.transform(img)
        if self.load_label and self.factorize_label:
            return img, self.labels[idx]
        elif self.load_label:
            return img, self.df[self.col_label].iloc[idx]
        else:
            return img

    def compute_attributes(self) -> None:
        self.labels, self.labels_map = pd.factorize(self.df[self.col_label].to_numpy())

    def get_subset(self, idx: list[int] | list[bool]) -> WildlifeDataset:
        """Returns a subset of the class.

        Args:
            idx (Union[List[int], List[bool]]): Indices in the dataframe of the subset.

        Returns:
            The subset class.
        """

        dataset = deepcopy(self)
        if len(self) == len(idx):
            dataset.df = dataset.df[idx].reset_index(drop=True)
        else:
            dataset.df = dataset.df.loc[idx].reset_index(drop=True)
        dataset.compute_attributes()
        return dataset

    def get_absolute_path(self, path: str) -> str:
        if self.root:
            return os.path.join(self.root, path)
        else:
            return path

    def get_image(self, idx: int) -> Image.Image:
        """Load an image with iloc `idx`.

        Args:
            idx (int): Index of the image.

        Returns:
            Loaded image.
        """

        data = self.df.iloc[idx]
        img_path = self.get_absolute_path(data[self.col_path])
        img = self.load_image(img_path)
        return img
    
    def load_image(self, path: str) -> Image.Image:
        """Load an image with `path`.

        Args:
            path (str): Path to the image.

        Returns:
            Loaded image.
        """

        return utils.load_image(path)

    def _prepare_segmentation(self, img, segmentation):
        if isinstance(segmentation, (list, np.ndarray)):
            w, h = img.size
            rles = mask_coco.frPyObjects([segmentation], h, w)
            return mask_coco.merge(rles)

        if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), (list, np.ndarray)):
            h, w = segmentation["size"]
            return mask_coco.frPyObjects(segmentation, h, w)

        if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), str):
            return segmentation

        if isinstance(segmentation, str):
            assert self.root is not None
            m = np.asfortranarray(utils.load_image(os.path.join(self.root, segmentation)))
            if m.ndim == 3:
                m = m[:, :, 0]
            return mask_coco.encode(m)

        if not np.any(pd.isnull(segmentation)):
            raise Exception("Segmentation type not recognized")

        return segmentation

    def _apply_mask(self, img: Image.Image, segmentation, invert: bool) -> Image.Image:
        if np.any(pd.isnull(segmentation)):
            return img
        mask = mask_coco.decode(segmentation).astype(bool)
        if invert:
            mask = ~mask
        return Image.fromarray(np.asarray(img) * mask[..., np.newaxis])

    def apply_segmentation(self, img: Image.Image, idx: int) -> Image.Image:
        """Applies segmentation or bounding box when loading an image.

        Args:
            img (Image): Loaded image.
            idx (int): Index of the image.

        Returns:
            Loaded image.
        """

        data = self.df.iloc[idx]
        segmentation = None
        
        # Prepare for segmentations        
        if self.img_load in {"full_mask", "full_hide", "bbox_mask", "bbox_hide"}:
            if "segmentation" not in data:
                raise ValueError(f"{self.img_load} selected but no segmentation found.")
            segmentation = self._prepare_segmentation(img, data["segmentation"])
                
        # Load full image as it is.
        if self.img_load == "full":
            return img
        
        # Mask background using segmentation mask.
        if self.img_load == "full_mask":
            return self._apply_mask(img, segmentation, False)

        # Hide object using segmentation mask
        if self.img_load == "full_hide":
            return self._apply_mask(img, segmentation, True)

        # Crop to bounding box
        if self.img_load == "bbox":
            if "bbox" not in data:
                raise ValueError(f"{self.img_load} selected but no bbox found.")
            raw = data["bbox"]
            bbox = json.loads(raw) if isinstance(raw, str) else raw
            if not np.any(pd.isnull(bbox)):
                img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            return img

        # Mask background using segmentation mask and crop to bounding box.
        if self.img_load == "bbox_mask":
            img = self._apply_mask(img, segmentation, False)
            return utils.crop_black(img)
        
        # Hide object using segmentation mask and crop to bounding box.
        if self.img_load == "bbox_hide":
            img = self._apply_mask(img, segmentation, True)
            return utils.crop_black(img)

        # Crop black background around images
        if self.img_load == "crop_black":
            return utils.crop_black(img)
        
        # Crop white background around images
        if self.img_load == "crop_white":
            return utils.crop_white(img)
        
        raise ValueError(f"Invalid img_load argument: {self.img_load}")

    @classmethod
    def get_data(
            cls,
            root: str,
            force: bool = False,
            **kwargs
            ) -> None:
        """Downloads and extracts the data. Wrapper around `cls._download` and `cls._extract.`

        Args:
            root (str): Where the data should be stored.
            force (bool, optional): It the root exists, whether it should be overwritten.
        """

        dataset_name = cls.__name__
        mark_file_name = os.path.join(root, cls.download_mark_name)
        
        already_downloaded = os.path.exists(mark_file_name)
        if not cls.saved_to_system_folder and already_downloaded and not force:
            print('DATASET %s: DOWNLOADING STARTED.' % dataset_name)
            print(cls.download_warning)
        else:
            print('DATASET %s: DOWNLOADING STARTED.' % dataset_name)
            cls.download(root, force=force, **kwargs)
            print('DATASET %s: EXTRACTING STARTED.' % dataset_name)
            cls.extract(root,  **kwargs)
            print('DATASET %s: FINISHED.\n' % dataset_name)

    @classmethod
    def download(
            cls,
            root: str,
            force: bool = False,
            **kwargs
            ) -> None:
        """Downloads the data. Wrapper around `cls._download`.

        Args:
            root (str): Where the data should be stored.
            force (bool, optional): It the root exists, whether it should be overwritten.
        """
        
        dataset_name = cls.__name__
        mark_file_name = os.path.join(root, cls.download_mark_name)
        
        already_downloaded = os.path.exists(mark_file_name)
        if cls.saved_to_system_folder:
            cls._download(**kwargs)
        elif already_downloaded and not force:
            print('DATASET %s: DOWNLOADING STARTED.' % dataset_name)            
            print(cls.download_warning)
        else:
            if os.path.exists(mark_file_name):
                os.remove(mark_file_name)
            with utils.data_directory(root):
                cls._download(**kwargs)
            open(mark_file_name, 'a').close()
            if hasattr(cls, 'summary') and 'licenses_url' in cls.summary and isinstance(cls.summary, str):
                with open(os.path.join(root, cls.license_file_name), 'w') as file:
                    file.write(cls.summary['licenses_url'])
        
    @classmethod    
    def extract(cls, root: str, **kwargs) -> None:
        """Extract the data. Wrapper around `cls._extract`.

        Args:
            root (str): Where the data should be stored.
        """

        if cls.saved_to_system_folder:
            cls._extract(**kwargs)
        else:
            with utils.data_directory(root):
                cls._extract(**kwargs)
            mark_file_name = os.path.join(root, cls.download_mark_name)
            open(mark_file_name, 'a').close()
    
    @classmethod
    def display_name(cls) -> str:
        """Returns name of the dataset without the v2 ending.

        Returns:
            Name of the dataset.
        """

        cls_parent = cls.__bases__[-1]
        while cls_parent != object and cls_parent.outdated_dataset:
            cls = cls_parent
            cls_parent = cls.__bases__[-1]
        return cls.__name__

    @classmethod
    def _download(cls):
        """Downloads the dataset. Needs to be implemented by subclasses.

        Raises:
            NotImplementedError: Needs to be implemented by subclasses.
        """

        raise NotImplementedError('Needs to be implemented by subclasses.')

    @classmethod
    def _extract(cls):
        """Extracts the dataset. Needs to be implemented by subclasses.

        Raises:
            NotImplementedError: Needs to be implemented by subclasses.
        """

        raise NotImplementedError('Needs to be implemented by subclasses.')

    def set_transform(self, transform):
        self.transform = transform

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
            replace_identity: list[tuple],
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
            identities_to_remove: list,
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
            replace_identity: list[tuple],
            col: str = 'identity'
            ) -> pd.DataFrame:
        """Replaces specified images with specified identities.

        It looks for a subset of image_name in df[self.col_path].
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
                if image_name in df_row[self.col_path] and old_identity == df_row[col]:
                    df.loc[index, col] = new_identity
                    n_replaced += 1
            if n_replaced == 0:
                print('File name %s with identity %s was not found.' % (image_name, str(old_identity)))
            elif n_replaced > 1:
                print('File name %s with identity %s was found multiple times.' % (image_name, str(old_identity)))
        return df

    def finalize_catalogue(
            self,
            df: pd.DataFrame | None = None,
            ) -> pd.DataFrame:
        """Reorders the dataframe and check file paths.

        Reorders the columns and removes constant columns.
        Checks if columns are in correct formats.
        Checks if ids are unique and if all files exist.

        Args:
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.

        Returns:
            A full dataframe of the data, slightly modified.
        """

        if df is None:
            df = self.df
        if self.update_wrong_labels:
            df = self.fix_labels(df)
        self.rename_column(df, 'path', self.col_path)
        self.rename_column(df, 'identity', self.col_label)
        self.check_required_columns(df)
        self.check_types_columns(df)
        df = self.reorder_df(df)
        if self.remove_columns:
            df = self.remove_constant_columns(df)
        self.check_unique_id(df)
        if self.check_files:
            self.check_files_exist(df[self.col_path])
            self.check_files_names(df[self.col_path])
            if 'segmentation' in df.columns:
                self.check_files_exist(df['segmentation'])
        return df

    def rename_column(self, df: pd.DataFrame, name_old, name_new):
        if name_old != name_new:
            if name_new in df.columns:
                raise Exception(f'Column {name_old} already present in dataframe. Cannot rename {name_old} to it.')
            else:
                return df.rename({name_old: name_new}, axis=1, inplace=True)

    def check_required_columns(self, df: pd.DataFrame | None = None) -> None:
        """Check if all required columns are present.

        Args:
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.
        """

        if df is None:
            df = self.df
        for col_name in ["image_id", self.col_label, self.col_path]:
            if col_name not in df.columns:
                raise Exception('Column %s must be in the dataframe columns.' % col_name)

    def check_types_columns(self, df: pd.DataFrame | None = None) -> None:
        """Checks if columns are in correct formats.

        The format are specified in `requirements`, which is list
        of tuples. The first value is the name of the column
        and the second value is a list of formats. The column
        must be at least one of the formats.

        Args:
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.
        """

        if df is None:
            df = self.df
        requirements = [
            ('image_id', ['int', 'str']),
            (self.col_label, ['int', 'str']),
            (self.col_path, ['str']),
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
    
    def check_types_column(self, col: pd.Series, col_name: str, allowed_types: list[str]) -> None:
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
        raise Exception('Column %s has wrong type. Allowed types = %s' % (col_name, str(allowed_types)))

    def reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders rows and columns in the dataframe.

        Rows are sorted based on id.
        Columns are reorder based on the `default_order` list.

        Args:
            df (pd.DataFrame): A full dataframe of the data.

        Returns:
            A full dataframe of the data, slightly modified.
        """

        default_order = ['image_id', self.col_label, self.col_path, 'bbox', 'date', 'keypoints', 'orientation', 'segmentation', 'species']
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

    def remove_constant_columns(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Removes columns with a single unique value.

        Args:
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.

        Returns:
            A full dataframe of the data, slightly modified.
        """

        if df is None:
            df = self.df
        for df_name in list(df.columns):
            if df[df_name].astype("str").nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df

    def check_unique_id(self, df: pd.DataFrame | None = None) -> None:
        """Checks if values in the id column are unique.

        Args:
            df (Optional[pd.DataFrame], optional): A full dataframe of the data.
        """

        if df is None:
            df = self.df
        if len(df["image_id"].unique()) != len(df):
            raise Exception("Image ID not unique.")

    def check_files_exist(self, col: pd.Series | str | None = None) -> None:
        """Checks if paths in a given column exist.

        Args:
            col (Optional[pd.Series | str], optional): A column of a dataframe.
        """

        if col is None:
            col = self.df[self.col_path]
        elif isinstance(col, str):
            col = self.df[col]
        bad_paths = []
        for path in col:
            if isinstance(path, str) and not os.path.exists(self.get_absolute_path(path)):
                bad_paths.append(path)
        if len(bad_paths) > 0:
            print("The following non-existing images were identified.")                
            for path in bad_paths:
                print(path)
            raise Exception('Some files not found')

    def check_files_names(self, col: pd.Series | str | None = None) -> None:
        """Checks if paths contain characters which may cause issues.

        Args:
            col (Optional[pd.Series | str], optional): A column of a dataframe.
        """

        if col is None:
            col = self.df[self.col_path]
        elif isinstance(col, str):
            col = self.df[col]
        bad_names = []
        for path in col:
            if not isinstance(path, str):
                continue
            try:
                path.encode("iso-8859-1")
            except UnicodeEncodeError:
                bad_names.append(path)
        if len(bad_names) > 0:
            print("The following not ISO-8859-1 file names were identified.")
            for path in bad_names:
                print(path)
            raise Exception("Non ISO-8859-1 characters in path may cause problems. Please change them.")

    def plot_grid(
            self,
            n_rows: int = 5,
            n_cols: int = 8,
            offset: int = 10,
            img_min: float = 100,
            rotate: bool = True,
            keep_aspect_ratios: bool = True,
            header_cols: Sequence[str] | None = None,
            idx: Sequence[bool] | Sequence[int] | np.ndarray | None = None,
            background_color: tuple[int, int, int] = (0, 0, 0),
            keep_transform: bool = False,
            **kwargs
            ) -> Figure | None:
        """Plots a grid of size (n_rows, n_cols) with images from the dataframe.

        Args:
            n_rows (int, optional): The number of rows in the grid.
            n_cols (int, optional): The number of columns in the grid.
            offset (int, optional): The offset between images.
            img_min (float, optional): The minimal size of the plotted images.
            rotate (bool, optional): Rotates the images to have the same orientation.
            keep_aspect_ratios (bool, optional): Whether aspect ratios are kept for images.
            header_cols (Optional[Sequence[str]], optional): List of headers for each column.
            idx (Optional[Union[Sequence[bool], Sequence[int]]], optional): List of indices to plot. None plots random images. Index -1 plots an empty image.
            background_color (Tuple[int, int, int], optional): Background color of the grid.
            keep_transform (bool, optional): Whether `self.transform` is applied.
        """

        if len(self.df) == 0:
            return None
        
        # Select indices of images to be plotted
        if idx is None:
            n = min(len(self.df), n_rows*n_cols)
            idx = np.random.permutation(len(self.df))[:n]
        else:
            if isinstance(idx, pd.Series):
                idx = idx.to_numpy()
            if isinstance(idx[0], (bool, np.bool_)):
                idx = np.where(idx)[0]
            n = min(np.array(idx).size, n_rows*n_cols)
            idx = np.asarray(idx).flatten()[:n]

        # Load images and compute their ratio
        ratios = []
        ims = []
        for k in idx:
            if k is not None and k >= 0:
                im = self.get_image(k)
                im = self.apply_segmentation(im, k)
                if keep_transform and self.transform:
                    im = self.transform(im)
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
                raise Exception("Length of header_cols must be the same as n_cols.")
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
                    if keep_aspect_ratios:
                        w, h = im.size
                        c = min(img_w / w, img_h / h)
                        im = im.resize((int(c*w), int(c*h)))
                    else:
                        im = im.resize((img_w,img_h))
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

# Alias for WildlifeDataset
class DatasetFactory(WildlifeDataset):
    pass
