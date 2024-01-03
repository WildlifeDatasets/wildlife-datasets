import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import hashlib
from collections.abc import Iterable
import urllib.request
from tqdm import tqdm
import shutil
from contextlib import contextmanager

def find_images(
        root: str,
        img_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
        ) -> pd.DataFrame:
    """Finds all image files in folder and subfolders.

    Args:
        root (str): The root folder where to look for images.
        img_extensions (Tuple[str, ...], optional): Image extensions to look for, by default ('.png', '.jpg', '.jpeg').

    Returns:
        Dataframe of relative paths of the images.
    """

    data = [] 
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({'path': os.path.relpath(path, start=root), 'file': file})
    return pd.DataFrame(data)

def create_id(string_col: pd.Series) -> pd.Series:
    """Creates unique ids from string based on MD5 hash.

    Args:
        string_col (pd.Series): List of ids.

    Returns:
        List of encoded ids.
    """

    entity_id = string_col.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id

def bbox_segmentation(bbox: List[float]) -> List[float]:
    """Convert bounding box to segmentation.

    Args:
        bbox (List[float]): Bounding box in the form [x, y, w, h].

    Returns:
        Segmentation mask in the form [x1, y1, x2, y2, ...].
    """

    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3], bbox[0], bbox[1]]

def segmentation_bbox(segmentation: List[float]) -> List[float]:
    """Convert segmentation to bounding box.

    Args:
        segmentation (List[float]): Segmentation mask in the form [x1, y1, x2, y2, ...].

    Returns:
        Bounding box in the form [x, y, w, h].
    """

    x = segmentation[0::2]
    y = segmentation[1::2]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    return [x_min, y_min, x_max-x_min, y_max-y_min]

def is_annotation_bbox(
        segmentation: List[float],
        bbox: List[float],
        tol: float = 0
        ) -> bool:
    """Checks whether segmentation is bounding box.

    Args:
        segmentation (List[float]): Segmentation mask in the form [x1, y1, x2, y2, ...].
        bbox (List[float]): Bounding box in the form [x, y, w, h].
        tol (float, optional): Tolerance for difference.

    Returns:
        True if segmentation is bounding box within tolerance.
    """

    bbox_seg = bbox_segmentation(bbox)
    if len(segmentation) == len(bbox_seg):
        for x, y in zip(segmentation, bbox_seg):
            if abs(x-y) > tol:
                return False
    else:
        return False
    return True

class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path='.'):
    with ProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(archive, extract_path='.', delete=False):
    shutil.unpack_archive(archive, extract_path)
    if delete:
        os.remove(archive)

@contextmanager
def data_directory(dir):
    '''
    Changes context such that data directory is used as current work directory.
    Data directory is created if it does not exist.
    '''
    current_dir = os.getcwd()
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(current_dir)

def kaggle_download(command, exception_text='', required_file=None):
    try:
        os.system(f"kaggle {command}")
    except:
        raise Exception(exception_text)
    if required_file is not None and not os.path.exists(required_file):
        raise Exception(exception_text)

def gdown_download(url, archive, exception_text=''):
    import gdown
    gdown.download(url, archive, quiet=False)
    if not os.path.exists(archive):
        raise Exception(exception_text)
