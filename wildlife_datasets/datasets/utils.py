import os
import pandas as pd
import numpy as np
from typing import Tuple
import hashlib
from collections.abc import Iterable, List, Dict


def find_images(
    root: str,
    img_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    ) -> pd.DataFrame:
    '''
    Find all image files in folder recursively based on img_extensions. 
    Save filename and relative path from root.
    '''
    data = [] 
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({'path': os.path.relpath(path, start=root), 'file': file})
    return pd.DataFrame(data)

def create_id(string_col: pd.Series) -> pd.Series:
    '''
    Creates unique id from string based on MD5 hash.
    '''
    entity_id = string_col.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id

def bbox_segmentation(bbox: List[float]) -> List[float]:
    '''
    Converts a bounding box into a segmentation mask.
    '''
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3], bbox[0], bbox[1]]

def segmentation_bbox(segmentation: List[float, ...]) -> List[float]:
    '''
    Converts a segmentation mask into a bounding box.
    '''
    x = segmentation[0::2]
    y = segmentation[1::2]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    return [x_min, y_min, x_max-x_min, y_max-y_min]

def is_annotation_bbox(
    segmentation: List[float, ...],
    bbox: List[float],
    tol: float = 0
    ) -> bool:
    '''
    Checks whether a segmentation mask is a boundign box.
    '''
    bbox_seg = bbox_segmentation(bbox)
    if len(segmentation) == len(bbox_seg):
        for x, y in zip(segmentation, bbox_seg):
            if abs(x-y) > tol:
                return False
    else:
        return False
    return True

def convert_keypoint(
    keypoint: List[float, ...],
    keypoints_names: List[str, ...]
    ) -> Dict[str, object]:
    '''
    Converts list of keypoints into a dictionary named by keypoint_names.
    '''
    keypoint_dict = {}
    if isinstance(keypoint, Iterable):
        for i in range(len(keypoints_names)):
            x = keypoint[2*i]
            y = keypoint[2*i+1]
            if np.isfinite(x) and np.isfinite(y):
                keypoint_dict[keypoints_names[i]] = [x, y]
    return keypoint_dict

def convert_keypoints(
    keypoints: pd.Series,
    keypoints_names: List[str, ...]
    ) -> List[Dict[str, object], ...]:
    '''
    Converts dataframe of lists of keypoints into a dictionary named by keypoint_names.
    '''
    return [convert_keypoint(keypoint, keypoints_names) for keypoint in keypoints]