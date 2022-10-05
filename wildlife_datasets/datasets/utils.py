import os
import pandas as pd
import numpy as np
from typing import Tuple
import hashlib
from collections.abc import Iterable


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

def bbox_segmentation(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3], bbox[0], bbox[1]]

def segmentation_bbox(segmentation):
    x = segmentation[0::2]
    y = segmentation[1::2]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    return [x_min, y_min, x_max-x_min, y_max-y_min]

def is_annotation_bbox(ann, bbox, tol=0):
    bbox_ann = bbox_segmentation(bbox)
    if len(ann) == len(bbox_ann):
        for x, y in zip(ann, bbox_ann):
            if abs(x-y) > tol:
                return False
    else:
        return False
    return True

def convert_keypoint(keypoint, keypoints_names):
    keypoint_dict = {}
    if isinstance(keypoint, Iterable):
        for i in range(len(keypoints_names)):
            x = keypoint[2*i]
            y = keypoint[2*i+1]
            if np.isfinite(x) and np.isfinite(y):
                keypoint_dict[keypoints_names[i]] = [x, y]
    return keypoint_dict

def convert_keypoints(keypoints: pd.Series, keypoints_names):
    return [convert_keypoint(keypoint, keypoints_names) for keypoint in keypoints]
