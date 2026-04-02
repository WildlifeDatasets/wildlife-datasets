import os

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from ..datasets import WildlifeDataset, utils


def load_segmentation(metadata: pd.DataFrame, file_name: str) -> pd.DataFrame:
    # Load segmentation
    segmentation = pd.read_csv(file_name)

    # Fix segmentation as a dict
    if "segmentation" in segmentation.columns:
        segmentation["segmentation"] = segmentation["segmentation"].apply(utils.parse_bbox_mask)

    # Merge metadata and segmentation (may result in nans in segmentations)
    cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    metadata = pd.merge(metadata, segmentation, on="image_id", how="left")
    metadata["bbox"] = metadata[cols].to_numpy().tolist()

    # Check that there is no image_id with two nans
    mask = metadata[cols].isnull().all(axis=1)
    max_n_image_id = metadata.loc[mask, "image_id"].value_counts().max()
    if max_n_image_id > 1:
        raise ValueError("There is image_id with multiple nan bounding boxes")

    # Generate new image_id
    cols_enhanced = ["image_id"] + cols
    new_image_id = metadata.loc[~mask, cols_enhanced].round(2).astype(str).agg("_".join, axis=1)
    new_image_id = utils.get_persistent_id(new_image_id)
    metadata.loc[~mask, "image_id"] = metadata.loc[~mask, "image_id"].astype(str) + "_" + new_image_id

    # Finalize the dataframe
    metadata = metadata.drop(cols, axis=1)
    metadata = metadata.reset_index(drop=True)
    return metadata


def run_detection(dataset: WildlifeDataset, model) -> pd.DataFrame:
    assert dataset.root is not None

    image_ids = np.empty(0, dtype=int)
    bboxes = np.empty((0, 4))
    rles = []
    scores = np.empty(0, dtype=float)
    labels = np.empty(0, dtype=int)
    names = []
    for _, df_row in tqdm(dataset.df.iterrows(), total=len(dataset)):
        file_name = os.path.join(dataset.root, df_row["path"])
        try:
            result = model.predict(source=file_name, verbose=False, save=False, show=False)[0]
            names = result.names
        except Exception:
            continue
        if result.masks is None:
            continue

        shape = result.orig_shape
        for box, mask in zip(result.boxes, result.masks.data):
            mask = mask.detach().cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask.astype("uint8"), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")

            image_ids = np.concatenate((image_ids, [df_row["image_id"]]))
            bboxes = np.concatenate((bboxes, box.xyxy.cpu().numpy()))
            rles.append(rle)
            scores = np.concatenate((scores, box.conf.cpu().numpy()))
            labels = np.concatenate((labels, box.cls.cpu().numpy()))

    if len(image_ids) == 0:
        return pd.DataFrame([])
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "bbox_x": bboxes[:, 0],
            "bbox_y": bboxes[:, 1],
            "bbox_w": bboxes[:, 2] - bboxes[:, 0],
            "bbox_h": bboxes[:, 3] - bboxes[:, 1],
            "segmentation": rles,
            "score": scores,
            "label": [names[x] for x in labels],
        }
    )
