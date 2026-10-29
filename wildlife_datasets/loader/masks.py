from __future__ import annotations

import gzip
import json
import posixpath
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils


class MaskReader(Mapping):
    """Loads segmentation masks and their metadata.

    Masks are indexed by annotation ID and decoded on demand.
    The masks can be loaded from a local directory or downloaded
    from Hugging Face using `from_hub`.
    """

    def __init__(self, path):
        path = Path(path)
        with gzip.open(path / "masks.json.gz", "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format") != "cfr-mask-rle-v1":
            raise ValueError("Expected a cfr-mask-rle-v1 release file.")
        self._records = {}
        for record in payload["annotations"]:
            key = str(record["annotation_id"])
            if key in self._records:
                raise ValueError(f"Duplicate mask annotation ID: {key}")
            self._records[key] = record
        annotations = pd.read_parquet(path / "metadata.parquet")
        required = {
            "annotation_id",
            "source_image",
            "source_size_wh",
            "source_crop_xyxy",
            "source_transform",
            "input_size_wh",
            "accepted",
        }
        if not required.issubset(annotations.columns):
            raise ValueError(f"Mask metadata is missing columns: {sorted(required - set(annotations.columns))}")
        annotations["annotation_id"] = annotations["annotation_id"].astype(str)
        if not annotations["annotation_id"].is_unique:
            raise ValueError("Duplicate mask annotation ID in metadata.parquet.")
        self.annotations = annotations.set_index("annotation_id", drop=False)
        if set(self.annotations.index) != set(self._records):
            raise ValueError("Mask and metadata annotation IDs differ; use files from the same release revision.")
        accepted = [self.is_present(key) for key in self.annotations.index]
        if not np.array_equal(self.annotations["accepted"].to_numpy(), accepted):
            raise ValueError("Mask and metadata acceptance flags differ.")
        self.summary = json.loads((path / "summary.json").read_text())
        self.revision = None

    @classmethod
    def from_hub(cls, dataset_name, *, revision="main", cache_dir=None, local_files_only=False):
        """Loads segmentation masks from Hugging Face.

        Args:
            dataset_name (str): Name of the dataset.
            revision (str, optional): Repository revision or commit hash.
            cache_dir (str or Path, optional): Directory for cached files.
            local_files_only (bool, optional): Whether to use only cached files.

        Returns:
            MaskReader or None: Reader containing the available masks,
                or None if no mask collection is available.
        """

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

        if (
            not isinstance(dataset_name, str)
            or not dataset_name
            or dataset_name in (".", "..")
            or any(separator in dataset_name for separator in ("/", "\\", "*", "?", "[", "]"))
        ):
            raise ValueError("dataset_name must be a published dataset name, such as 'NyalaData'.")
        paths, missing = [], []
        for name in ("masks.json.gz", "metadata.parquet", "summary.json"):
            try:
                path = Path(
                    hf_hub_download(
                        repo_id="ankitkupadhyay/cfr-wildlife-masks",
                        filename=f"data/{dataset_name}/{name}",
                        repo_type="dataset",
                        revision=revision,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                )
            except LocalEntryNotFoundError:
                if local_files_only:
                    return None
                # Offline download failures still surface during preparation.
                raise
            except EntryNotFoundError:
                missing.append(name)
                continue
            # Pin the remaining files to the first file's cached revision.
            revision = path.parents[2].name
            paths.append(path)
        if not paths:
            return None
        if missing:
            raise FileNotFoundError(f"Incomplete published mask release for {dataset_name}: {missing}")
        reader = cls(paths[0].parent)
        reader.revision = revision
        return reader

    def __getitem__(self, key):
        record = self._records[str(key)]
        if not self.is_present(key):
            return None
        result = mask_utils.decode(record["segmentation"])
        if result.ndim != 2:
            raise ValueError(f"Expected one 2-D mask for annotation {key}.")
        return result.astype(bool)

    def __contains__(self, key):
        return str(key) in self._records

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def is_present(self, key):
        """Checks whether an annotation has an available segmentation mask.

        Args:
            key (str): Annotation ID.

        Returns:
            bool: Whether an accepted mask is available.
        """

    def match(self, catalogue, *, path_col="path"):
        """Matches dataset records to segmentation mask annotations.

        Matching is based on image paths and, when necessary, bounding boxes.

        Args:
            catalogue (pd.DataFrame): Dataset records to match.
            path_col (str, optional): Column containing image paths.

        Returns:
            list: Annotation IDs corresponding to dataset records.
                Unmatched records are represented by None.
        """
        by_filename = defaultdict(list)
        for key, row in self.annotations.iterrows():
            if pd.notna(row["source_image"]):
                source = _path(row["source_image"])
                by_filename[source.rsplit("/", 1)[-1]].append((key, source))
        ids = []
        for _, row in catalogue.iterrows():
            path = _path(row[path_col])
            entries = by_filename[path.rsplit("/", 1)[-1]]
            exact = [key for key, source in entries if path == source]
            candidates = exact or [
                key for key, source in entries if path.endswith("/" + source) or source.endswith("/" + path)
            ]
            if len(candidates) > 1:
                bbox = _vector(row.get("bbox"), 4)
                if bbox is not None:
                    box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    candidates = [
                        key
                        for key in candidates
                        if (source_box := _vector(self.annotations.loc[key, "source_crop_xyxy"], 4)) is not None
                        and np.allclose(box, source_box, atol=1, rtol=0)
                    ]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous mask match for {path}; check the source path and bounding box.")
            ids.append(candidates[0] if candidates else None)
        return ids

    def _source_mask(self, key, image_size):
        """Restores a segmentation mask to the original image coordinates.

        Args:
            key (str): Annotation ID.
            image_size (tuple): Original image dimensions (width, height).

        Returns:
            np.ndarray or None: Boolean mask aligned with the original image.
        """

        mask = self[key]
        if mask is None:
            return None
        info = self.annotations.loc[str(key)]
        input_size = _vector(info["input_size_wh"], 2)
        if input_size is None or tuple(input_size.astype(int)) != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"Mask grid and input_size_wh differ for {key}.")
        source_size = _vector(info["source_size_wh"], 2)
        if source_size is not None and tuple(source_size.astype(int)) != image_size:
            raise ValueError(f"Source image size {image_size} differs from published {tuple(source_size)} for {key}.")
        box = _vector(info["source_crop_xyxy"], 4)
        if box is None:
            if info["source_transform"] != "resize_full_dataset_image":
                raise ValueError(f"No source crop geometry for {key}; cannot safely align this mask.")
            box = np.array([0, 0, *image_size])
        x0, y0, x1, y1 = np.rint(box).astype(int)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid source crop for {key}.")
        restored = np.asarray(Image.fromarray(mask).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST), dtype=bool)
        width, height = image_size
        left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if right <= left or bottom <= top:
            raise ValueError(f"Source crop does not overlap the image for {key}.")
        canvas = np.zeros((height, width), dtype=bool)
        canvas[top:bottom, left:right] = restored[top - y0 : bottom - y0, left - x0 : right - x0]
        return canvas


def _path(value):
    return posixpath.normpath(str(value).replace("\\", "/"))


def _vector(value, length):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    values = np.asarray(value, dtype=float)
    return values if values.shape == (length,) and np.isfinite(values).all() else None


def _has_segmentation(value):
    """Checks whether a segmentation annotation is available.

    Args:
        value: Segmentation annotation.

    Returns:
        bool: Whether the annotation contains segmentation data.
    """
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        return _has_segmentation(json.loads(value))
    if isinstance(value, np.ndarray):
        return bool(value.size)
    if isinstance(value, (list, tuple, dict, str, Path)):
        return bool(value)
    return bool(pd.notna(value))


def _decode_segmentation(segmentation, image_size, *, root=None):
    """Decodes a segmentation annotation into a Boolean mask.

    Supports polygons, COCO RLE, mask images, and NumPy arrays.

    Args:
        segmentation: Segmentation annotation to decode.
        image_size (tuple): Image dimensions (width, height).
        root (str or Path, optional): Root directory for mask files.

    Returns:
        np.ndarray or None: Boolean mask aligned with the image,
            or None if no segmentation is available.
    """
    if not _has_segmentation(segmentation):
        return None
    if isinstance(segmentation, str) and segmentation.lstrip().startswith(("[", "{")):
        return _decode_segmentation(json.loads(segmentation), image_size, root=root)
    width, height = image_size
    if isinstance(segmentation, (str, Path)):
        path = Path(root or ".") / segmentation
        with Image.open(path) as image:
            mask = np.array(ImageOps.exif_transpose(image))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    elif isinstance(segmentation, np.ndarray) and segmentation.ndim == 2:
        mask = segmentation
        if not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Native array masks must be binary (0/1 or 0/255).")
    elif isinstance(segmentation, dict):
        if tuple(segmentation.get("size", ())) != (height, width):
            raise ValueError("Native mask size differs from the source image size.")
        rle = segmentation
        if isinstance(rle.get("counts"), (list, np.ndarray)):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)
    elif isinstance(segmentation, (list, tuple, np.ndarray)):
        polygons = segmentation
        if np.isscalar(polygons[0]):
            polygons = [polygons]
        if any(len(polygon) < 6 or len(polygon) % 2 or not np.isfinite(polygon).all() for polygon in polygons):
            raise ValueError("Native polygons need at least three finite (x, y) points.")
        rles = mask_utils.frPyObjects([list(polygon) for polygon in polygons], height, width)
        mask = mask_utils.decode(mask_utils.merge(rles))
    else:
        raise ValueError("Unrecognized native segmentation; use polygons, COCO RLE, a mask path, or a 2-D array.")
    if mask.shape != (height, width):
        raise ValueError(f"Native mask shape {mask.shape} differs from source image shape {(height, width)}.")
    return mask.astype(bool)


def _load_mask(dataset, idx, *, image=None):
    """Loads a segmentation mask for a given image.

    Native annotations are preferred over generated masks.

    Args:
        dataset (WildlifeDataset): Dataset containing the image.
        idx (int): Position of the image in the dataset.
        image (PIL.Image.Image, optional): Previously loaded image.

    Returns:
        np.ndarray or None: Boolean mask aligned with the image,
            or None if no mask is available.
    """
    row = dataset.df.iloc[idx]
    native = row.get("segmentation")
    has_native = _has_segmentation(native)
    if not has_native:
        if dataset._mask_reader is None:
            return None
        key = row["mask_id"]
        if pd.isna(key) or not dataset._mask_reader.is_present(key):
            return None
    if image is None:
        image = dataset.get_image(idx)
    if has_native:
        return _decode_segmentation(native, image.size, root=dataset.root)
    return dataset._mask_reader._source_mask(key, image.size)
