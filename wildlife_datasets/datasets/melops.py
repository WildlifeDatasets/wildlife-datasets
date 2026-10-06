import os
import warnings

import numpy as np
import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://zenodo.org/records/17404087",
    "publication_url": "https://doi.org/10.1038/s41597-026-07045-1",
    "cite": "sordalen2026melops",
    "animals": {"corkwing wrasse"},
    "animals_simple": "fish",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": 24578,
    "reported_n_individuals": 9861,
    "wild": True,
    "clear_photos": True,
    "pose": "double",
    "unique_pattern": True,
    "from_video": False,
    "cropped": True,
    "span": "7 years",
    "size": 27700,
}


class Melops(DownloadURL, WildlifeDataset):
    """Melops body-crop dataset from Zenodo.

    The Zenodo record contains full images, head crops, body crops, headless
    crops, annotation files, and color-analysis outputs. This wrapper downloads
    only the body-crop archive plus the small metadata/annotation files. Body
    crops are the default image returned by ``Melops(root)``.

    Examples:
        Download and load body crops:

        >>> from wildlife_datasets import datasets
        >>> root = "/data/wildlife_datasets/data/Melops"
        >>> datasets.Melops.get_data(root)
        >>> dataset = datasets.Melops(root)

        Load head crops on the fly from the body images:

        >>> heads = datasets.Melops(root, bbox="head", img_load="bbox")
        >>> img = heads[0]

        Load headless body crops on the fly:

        >>> headless = datasets.Melops(root, bbox="headless", img_load="bbox")

        Load head and white-card keypoints when the annotation file is present:

        >>> pose = datasets.Melops(root, load_keypoints=True)
        >>> pose.df[["keypoints_head", "keypoints_white_card"]].head()

    Args:
        bbox: Optional part name used for the standard ``bbox`` column. Use
            ``"body"``, ``"head"``, or ``"headless"``. The bounding box is given
            in pixels of the body crop.
        drop_missing: The Zenodo body archive is missing one metadata image
            (``P8316190_2020``). When ``True``, missing body-image rows are
            dropped with a warning. When ``False``, loading fails on missing
            body images.
        load_keypoints: Read ``keypoint_head.csv`` and
            ``keypoint_white_card.csv`` when available. Source keypoints remain
            available as ``*_source`` columns; transformed keypoints are in the
            body-crop coordinate system.
    """

    summary = summary
    bbox_parts = ("body", "head", "headless")
    head_keypoint_names = (
        "snout",
        "bottom_opercular",
        "eyes_snout",
        "eyes_opercular",
        "pectoral_bottom",
        "pectoral_top",
        "papilla",
        "caudal_bottom",
        "caudal_top",
        "black_point",
        "eyes_pectoral",
        "eyes_top",
        "pelvic",
        "top_opercular",
        "medium_opercular",
        "anal_fin",
    )
    white_card_keypoint_names = ("stand_1", "stand_2", "stand_3", "stand_4")
    bbox_fields = ("xcenter", "ycenter", "width", "height")
    keypoint_files = (
        ("keypoint_head.csv", head_keypoint_names, "keypoints_head"),
        ("keypoint_white_card.csv", white_card_keypoint_names, "keypoints_white_card"),
    )
    downloads = [
        ("https://zenodo.org/records/17404087/files/Melops_body.zip?download=1", "Melops_body.zip"),
        ("https://zenodo.org/records/17404087/files/Melops_metadata.txt?download=1", "Melops_metadata.txt"),
        (
            "https://zenodo.org/records/17404087/files/Image%20manipulation%20and%20management.zip?download=1",
            "Image manipulation and management.zip",
        ),
    ]

    @staticmethod
    def _find_file(root: str, file_name: str) -> str | None:
        for path, _, files in os.walk(root):
            if file_name in files:
                return os.path.join(path, file_name)
        return None

    @classmethod
    def _bbox_columns(cls, part: str) -> list[str]:
        return [f"{part}_{field}" for field in cls.bbox_fields]

    @staticmethod
    def _array_series(values: np.ndarray, index: pd.Index) -> pd.Series:
        return pd.Series(list(values), index=index)

    def _find_body_images(self) -> pd.DataFrame:
        root = self.get_root()
        images = utils.find_images(root)
        if len(images) == 0:
            raise FileNotFoundError(f"No body images found in {root}.")

        rel_path = images["path"].str.cat(images["file"], sep=os.path.sep).str.lower()
        body_images = images[rel_path.str.contains("body") & ~rel_path.str.contains("headless")]
        if len(body_images) > 0:
            images = body_images

        images = images.copy()
        images["filename_year"] = images["file"].apply(lambda x: os.path.splitext(x)[0])
        duplicates = images[images["filename_year"].duplicated()]["filename_year"].unique()
        if len(duplicates) > 0:
            raise ValueError(f"Duplicate Melops image names found, for example {duplicates[0]}.")

        images["path"] = [
            file_name if path == "." else os.path.join(path, file_name)
            for path, file_name in zip(images["path"], images["file"])
        ]
        return images[["filename_year", "path"]]

    def _add_optional_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        root = self.get_root()

        bbox_path = self._find_file(root, "Melops_bbox_coords.txt")
        if bbox_path is not None:
            bboxes = pd.read_csv(bbox_path, sep="\t")
            df = df.merge(bboxes, on="filename_year", how="left")
            for part in self.bbox_parts:
                cols = self._bbox_columns(part)
                if all(col in df for col in cols):
                    df[f"bbox_{part}_source"] = self._array_series(df[cols].to_numpy(), df.index)

        return df

    @classmethod
    def _original_size(cls, df: pd.DataFrame) -> np.ndarray:
        # head_w and head_h are the head bbox sizes in pixels of the original image
        return df[["head_w", "head_h"]].to_numpy() / df[["head_width", "head_height"]].to_numpy()

    @classmethod
    def _bbox_to_body_crop(cls, df: pd.DataFrame, part: str) -> np.ndarray:
        body = df[cls._bbox_columns("body")].to_numpy()
        boxes = df[cls._bbox_columns(part)].to_numpy()
        original_size = cls._original_size(df)
        crop_size = body[:, 2:] * original_size

        body_top_left = body[:, :2] - body[:, 2:] / 2
        box_top_left = boxes[:, :2] - boxes[:, 2:] / 2
        top_left = (box_top_left - body_top_left) * original_size
        size = boxes[:, 2:] * original_size

        bottom_right = np.clip(top_left + size, 0, crop_size)
        top_left = np.clip(top_left, 0, crop_size)
        return np.column_stack([top_left, bottom_right - top_left])

    def _check_bbox_columns(self, df: pd.DataFrame) -> None:
        missing = [col for part in self.bbox_parts for col in self._bbox_columns(part) if col not in df]
        if missing:
            raise FileNotFoundError(
                f"Melops_bbox_coords.txt is required for bbox conversion but these columns are missing: {missing}."
            )

    @staticmethod
    def _keypoints_to_wide(keypoints: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
        keypoints = keypoints.copy()
        keypoints["filename_year"] = keypoints["image"].apply(lambda x: os.path.splitext(x)[0])
        keypoints = keypoints.dropna(subset=["keypoint_id"])
        keypoints["keypoint_id"] = keypoints["keypoint_id"].astype(int)
        keypoints = keypoints[keypoints["keypoint_id"].between(0, len(names) - 1)]
        if len(keypoints) == 0:
            return pd.DataFrame(columns=["filename_year", "keypoints"])

        columns = pd.MultiIndex.from_product([range(len(names)), ["x_pred", "y_pred"]])
        keypoints = keypoints.pivot_table(
            index="filename_year",
            columns="keypoint_id",
            values=["x_pred", "y_pred"],
            aggfunc="last",
        )
        keypoints = keypoints.swaplevel(axis=1).reindex(columns=columns)
        return pd.DataFrame({"filename_year": keypoints.index, "keypoints": list(keypoints.to_numpy())})

    @classmethod
    def _source_keypoints_to_body_crop(cls, df: pd.DataFrame, keypoints: np.ndarray) -> np.ndarray:
        points = keypoints.reshape(len(df), -1, 2)
        body = df[cls._bbox_columns("body")].to_numpy()
        body_top_left = (body[:, :2] - body[:, 2:] / 2) * cls._original_size(df)
        points = points - body_top_left[:, None, :]
        return points.reshape(len(df), -1)

    def _add_keypoints(self, df: pd.DataFrame) -> pd.DataFrame:
        root = self.get_root()
        for file_name, names, column in self.keypoint_files:
            path = self._find_file(root, file_name)
            if path is None:
                continue
            keypoints = pd.read_csv(path)
            keypoints = self._keypoints_to_wide(keypoints, names)
            keypoints = keypoints.rename(columns={"keypoints": f"{column}_source"})
            df = df.merge(keypoints, on="filename_year", how="left")

            values = np.full((len(df), 2 * len(names)), np.nan)
            found = df[f"{column}_source"].apply(lambda x: isinstance(x, np.ndarray))
            if found.any():
                values[found] = np.stack(df.loc[found, f"{column}_source"].to_numpy())
            df[f"{column}_source"] = self._array_series(values, df.index)
            df[column] = self._array_series(self._source_keypoints_to_body_crop(df, values), df.index).apply(
                lambda kp: utils.keypoints_to_dict(kp, names, values_per_point=2)
            )
        if "keypoints_head" in df:
            df["keypoints"] = df["keypoints_head"]
        return df

    def create_catalogue(
        self,
        bbox: str | None = None,
        drop_missing: bool = True,
        load_keypoints: bool = False,
    ) -> pd.DataFrame:
        root = self.get_root()
        if bbox is not None and bbox not in self.bbox_parts:
            raise ValueError(f"bbox must be one of {self.bbox_parts} or None.")

        metadata_path = self._find_file(root, "Melops_metadata.txt")
        if metadata_path is None:
            raise FileNotFoundError("Could not find Melops_metadata.txt.")

        df = pd.read_csv(metadata_path, sep="\t")
        df = df.merge(self._find_body_images(), on="filename_year", how="left")
        missing = df["path"].isnull()
        if missing.any():
            examples = df.loc[missing, "filename_year"].head(3).tolist()
            message = f"Missing {missing.sum()} Melops body images, for example {examples}."
            if not drop_missing:
                raise FileNotFoundError(message)
            message += " These rows will be dropped."
            warnings.warn(message)
            df = df[~missing].reset_index(drop=True)

        df = self._add_optional_annotations(df)
        if bbox is not None or load_keypoints:
            self._check_bbox_columns(df)
        if bbox is not None:
            df["bbox"] = self._array_series(self._bbox_to_body_crop(df, bbox), df.index)
        if load_keypoints:
            df = self._add_keypoints(df)
        bbox_columns = [col for part in self.bbox_parts for col in self._bbox_columns(part)]
        bbox_columns += [f"bbox_{part}_source" for part in self.bbox_parts] + ["head_w", "head_h"]
        df = df.drop(columns=[col for col in bbox_columns if col in df])

        df["identity"] = df["ID"].astype(str)
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y").dt.strftime("%Y-%m-%d")
        df["species"] = "corkwing wrasse"
        df["blurry"] = df["blurry"].map({0: 1, 1: 2})
        df = df.drop(columns=["ID", "year"])
        df = df.rename(
            columns={
                "filename_year": "image_id",
                "side": "orientation",
                "lat": "latitude",
                "lon": "longitude",
                "sex": "gender",
                "trapID": "camera",
                "area": "location",
                "blurry": "image_quality",
            }
        )

        return self.finalize_catalogue(df)
