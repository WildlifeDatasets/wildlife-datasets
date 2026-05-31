import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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

        Load body crops with pixel-space bbox columns:

        >>> annotated = datasets.Melops(root, load_image_size=True)
        >>> annotated.df[["bbox_body", "bbox_head", "bbox_headless"]].head()

        Load head and white-card keypoints when the annotation file is present:

        >>> pose = datasets.Melops(root, load_keypoints=True)
        >>> pose.df[["keypoints_head", "keypoints_white_card"]].head()

    Args:
        bbox: Optional part name used for the standard ``bbox`` column. Use
            ``"body"``, ``"head"``, or ``"headless"``. Setting this also reads
            image sizes and creates pixel-space ``bbox_body``, ``bbox_head``,
            and ``bbox_headless`` columns.
        drop_missing: The Zenodo body archive is missing one metadata image
            (``P8316190_2020``). When ``True``, missing body-image rows are
            dropped with a warning. When ``False``, loading fails on missing
            body images.
        load_image_size: Read body-crop image dimensions and add
            ``image_width`` and ``image_height``. This is needed for
            pixel-space bbox conversion.
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

    @staticmethod
    def _relative_image_path(row: pd.Series) -> str:
        if row["path"] == ".":
            return row["file"]
        return os.path.join(row["path"], row["file"])

    @staticmethod
    def _to_series(values: np.ndarray) -> pd.Series:
        return pd.Series(list(values))

    def _find_body_images(self) -> pd.DataFrame:
        assert self.root is not None
        images = utils.find_images(self.root)
        if len(images) == 0:
            raise FileNotFoundError(f"No body images found in {self.root}.")

        rel_path = images["path"].str.cat(images["file"], sep=os.path.sep).str.lower()
        body_images = images[rel_path.str.contains("body") & ~rel_path.str.contains("headless")]
        if len(body_images) > 0:
            images = body_images

        images = images.copy()
        images["filename_year"] = images["file"].apply(lambda x: os.path.splitext(x)[0])
        duplicates = images[images["filename_year"].duplicated()]["filename_year"].unique()
        if len(duplicates) > 0:
            raise ValueError(f"Duplicate Melops image names found, for example {duplicates[0]}.")

        images["path"] = images.apply(self._relative_image_path, axis=1)
        return images[["filename_year", "path"]]

    def _add_optional_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.root is not None

        bbox_path = self._find_file(self.root, "Melops_bbox_coords.txt")
        if bbox_path is not None:
            bboxes = pd.read_csv(bbox_path, sep="\t")
            df = df.merge(bboxes, on="filename_year", how="left")

        color_path = self._find_file(self.root, "colour_extraction_correction.csv")
        if color_path is not None:
            colors = pd.read_csv(color_path)
            colors["filename_year"] = colors["image"].apply(lambda x: os.path.splitext(x)[0])
            colors = colors.drop(columns=["image"])
            df = df.merge(colors, on="filename_year", how="left")

        return df

    def _add_source_bboxes(self, df: pd.DataFrame) -> None:
        for part in self.bbox_parts:
            cols = [f"{part}_xcenter", f"{part}_ycenter", f"{part}_width", f"{part}_height"]
            if all(col in df for col in cols):
                df[f"bbox_{part}_source"] = self._to_series(df[cols].to_numpy())

    def _check_bbox_columns(self, df: pd.DataFrame) -> None:
        missing = []
        for part in self.bbox_parts:
            missing.extend(
                col
                for col in [f"{part}_xcenter", f"{part}_ycenter", f"{part}_width", f"{part}_height"]
                if col not in df
            )
        if missing:
            raise FileNotFoundError(
                f"Melops_bbox_coords.txt is required for bbox conversion but these columns are missing: {missing}."
            )

    def _add_image_sizes(self, df: pd.DataFrame) -> None:
        assert self.root is not None

        def get_size(path):
            with Image.open(os.path.join(self.root, path)) as img:
                return img.size

        sizes = [get_size(path) for path in tqdm(df["path"], desc="Melops image sizes", mininterval=1, ncols=100)]
        df[["image_width", "image_height"]] = pd.DataFrame(sizes, index=df.index)

    @staticmethod
    def _bbox_to_body_crop(df: pd.DataFrame, part: str) -> np.ndarray:
        body = df[["body_xcenter", "body_ycenter", "body_width", "body_height"]].to_numpy()
        boxes = df[[f"{part}_xcenter", f"{part}_ycenter", f"{part}_width", f"{part}_height"]].to_numpy()
        image_size = df[["image_width", "image_height"]].to_numpy()

        body_top_left = body[:, :2] - body[:, 2:] / 2
        box_top_left = boxes[:, :2] - boxes[:, 2:] / 2
        top_left = (box_top_left - body_top_left) / body[:, 2:] * image_size
        size = boxes[:, 2:] / body[:, 2:] * image_size

        bottom_right = np.clip(top_left + size, 0, image_size)
        top_left = np.clip(top_left, 0, image_size)
        return np.column_stack([top_left, bottom_right - top_left])

    def _add_body_crop_bboxes(self, df: pd.DataFrame, bbox: str | None) -> None:
        self._check_bbox_columns(df)
        for part in self.bbox_parts:
            df[f"bbox_{part}"] = self._to_series(self._bbox_to_body_crop(df, part))
        if bbox is not None:
            if bbox not in self.bbox_parts:
                raise ValueError(f"bbox must be one of {self.bbox_parts} or None.")
            df["bbox"] = df[f"bbox_{bbox}"]

    @staticmethod
    def _keypoints_to_wide(keypoints: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
        keypoints = keypoints.copy()
        keypoints["filename_year"] = keypoints["image"].apply(lambda x: os.path.splitext(x)[0])
        keypoints = keypoints.dropna(subset=["keypoint_id"])
        keypoints = keypoints[keypoints["keypoint_id"].between(0, len(names) - 1)]

        records = []
        for filename_year, points in keypoints.groupby("filename_year"):
            values = np.full(2 * len(names), np.nan)
            for _, point in points.iterrows():
                idx = int(point["keypoint_id"])
                values[2 * idx : 2 * idx + 2] = [point["x_pred"], point["y_pred"]]
            records.append({"filename_year": filename_year, "keypoints": values})
        return pd.DataFrame(records)

    @staticmethod
    def _source_keypoints_to_body_crop(df: pd.DataFrame, keypoints: np.ndarray) -> np.ndarray:
        points = keypoints.reshape(len(df), -1, 2)
        source_size = df[["image_width", "image_height"]].to_numpy() / df[["body_width", "body_height"]].to_numpy()
        body_top_left = (
            df[["body_xcenter", "body_ycenter"]].to_numpy() - df[["body_width", "body_height"]].to_numpy() / 2
        )
        body_top_left = body_top_left * source_size
        points = points - body_top_left[:, None, :]
        return points.reshape(len(df), -1)

    def _add_keypoints(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.root is not None
        for file_name, names, column in [
            ("keypoint_head.csv", self.head_keypoint_names, "keypoints_head"),
            ("keypoint_white_card.csv", self.white_card_keypoint_names, "keypoints_white_card"),
        ]:
            path = self._find_file(self.root, file_name)
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
            df[f"{column}_source"] = self._to_series(values)
            df[column] = self._to_series(self._source_keypoints_to_body_crop(df, values))
        if "keypoints_head" in df:
            df["keypoints"] = df["keypoints_head"]
        return df

    def create_catalogue(
        self,
        bbox: str | None = None,
        drop_missing: bool = True,
        load_image_size: bool = False,
        load_keypoints: bool = False,
    ) -> pd.DataFrame:
        assert self.root is not None
        if bbox is not None and bbox not in self.bbox_parts:
            raise ValueError(f"bbox must be one of {self.bbox_parts} or None.")

        metadata_path = self._find_file(self.root, "Melops_metadata.txt")
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
        self._add_source_bboxes(df)
        if bbox is not None or load_image_size or load_keypoints:
            self._add_image_sizes(df)
            self._add_body_crop_bboxes(df, bbox)
        if load_keypoints:
            df = self._add_keypoints(df)

        df["image_id"] = df["filename_year"]
        df["identity"] = df["ID"].astype(str)
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y").dt.strftime("%Y-%m-%d")
        df["orientation"] = df["side"]
        df["species"] = "corkwing wrasse"
        df = df.rename(columns={"lat": "latitude", "lon": "longitude"})

        return self.finalize_catalogue(df)
