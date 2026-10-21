import os

import pandas as pd

from ..detection import load_segmentation as utils_load_segmentation
from .datasets import WildlifeDataset
from .utils import find_images, get_persistent_id


class Dataset_Folder(WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        root = self.get_root()
        data = find_images(root)
        df = pd.DataFrame({"identity": len(data) * ["unknown"], "path": data["path"] + os.path.sep + data["file"]})
        df["image_id"] = get_persistent_id(df["path"])
        df = self.modify_catalogue(df)
        return self.finalize_catalogue(df)

    def modify_catalogue(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class Dataset_FolderStructured(WildlifeDataset):
    def create_catalogue(self, levels: tuple[str | None, ...] = ("identity",)) -> pd.DataFrame:
        root = self.get_root()
        data = find_images(root)
        folders = data["path"].str.split(os.path.sep, expand=True)
        depth = folders.notna().sum(axis=1).where(data["path"] != ".", 0)
        wrong_depth = depth != len(levels)
        if wrong_depth.any():
            examples = (data["path"][wrong_depth] + os.path.sep + data["file"][wrong_depth]).head(3).tolist()
            raise ValueError(
                f"Images must be exactly {len(levels)} folder(s) deep to match levels {levels}, e.g. {examples}"
            )

        df = pd.DataFrame({"path": data["path"] + os.path.sep + data["file"]})
        for i, level in enumerate(levels):
            if level is not None:
                df[level] = folders[i]
        if "identity" not in df.columns:
            df["identity"] = "unknown"
        df["image_id"] = get_persistent_id(df["path"])
        df = self.modify_catalogue(df)
        return self.finalize_catalogue(df)

    def modify_catalogue(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class Dataset_Metadata(WildlifeDataset):
    def create_catalogue(self, file_name: str = "metadata.csv", load_segmentation: bool = False) -> pd.DataFrame:

        root = self.get_root()
        metadata = pd.read_csv(os.path.join(root, file_name))
        metadata = self.modify_catalogue(metadata)
        if load_segmentation:
            file_name = os.path.join(root, "segmentation.csv")
            metadata = utils_load_segmentation(metadata, file_name)
        return self.finalize_catalogue(metadata)

    def modify_catalogue(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return metadata
