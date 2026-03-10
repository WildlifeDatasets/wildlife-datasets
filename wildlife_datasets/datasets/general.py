import os

import pandas as pd

from .datasets import WildlifeDataset
from .utils import find_images, get_persistent_id
from .utils import load_segmentation as utils_load_segmentation


class Dataset_Folder(WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = find_images(self.root)
        df = pd.DataFrame({"identity": len(data) * ["unknown"], "path": data["path"] + os.path.sep + data["file"]})
        df["image_id"] = get_persistent_id(df["path"])
        return df


class Dataset_Metadata(WildlifeDataset):
    def create_catalogue(self, file_name: str = "metadata.csv", load_segmentation: bool = False) -> pd.DataFrame:

        assert self.root is not None
        metadata = pd.read_csv(os.path.join(self.root, file_name))
        metadata = self.modify_metadata(metadata)
        if load_segmentation:
            file_name = os.path.join(self.root, "segmentation.csv")
            metadata = utils_load_segmentation(metadata, file_name)
        return self.finalize_catalogue(metadata)

    def modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return metadata
