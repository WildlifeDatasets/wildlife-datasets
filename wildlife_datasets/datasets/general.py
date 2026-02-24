import os

import pandas as pd

from .datasets import WildlifeDataset


class Dataset_Metadata(WildlifeDataset):
    def create_catalogue(
            self,
            file_name: str = "metadata.csv",
            load_segmentation: bool = False
            ) -> pd.DataFrame:
        
        assert self.root is not None
        metadata = pd.read_csv(os.path.join(self.root, file_name))
        metadata = self.modify_metadata(metadata)
        if load_segmentation:
            metadata = self.load_segmentation(metadata)
        return self.finalize_catalogue(metadata)

    def load_segmentation(self, metadata: pd.DataFrame) -> pd.DataFrame:
        assert self.root is not None
        cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        segmentation = pd.read_csv(os.path.join(self.root, "segmentation.csv"))
        metadata = pd.merge(metadata, segmentation, on="image_id", how="left")
        metadata["bbox"] = metadata[cols].to_numpy().tolist()
        metadata = metadata.drop(cols, axis=1)
        metadata = metadata.reset_index(drop=True)
        new_image_id = metadata.groupby("image_id").cumcount()
        metadata["image_id"] = metadata["image_id"].astype(str) + "_" + new_image_id.astype(str)
        return metadata
        
    def modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return metadata