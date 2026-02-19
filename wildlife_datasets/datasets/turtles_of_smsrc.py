import ast
import os

import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadINaturalist

summary = {
    "licenses": "",
    "licenses_url": "",
    "url": "",
    "publication_url": None,
    "cite": "",
    "animals": {"Green Sea Turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2025,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "1 year",
    "size": None,
}


class TurtlesOfSMSRC(DownloadINaturalist, WildlifeDataset):
    summary = summary
    project_id = "turtles-of-smsrc"
    metadata_fields = (
        "species_guess",
        "observed_on",
        "location",
    )

    def load_segmentation(self, df):
        cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        segmentation = pd.read_csv(f"{self.root}/segmentation.csv")
        df = pd.merge(df, segmentation, on="image_id", how="left")
        df["bbox"] = list(df[cols].to_numpy())
        df = df.drop(cols, axis=1)
        df = df.reset_index(drop=True)
        for _, df_id in df.groupby("image_id"):
            image_ids = [f"{image_id}_{i}" for i, image_id in enumerate(df_id["image_id"])]
            df.loc[df_id.index, "image_id"] = image_ids
        return df

    def create_catalogue(self, load_segmentation=False) -> pd.DataFrame:
        assert self.root is not None
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        df["image_id"] = df["observation_id"].astype(str) + "_" + df["photo_id"].astype(str)
        df["identity"] = df["observation_id"]
        mask = ~df["location"].isnull()
        df["latitute"] = df.loc[mask, "location"].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        df["longitude"] = df.loc[mask, "location"].apply(lambda x: ast.literal_eval(x)[1]).astype(float)
        df = df.drop(["location", "photo_id", "photo_url"], axis=1)
        df = df.rename(
            {"observation_id": "encounter_id", "species_guess": "species", "file_name": "path", "observed_on": "date"},
            axis=1,
        )
        if load_segmentation:
            df = self.load_segmentation(df)

        return self.finalize_catalogue(df)
